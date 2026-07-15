"""External tool scan connector for Garak, PyRIT, and DeepTeam."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from core.config import get_settings
from core.logging import get_logger
from core.schemas import TargetConfig
from engine.deepteam_catalog import (
    DEEPTEAM_CREDENTIAL_FIELDS,
    DEEPTEAM_CREDENTIAL_REQUIRED,
    DEEPTEAM_CREDENTIAL_SCHEMA,
    DEEPTEAM_MULTI_TURN_ATTACKS,
    DEEPTEAM_PROVIDERS,
    DEEPTEAM_SECRET_CREDENTIAL_KEYS,
    DEEPTEAM_SINGLE_TURN_ATTACKS,
    DEEPTEAM_VULNERABILITIES,
    DEEPTEAM_VULNERABILITY_CATEGORIES,
)

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOL_REPORT_DIR = ROOT_DIR / "reports" / "tool-scans"

GARAK_TARGET_TYPES = (
    "ollama",
    "openai",
    "azure",
    "groq",
    "huggingface",
    "bedrock",
    "litellm",
    "mistral",
    "rest",
    "websocket",
    "rasa",
    "cohere",
    "replicate",
    "watsonx",
    "nim",
    "nvcf",
    "ggml",
    "langchain",
    "langchain_serve",
)

GARAK_PROBE_PRESETS: dict[str, tuple[str, str]] = {
    "owasp": ("--probe_tags", "owasp"),
    "all": ("--probes", "all"),
    "prompt_injection": ("--probes", "promptinject.HijackHateHumans"),
    "dan": ("--probes", "dan.DanInTheWild"),
    "encoding": ("--probes", "encoding.InjectBase64"),
    "lmrc": ("--probes", "lmrc.Profanity"),
    "malwaregen": ("--probes", "malwaregen.TopLevel"),
    "sys_prompt": ("--probes", "sysprompt_extraction.SystemPromptExtraction"),
}

GARAK_COMMON_BUFFS = ("lowercase", "paraphrase.Fast", "paraphrase.PegasusT5")

# Single source of truth for per-target-type Garak credentials/config: the allowed env
# vars for each garak_target_type, whether each is required, and whether it's secret
# (secret values are redacted from captured stdout/stderr and never logged). Adding a
# new provider is one entry here - no new Pydantic fields or frontend JSX required.
GARAK_CREDENTIAL_FIELDS: dict[str, list[dict[str, Any]]] = {
    "azure": [
        {"key": "AZURE_MODEL_NAME", "label": "Azure model name", "placeholder": "gpt-4o, gpt-4o-mini, gpt-35-turbo...", "secret": False, "required": True},
        {"key": "AZURE_ENDPOINT", "label": "Azure endpoint", "placeholder": "https://your-resource.openai.azure.com/", "secret": False, "required": True},
        {"key": "AZURE_API_VERSION", "label": "API version", "placeholder": "2024-06-01", "secret": False, "required": False},
        {"key": "AZURE_API_KEY", "label": "Azure API key", "placeholder": "Azure OpenAI API key", "secret": True, "required": True},
    ],
    "bedrock": [
        {"key": "AWS_ACCESS_KEY_ID", "label": "AWS access key ID", "placeholder": "AKIA...", "secret": True, "required": True},
        {"key": "AWS_SECRET_ACCESS_KEY", "label": "AWS secret access key", "placeholder": "", "secret": True, "required": True},
        {"key": "AWS_SESSION_TOKEN", "label": "AWS session token (optional, for temporary/STS credentials)", "placeholder": "", "secret": True, "required": False},
        {"key": "AWS_REGION", "label": "AWS region", "placeholder": "us-east-1", "secret": False, "required": True},
    ],
    "ollama": [
        {"key": "OLLAMA_HOST", "label": "Ollama host", "placeholder": "http://localhost:11434", "secret": False, "required": False},
    ],
}
GARAK_CREDENTIAL_SCHEMA: dict[str, tuple[str, ...]] = {
    target_type: tuple(field["key"] for field in fields) for target_type, fields in GARAK_CREDENTIAL_FIELDS.items()
}
GARAK_CREDENTIAL_REQUIRED: dict[str, tuple[str, ...]] = {
    target_type: tuple(field["key"] for field in fields if field["required"]) for target_type, fields in GARAK_CREDENTIAL_FIELDS.items()
}
GARAK_SECRET_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    field["key"] for fields in GARAK_CREDENTIAL_FIELDS.values() for field in fields if field["secret"]
)


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    name: str
    executable: str
    purpose: str
    profiles: tuple[str, ...]
    install_hint: str


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "garak": ToolDefinition(
        id="garak",
        name="Garak",
        executable="garak",
        purpose="Probe LLM applications with model and plugin based vulnerability checks.",
        profiles=("quick", "standard", "deep"),
        install_hint="Install Garak in the backend environment, then restart the API process.",
    ),
    "pyrit": ToolDefinition(
        id="pyrit",
        name="PyRIT",
        executable="pyrit_scan",
        purpose="Run prompt orchestration, scoring, and red-team attack workflows.",
        profiles=("quick", "standard", "deep"),
        install_hint="Install PyRIT in the backend environment, then restart the API process.",
    ),
    "deepteam": ToolDefinition(
        id="deepteam",
        name="DeepTeam",
        executable="deepteam",
        purpose="Evaluate adversarial prompts and model safety behavior across attack categories.",
        profiles=("quick", "standard", "deep"),
        install_hint="Install DeepTeam in the backend environment, then restart the API process.",
    ),
}


class ToolScanRequest(BaseModel):
    tool_id: str = Field(pattern="^(garak|pyrit|deepteam)$")
    target: TargetConfig | None = None
    # References a saved scan_targets row instead of embedding fresh credentials in
    # this request - see engine/tool_scan_jobs.py::submit_scan_job, which resolves
    # and decrypts the saved target server-side and fills the tool-specific
    # credential fields below before this request is ever persisted or queued.
    target_id: str | None = None
    profile: str = Field(default="standard", pattern="^(quick|standard|deep)$")
    options: str = ""
    # 600s (10min) default and 3600s (1hr) ceiling match worker/celery_app.py's
    # task_time_limit=3600 - real probes/scenarios vary a lot in duration (number of
    # generations, probe count, target latency), and the old 120s default/1800s
    # ceiling cut off scans well before the Celery task itself would have timed out.
    timeout_seconds: int = Field(default=600, ge=5, le=3600)
    dry_run: bool = False
    garak_target_type: str = "ollama"
    garak_target_name: str = ""
    garak_probe_mode: Literal["owasp", "all", "prompt_injection", "dan", "encoding", "lmrc", "malwaregen", "sys_prompt", "custom_probe", "custom_tag"] = "owasp"
    garak_probe_value: str = ""
    garak_buffs: list[str] = Field(default_factory=list)
    garak_credentials: dict[str, str] = Field(default_factory=dict)
    pyrit_scenario: str = ""
    pyrit_target_type: str = "openai_chat"
    pyrit_attack_mode: Literal["single_turn", "multi_turn"] = "single_turn"
    pyrit_initializers: list[str] = Field(default_factory=lambda: ["target", "load_default_datasets"])
    pyrit_strategies: list[str] = Field(default_factory=list)
    pyrit_endpoint_url: str = ""
    pyrit_deployment_name: str = ""
    pyrit_model_name: str = ""
    pyrit_api_key: str = ""
    pyrit_max_concurrency: int | None = None
    pyrit_max_retries: int | None = None
    pyrit_max_dataset_size: int | None = None
    pyrit_dataset_names: list[str] = Field(default_factory=list)
    pyrit_memory_labels: str | None = None
    deepteam_purpose: str = ""
    deepteam_provider: str = "ollama"
    deepteam_model: str = ""
    deepteam_temperature: float = 1.0
    deepteam_credentials: dict[str, str] = Field(default_factory=dict)
    # [{"name": "Bias", "types": ["religion"]}, ...] - see engine/deepteam_catalog.py
    deepteam_vulnerabilities: list[dict[str, Any]] = Field(default_factory=list)
    deepteam_attacks: list[str] = Field(default_factory=list)
    deepteam_attacks_per_vulnerability_type: int = Field(default=1, ge=1, le=10)
    deepteam_max_concurrent: int = Field(default=8, ge=1, le=50)

    @field_validator("garak_target_type")
    @classmethod
    def validate_garak_target_type(cls, value: str) -> str:
        if value not in GARAK_TARGET_TYPES:
            raise ValueError(f"Unsupported Garak target type: {value}")
        return value

    @model_validator(mode="after")
    def validate_garak_credentials(self) -> "ToolScanRequest":
        if self.tool_id != "garak":
            return self
        if self.target_id:
            # Credentials come from the saved target, resolved server-side after this
            # validator runs (submit_scan_job) - nothing to check yet.
            return self
        allowed = set(GARAK_CREDENTIAL_SCHEMA.get(self.garak_target_type, ()))
        unknown = set(self.garak_credentials) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported credential field(s) for target type '{self.garak_target_type}': {', '.join(sorted(unknown))}."
            )
        missing = [
            key
            for key in GARAK_CREDENTIAL_REQUIRED.get(self.garak_target_type, ())
            if not self.garak_credentials.get(key, "").strip()
        ]
        if missing:
            raise ValueError(
                f"Missing required credential field(s) for target type '{self.garak_target_type}': {', '.join(missing)}."
            )
        return self

    @model_validator(mode="after")
    def validate_deepteam_credentials(self) -> "ToolScanRequest":
        if self.tool_id != "deepteam":
            return self
        if self.deepteam_provider not in DEEPTEAM_PROVIDERS:
            raise ValueError(f"Unsupported DeepTeam provider: {self.deepteam_provider}")
        if self.target_id:
            # Credentials come from the saved target, resolved server-side after this
            # validator runs (submit_scan_job) - nothing to check yet.
            return self
        allowed = set(DEEPTEAM_CREDENTIAL_SCHEMA.get(self.deepteam_provider, ()))
        unknown = set(self.deepteam_credentials) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported credential field(s) for provider '{self.deepteam_provider}': {', '.join(sorted(unknown))}."
            )
        missing = [
            key
            for key in DEEPTEAM_CREDENTIAL_REQUIRED.get(self.deepteam_provider, ())
            if not self.deepteam_credentials.get(key, "").strip()
        ]
        if missing:
            raise ValueError(
                f"Missing required credential field(s) for provider '{self.deepteam_provider}': {', '.join(missing)}."
            )
        return self


class ToolScanResult(BaseModel):
    scan_id: str
    tool_id: str
    tool_name: str
    status: str
    started_at: datetime
    completed_at: datetime
    command: list[str]
    executable_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    error: str | None = None
    install_hint: str | None = None
    dry_run: bool = False
    report_paths: list[str] = Field(default_factory=list)
    pyrit_scenario: str | None = None
    pyrit_target_type: str | None = None
    pyrit_attack_mode: str | None = None
    pyrit_strategies: list[str] = Field(default_factory=list)
    llm_analysis: dict[str, Any] | None = None


def _get_tool_python(tool_id: str) -> str:
    settings = get_settings()
    mapping = {
        "garak": getattr(settings, "garak_python", ""),
        "pyrit": getattr(settings, "pyrit_python", ""),
        "deepteam": getattr(settings, "deepteam_python", ""),
    }
    path = mapping.get(tool_id, "").strip()
    if not path:
        path = shutil.which(tool_id) or tool_id
    return path


def _get_pyrit_executable() -> str:
    pyrit_bin = shutil.which("pyrit_scan") or shutil.which("pyrit")
    if pyrit_bin:
        return pyrit_bin

    python = _get_tool_python("pyrit")
    candidates = [
        Path(python).parent / "pyrit_scan",
        Path(python).parent / "pyrit_scan.exe",
        Path(python).parent / "pyrit",
        Path(python).parent / "pyrit.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return "pyrit_scan"


def _get_deepteam_executable() -> str:
    # deepteam has no __main__.py (confirmed - `python -m deepteam` fails), it's a
    # typer app installed as a console script, same shape as pyrit_scan.
    deepteam_bin = shutil.which("deepteam")
    if deepteam_bin:
        return deepteam_bin

    python = _get_tool_python("deepteam")
    candidates = [Path(python).parent / "deepteam", Path(python).parent / "deepteam.exe"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return "deepteam"


def _is_tool_available(tool_id: str) -> bool:
    python = _get_tool_python(tool_id)

    if tool_id == "pyrit":
        pyrit_bin = _get_pyrit_executable()
        try:
            result = subprocess.run(
                [pyrit_bin, "--help"],
                capture_output=True,
                timeout=15,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    if tool_id == "deepteam":
        deepteam_bin = _get_deepteam_executable()
        try:
            result = subprocess.run(
                [deepteam_bin, "--help"],
                capture_output=True,
                timeout=15,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    try:
        result = subprocess.run(
            [python, "-m", tool_id, "--help"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def list_tool_connectors() -> list[dict[str, Any]]:
    connectors: list[dict[str, Any]] = []
    for tool in TOOL_DEFINITIONS.values():
        python_path = _get_tool_python(tool.id)
        available = _is_tool_available(tool.id)
        connectors.append(
            {
                "id": tool.id,
                "name": tool.name,
                "purpose": tool.purpose,
                "profiles": list(tool.profiles),
                "executable": tool.executable,
                "executable_path": python_path,
                "available": available,
                "status": "Connector ready" if available else "Tool missing",
                "install_hint": tool.install_hint,
                "command_template": _command_template(tool.id),
                "target_types": list(GARAK_TARGET_TYPES) if tool.id == "garak" else [],
                "probe_modes": list(GARAK_PROBE_PRESETS.keys()) + ["custom_probe", "custom_tag"] if tool.id == "garak" else [],
                "buffs": list(GARAK_COMMON_BUFFS) if tool.id == "garak" else [],
                "credential_fields": GARAK_CREDENTIAL_FIELDS if tool.id == "garak" else (
                    DEEPTEAM_CREDENTIAL_FIELDS if tool.id == "deepteam" else {}
                ),
                "providers": list(DEEPTEAM_PROVIDERS) if tool.id == "deepteam" else [],
                "vulnerability_categories": DEEPTEAM_VULNERABILITY_CATEGORIES if tool.id == "deepteam" else {},
                "vulnerabilities": DEEPTEAM_VULNERABILITIES if tool.id == "deepteam" else {},
                "single_turn_attacks": DEEPTEAM_SINGLE_TURN_ATTACKS if tool.id == "deepteam" else [],
                "multi_turn_attacks": DEEPTEAM_MULTI_TURN_ATTACKS if tool.id == "deepteam" else [],
            }
        )
    return connectors


def run_tool_scan(request: ToolScanRequest) -> ToolScanResult:
    tool = TOOL_DEFINITIONS[request.tool_id]
    started_at = datetime.now(UTC)
    executable_path = _get_tool_python(tool.id)
    scan_id = str(uuid4())

    TOOL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tool-scan-") as tmpdir:
        target_path: Path | None = None
        if request.target is not None:
            target_path = Path(tmpdir) / "target.json"
            target_path.write_text(json.dumps(request.target.model_dump(mode="json"), indent=2), encoding="utf-8")
        result_meta = _result_metadata(request)

        try:
            command, report_prefix = _build_command(tool, request, target_path, scan_id, tmpdir)
        except ValueError as exc:
            return ToolScanResult(
                scan_id=scan_id,
                tool_id=tool.id,
                tool_name=tool.name,
                status="FAILED",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                command=[tool.executable],
                executable_path=executable_path,
                error=str(exc),
                **result_meta,
            )

        if request.dry_run:
            return ToolScanResult(
                scan_id=scan_id,
                tool_id=tool.id,
                tool_name=tool.name,
                status="VALIDATED",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                command=command,
                executable_path=executable_path,
                dry_run=True,
                install_hint=None if _is_tool_available(tool.id) else tool.install_hint,
                report_paths=_matching_reports(report_prefix),
                **result_meta,
            )

        if not _is_tool_available(tool.id):
            return ToolScanResult(
                scan_id=scan_id,
                tool_id=tool.id,
                tool_name=tool.name,
                status="FAILED",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                command=command,
                executable_path=executable_path,
                error=f"{tool.name} is not available at '{executable_path}'. Ensure the tool environment is installed and reachable.",
                install_hint=tool.install_hint,
                report_paths=_matching_reports(report_prefix),
                **result_meta,
            )

        try:
            # Concurrency safety: each scan gets an isolated child-process environment.
            # Never mutate os.environ globally and never write .env files for per-scan target data.
            # encoding/errors are explicit because tools like garak print UTF-8 (emoji banners,
            # etc.) and text=True alone falls back to the platform's default encoding - cp1252
            # on Windows - which cannot decode that output and crashes the subprocess reader thread.
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=request.timeout_seconds,
                env=_tool_env(request),
            )
        except subprocess.TimeoutExpired as exc:
            return ToolScanResult(
                scan_id=scan_id,
                tool_id=tool.id,
                tool_name=tool.name,
                status="FAILED",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                command=command,
                executable_path=executable_path,
                stdout=_redact_secrets(exc.stdout or "", request),
                stderr=_redact_secrets(exc.stderr or "", request),
                error=f"{tool.name} scan timed out after {request.timeout_seconds} seconds.",
                report_paths=_matching_reports(report_prefix),
                **result_meta,
            )
        except OSError as exc:
            return ToolScanResult(
                scan_id=scan_id,
                tool_id=tool.id,
                tool_name=tool.name,
                status="FAILED",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                command=command,
                executable_path=executable_path,
                error=str(exc),
                report_paths=_matching_reports(report_prefix),
                **result_meta,
            )

    status, error = _completed_status(tool, completed.stdout, completed.stderr, completed.returncode)
    result = ToolScanResult(
        scan_id=scan_id,
        tool_id=tool.id,
        tool_name=tool.name,
        status=status,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        command=command,
        executable_path=executable_path,
        stdout=_redact_secrets(completed.stdout, request),
        stderr=_redact_secrets(completed.stderr, request),
        return_code=completed.returncode,
        error=error,
        report_paths=_matching_reports(report_prefix),
        **result_meta,
    )

    return result


def _build_command(
    tool: ToolDefinition, request: ToolScanRequest, target_path: Path | None, scan_id: str, tmpdir: str
) -> tuple[list[str], Path | None]:
    if tool.id == "garak":
        command, report_prefix = _build_garak_command(tool, request, scan_id)
    elif tool.id == "pyrit":
        command, report_prefix = _build_pyrit_command(tool, request)
    else:
        command, report_prefix = _build_deepteam_command(request, scan_id, tmpdir)

    if request.options.strip():
        command.extend(shlex.split(request.options))
    return command, report_prefix


def _build_garak_command(tool: ToolDefinition, request: ToolScanRequest, scan_id: str) -> tuple[list[str], Path]:
    target_name = request.garak_target_name.strip()
    if not target_name:
        raise ValueError("Garak model name is required.")

    report_prefix = TOOL_REPORT_DIR / f"garak-{scan_id}"
    python = _get_tool_python("garak")
    command = [
        python,
        "-m",
        "garak",
        "--target_type",
        request.garak_target_type,
        "--target_name",
        target_name,
    ]

    if request.garak_probe_mode in GARAK_PROBE_PRESETS:
        flag, value = GARAK_PROBE_PRESETS[request.garak_probe_mode]
    else:
        value = request.garak_probe_value.strip()
        if not value:
            raise ValueError("Select or enter a Garak probe value.")
        flag = "--probes" if request.garak_probe_mode == "custom_probe" else "--probe_tags"

    command.extend([flag, value, "--report_prefix", str(report_prefix)])
    ollama_host = request.garak_credentials.get("OLLAMA_HOST", "").strip()
    if request.garak_target_type == "ollama" and ollama_host:
        command.extend(
            ["--generator_options", json.dumps({"ollama": {"host": ollama_host}})]
        )
    azure_api_version = request.garak_credentials.get("AZURE_API_VERSION", "").strip()
    if request.garak_target_type == "azure" and azure_api_version:
        command.extend(
            ["--generator_options", json.dumps({"azure": {"api_version": azure_api_version}})]
        )
    buffs = [buff.strip() for buff in request.garak_buffs if buff.strip()]
    if buffs:
        command.extend(["--buffs", ",".join(buffs)])
    return command, report_prefix


def _build_pyrit_command(tool: ToolDefinition, request: ToolScanRequest) -> tuple[list[str], Path | None]:
    scenario = request.pyrit_scenario.strip()
    if not scenario:
        raise ValueError("PyRIT scenario is required.")
    if not request.pyrit_endpoint_url.strip():
        raise ValueError("PyRIT endpoint URL is required.")
    if not request.pyrit_deployment_name.strip():
        raise ValueError("PyRIT deployment name is required.")
    if not request.pyrit_model_name.strip():
        raise ValueError("PyRIT model name is required.")
    if not request.pyrit_api_key.strip():
        raise ValueError("PyRIT API key is required.")

    initializers = [value.strip() for value in request.pyrit_initializers if value and value.strip()]
    if not initializers:
        initializers = ["target", "load_default_datasets"]

    seen: set[str] = set()
    strategies = [
        value.strip()
        for value in request.pyrit_strategies
        if value and value.strip() and not (value.strip() in seen or seen.add(value.strip()))
    ]

    command = [
        _get_pyrit_executable(),
        scenario,
        "--target",
        "openai_chat",
        "--initializers",
        *initializers,
    ]

    if strategies:
        command.extend(["--strategies", *strategies])

    if request.pyrit_max_concurrency is not None:
        command.extend(["--max-concurrency", str(request.pyrit_max_concurrency)])
    if request.pyrit_max_retries is not None:
        command.extend(["--max-retries", str(request.pyrit_max_retries)])
    if request.pyrit_max_dataset_size is not None:
        command.extend(["--max-dataset-size", str(request.pyrit_max_dataset_size)])
    if request.pyrit_dataset_names:
        command.extend(["--dataset-names", *request.pyrit_dataset_names])
    if request.pyrit_memory_labels and request.pyrit_memory_labels.strip():
        try:
            memory_labels = json.loads(request.pyrit_memory_labels.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for memory labels: {exc.msg}") from exc
        command.extend(["--memory-labels", json.dumps(memory_labels, separators=(",", ":"))])

    return command, None


def _build_deepteam_command(request: ToolScanRequest, scan_id: str, tmpdir: str) -> tuple[list[str], Path]:
    """DeepTeam is YAML-config-driven (`deepteam run config.yaml`), unlike garak/pyrit's
    CLI-flag approach - see engine/deepteam_catalog.py for why the vulnerability/attack
    catalog is scoped to exactly what deepteam.cli.main.VULN_MAP/ATTACK_MAP recognize.

    The generated config.yaml is written into `tmpdir` (the same TemporaryDirectory
    run_tool_scan already uses for target.json, auto-deleted right after the subprocess
    call) rather than the persistent reports directory - deepteam's load_model() reads
    azure/bedrock api_key fields directly out of the YAML spec dict (no env-var
    alternative exists in its code, unlike garak/pyrit's fully env-var-only credential
    passing), so this file briefly holds a real secret on disk and must not persist.
    The scan's actual *results* (no secrets) go to output_folder under the persistent
    TOOL_REPORT_DIR so worker/result_parser.py::parse_deepteam_findings can read them
    after this function returns.
    """

    if not request.deepteam_purpose.strip():
        raise ValueError("DeepTeam target purpose is required.")
    if not request.deepteam_model.strip():
        raise ValueError("DeepTeam target model is required.")
    if not request.deepteam_vulnerabilities:
        raise ValueError("Select at least one DeepTeam vulnerability.")
    if not request.deepteam_attacks:
        raise ValueError("Select at least one DeepTeam attack.")

    output_folder = TOOL_REPORT_DIR / f"deepteam-{scan_id}"
    config_path = Path(tmpdir) / "deepteam-config.yaml"

    creds = request.deepteam_credentials
    target_model: dict[str, Any] = {
        "provider": request.deepteam_provider,
        "model": request.deepteam_model.strip(),
        "temperature": request.deepteam_temperature,
    }
    if request.deepteam_provider == "ollama":
        target_model["base_url"] = creds.get("base_url", "").strip() or "http://localhost:11434"
    elif request.deepteam_provider == "azure":
        target_model["azure_endpoint"] = creds.get("endpoint", "").strip()
        target_model["deployment_name"] = creds.get("deployment_name", "").strip()
        if creds.get("api_version", "").strip():
            target_model["openai_api_version"] = creds["api_version"].strip()
        target_model["api_key"] = creds.get("api_key", "").strip()
    elif request.deepteam_provider == "bedrock":
        target_model["region_name"] = creds.get("region_name", "").strip()
        target_model["aws_access_key_id"] = creds.get("aws_access_key_id", "").strip()
        target_model["aws_secret_access_key"] = creds.get("aws_secret_access_key", "").strip()
    # openai/anthropic: no extra fields - load_model() reads solely from the worker's
    # ambient OPENAI_API_KEY/ANTHROPIC_API_KEY env vars for these two providers (same
    # ones the fixed simulator/evaluation models already require).

    config = {
        "models": {"simulator": "gpt-4.1", "evaluation": "gpt-4.1"},
        "target": {"purpose": request.deepteam_purpose.strip(), "model": target_model},
        "system_config": {
            "max_concurrent": request.deepteam_max_concurrent,
            "attacks_per_vulnerability_type": request.deepteam_attacks_per_vulnerability_type,
            "run_async": True,
            "ignore_errors": True,
            "output_folder": str(output_folder),
        },
        "default_vulnerabilities": request.deepteam_vulnerabilities,
        "attacks": [{"name": name} for name in request.deepteam_attacks],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    command = [_get_deepteam_executable(), "run", str(config_path)]
    return command, output_folder


def _tool_env(request: ToolScanRequest) -> dict[str, str]:
    env = os.environ.copy()
    # Tools like garak and pyrit_scan print emoji/box-drawing characters (progress bars,
    # result banners) straight to stdout. When stdout is redirected to a pipe (as it is
    # here), Python picks the child process's own default encoding - cp1252 on Windows -
    # for that stream, which crashes the child with UnicodeEncodeError. Forcing UTF-8 here
    # fixes the write side; the parent's explicit encoding="utf-8" on subprocess.run (below)
    # only covers the read side and doesn't help the child's own output encoding.
    env["PYTHONIOENCODING"] = "utf-8"
    if request.tool_id == "garak":
        # Per-scan credentials, allowlisted per target type in GARAK_CREDENTIAL_SCHEMA.
        # Never sourced from the worker's static .env; held only in this in-memory dict
        # for the lifetime of the subprocess call, then garbage collected. The validator
        # on ToolScanRequest already rejects unknown keys, but re-check here too so this
        # stays safe even if _tool_env is ever called on an unvalidated request.
        allowed = set(GARAK_CREDENTIAL_SCHEMA.get(request.garak_target_type, ()))
        for key, value in request.garak_credentials.items():
            if key in allowed and value.strip():
                env[key] = value.strip()
    if request.tool_id == "pyrit":
        # Intentionally override only target-under-test OPENAI_CHAT_* keys.
        # ADVERSARIAL_CHAT_* and OBJECTIVE_SCORER_CHAT_* must stay static from worker .env.
        if request.pyrit_endpoint_url.strip():
            env["OPENAI_CHAT_ENDPOINT"] = request.pyrit_endpoint_url.strip()
        if request.pyrit_api_key.strip():
            env["OPENAI_CHAT_KEY"] = request.pyrit_api_key.strip()
        if request.pyrit_model_name.strip():
            env["OPENAI_CHAT_MODEL"] = request.pyrit_model_name.strip()
            env["OPENAI_CHAT_UNDERLYING_MODEL"] = request.pyrit_model_name.strip()
        if request.pyrit_deployment_name.strip():
            env["OPENAI_CHAT_DEPLOYMENT"] = request.pyrit_deployment_name.strip()
    return env


def _secret_values(request: ToolScanRequest) -> list[str]:
    """Collect this request's secret values so they can be scrubbed from captured output."""

    values = [
        value.strip()
        for key, value in request.garak_credentials.items()
        if key in GARAK_SECRET_CREDENTIAL_KEYS and value.strip()
    ]
    if request.pyrit_api_key.strip():
        values.append(request.pyrit_api_key.strip())
    values.extend(
        value.strip()
        for key, value in request.deepteam_credentials.items()
        if key in DEEPTEAM_SECRET_CREDENTIAL_KEYS and value.strip()
    )
    return values


def _redact_secrets(text: str, request: ToolScanRequest) -> str:
    """Scrub known secret values out of subprocess stdout/stderr before it reaches the UI/logs.

    Tools occasionally echo their own config (including credentials) in verbose/debug
    output; this is a defense-in-depth backstop on top of never passing secrets via CLI args.
    """

    if not text:
        return text or ""
    for value in _secret_values(request):
        text = text.replace(value, "***REDACTED***")
    return text


def _completed_status(tool: ToolDefinition, stdout: str, stderr: str, return_code: int) -> tuple[str, str | None]:
    output = f"{stdout}\n{stderr}".lower()
    if tool.id == "garak" and "environment variable is required" in output:
        return "FAILED", "Garak reported a missing required environment variable."
    return ("COMPLETED" if return_code == 0 else "FAILED"), None


def _matching_reports(report_prefix: Path | None) -> list[str]:
    if report_prefix is None:
        return []
    # Garak's report_prefix matches flat sibling files directly (garak-<id>.report.jsonl
    # etc). DeepTeam's report_prefix is itself a directory (output_folder) that
    # RiskAssessment.save() writes a timestamped JSON file into one level down - the
    # second glob catches that case without affecting garak's flat-file matches.
    direct = [path for path in report_prefix.parent.glob(f"{report_prefix.name}*") if path.is_file()]
    nested = [path for path in report_prefix.parent.glob(f"{report_prefix.name}*/*") if path.is_file()]
    return sorted(str(path) for path in direct + nested)


def _command_template(tool_id: str) -> str:
    if tool_id == "garak":
        return "garak --target_type ollama --target_name llama3.2 --probe_tags owasp --buffs lowercase,paraphrase.Fast --report_prefix reports/tool-scans/garak-<scan_id>"
    if tool_id == "pyrit":
        return "pyrit_scan <scenario_name> --target <target_name> --initializers <initializer1> <initializer2> --strategies <strategy1> <strategy2> --memory-labels '{\"experiment\":\"test1\"}'"
    return "deepteam run <generated-config.yaml> (target/vulnerabilities/attacks assembled from your selections)"


def _result_metadata(request: ToolScanRequest) -> dict[str, Any]:
    if request.tool_id != "pyrit":
        return {}
    seen: set[str] = set()
    strategies = [
        value.strip()
        for value in request.pyrit_strategies
        if value and value.strip() and not (value.strip() in seen or seen.add(value.strip()))
    ]
    return {
        "pyrit_scenario": request.pyrit_scenario.strip() or None,
        "pyrit_target_type": "openai_chat",
        "pyrit_attack_mode": request.pyrit_attack_mode,
        "pyrit_strategies": strategies,
    }
