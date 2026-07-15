"""Orchestration for the async Tool Scan job pipeline (queue+worker replacement for the
synchronous engine/tool_scan.py::run_tool_scan call core/api.py used to make inline).
Mirrors how core/api.py already delegates business logic to engine/scan_orchestrator.py
for /scans rather than inlining it directly in the route handler."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from core.crypto import decrypt, encrypt
from database.models import ScanFindingRecord, ScanJobRecord, ScanTargetRecord
from database.repository import Repository
from engine.deepteam_catalog import (
    DEEPTEAM_CREDENTIAL_REQUIRED,
    DEEPTEAM_CREDENTIAL_SCHEMA,
    DEEPTEAM_PROVIDERS,
    DEEPTEAM_SECRET_CREDENTIAL_KEYS,
)
from engine.tool_scan import (
    GARAK_CREDENTIAL_REQUIRED,
    GARAK_CREDENTIAL_SCHEMA,
    GARAK_SECRET_CREDENTIAL_KEYS,
    GARAK_TARGET_TYPES,
    ToolScanRequest,
)
from worker.celery_app import celery_app

# PyRIT currently only supports one target shape (see ToolScanRequest.pyrit_* fields /
# engine/tool_scan.py::_build_pyrit_command), so unlike Garak's per-target-type schema
# this is a single fixed field set. Only relevant to the saved-target ("Connection")
# feature below - PyRIT's own execution path doesn't use a generic credentials dict.
PYRIT_CREDENTIAL_FIELDS: tuple[str, ...] = ("endpoint_url", "deployment_name", "model_name", "api_key")
PYRIT_SECRET_CREDENTIAL_KEYS: frozenset[str] = frozenset({"api_key"})


class ScanJobSubmitResponse(BaseModel):
    job_id: str
    status: str


class ScanJobStatusResponse(BaseModel):
    job_id: str
    tool_id: str
    status: str
    error: str | None = None
    report_paths: list[str] = []
    llm_analysis: dict[str, Any] | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @classmethod
    def from_record(cls, job: ScanJobRecord) -> "ScanJobStatusResponse":
        return cls(
            job_id=job.id,
            tool_id=job.tool_id,
            status=job.status,
            error=job.error,
            report_paths=job.report_paths or [],
            llm_analysis=job.llm_analysis,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )


class ScanFindingResponse(BaseModel):
    id: int
    probe: str
    detector: str
    passed: bool
    score: float | None = None
    severity: str
    prompt: str
    response: str
    conversation_id: str
    turn: int
    created_at: datetime

    @classmethod
    def from_record(cls, finding: ScanFindingRecord) -> "ScanFindingResponse":
        return cls(
            id=finding.id,
            probe=finding.probe,
            detector=finding.detector,
            passed=finding.passed,
            score=float(finding.score) if finding.score is not None else None,
            severity=finding.severity,
            prompt=finding.prompt,
            response=finding.response,
            conversation_id=finding.conversation_id,
            turn=finding.turn,
            created_at=finding.created_at,
        )


class ScanTargetCreateRequest(BaseModel):
    tool_id: str
    target_type: str
    name: str
    # Garak's --target_name (model/deployment identifier) - not part of `credentials`,
    # see ScanTargetRecord.target_name. Unused for pyrit (deployment_name already lives
    # inside credentials there).
    target_name: str = ""
    credentials: dict[str, str] = {}


class ScanTargetResponse(BaseModel):
    """Never carries *secret* credential values (e.g. API keys) - those stay
    server-side and are only ever decrypted inside the Celery worker. Non-secret
    fields (endpoint URL, model name, API version, deployment name) are returned in
    `visible_credentials` so the UI can display/restore them when a saved connection
    is selected - they were never encrypted in the first place (see
    _secret_keys_for/save_scan_target: only the secret_keys subset gets encrypt()'d
    before storage, everything else is stored as plaintext already)."""

    id: str
    tool_id: str
    target_type: str
    name: str
    target_name: str
    configured_fields: list[str]
    visible_credentials: dict[str, str]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, target: ScanTargetRecord) -> "ScanTargetResponse":
        secret_keys = _secret_keys_for(target.tool_id)
        return cls(
            id=target.id,
            tool_id=target.tool_id,
            target_type=target.target_type,
            name=target.name,
            target_name=target.target_name,
            configured_fields=sorted(target.encrypted_credentials.keys()),
            visible_credentials={
                key: value for key, value in target.encrypted_credentials.items() if key not in secret_keys
            },
            created_at=target.created_at,
            updated_at=target.updated_at,
        )


def _encrypt_request_payload(request: ToolScanRequest) -> dict[str, Any]:
    """Encrypt the same fields engine/tool_scan.py already treats as secret
    (GARAK_SECRET_CREDENTIAL_KEYS, plus pyrit_api_key) before the request is persisted.
    Decryption happens once, inside the Celery task, immediately before injecting them
    as subprocess-scoped env vars (worker/tasks.py::_decrypt_payload)."""

    payload = request.model_dump(mode="json")
    garak_credentials = payload.get("garak_credentials") or {}
    payload["garak_credentials"] = {
        key: (encrypt(value) if key in GARAK_SECRET_CREDENTIAL_KEYS and value else value)
        for key, value in garak_credentials.items()
    }
    if payload.get("pyrit_api_key"):
        payload["pyrit_api_key"] = encrypt(payload["pyrit_api_key"])
    deepteam_credentials = payload.get("deepteam_credentials") or {}
    payload["deepteam_credentials"] = {
        key: (encrypt(value) if key in DEEPTEAM_SECRET_CREDENTIAL_KEYS and value else value)
        for key, value in deepteam_credentials.items()
    }
    return payload


_PUBLISH_TIMEOUT_SECONDS = 5.0


def _secret_keys_for(tool_id: str) -> frozenset[str]:
    if tool_id == "garak":
        return GARAK_SECRET_CREDENTIAL_KEYS
    if tool_id == "deepteam":
        return DEEPTEAM_SECRET_CREDENTIAL_KEYS
    return PYRIT_SECRET_CREDENTIAL_KEYS


async def _apply_saved_target(request: ToolScanRequest, repo: Repository) -> ToolScanRequest:
    """Resolve request.target_id into a saved ScanTargetRecord, decrypt its credentials,
    and fill in the tool-specific fields run_tool_scan actually reads - so the rest of
    the pipeline (encryption for storage, the worker's execution) never needs to know
    whether credentials came from a saved Connection or were entered ad-hoc.
    Raises sqlalchemy.exc.NoResultFound if target_id doesn't exist (left to the caller/
    route to turn into a 404 - deliberately not caught here)."""

    target = await repo.get_scan_target(request.target_id)
    secret_keys = _secret_keys_for(target.tool_id)
    decrypted = {
        key: (decrypt(value) if key in secret_keys and value else value)
        for key, value in target.encrypted_credentials.items()
    }
    if target.tool_id == "garak":
        # garak_target_name is intentionally NOT overridden here - the frontend
        # pre-fills it from target.target_name when the connection is selected, but
        # leaves it editable per-scan (unlike credentials, which are fully hidden and
        # always resolved server-side). Whatever the request already carries wins.
        return request.model_copy(update={"garak_target_type": target.target_type, "garak_credentials": decrypted})
    if target.tool_id == "pyrit":
        return request.model_copy(
            update={
                "pyrit_endpoint_url": decrypted.get("endpoint_url", ""),
                "pyrit_deployment_name": decrypted.get("deployment_name", ""),
                "pyrit_model_name": decrypted.get("model_name", ""),
                "pyrit_api_key": decrypted.get("api_key", ""),
            }
        )
    if target.tool_id == "deepteam":
        # deepteam_model (equivalent to garak_target_name) is intentionally NOT
        # overridden here, same reasoning as garak above - it's a per-scan editable
        # field pre-filled by the frontend from target.target_name.
        return request.model_copy(update={"deepteam_provider": target.target_type, "deepteam_credentials": decrypted})
    return request


async def submit_scan_job(request: ToolScanRequest, repo: Repository) -> ScanJobSubmitResponse:
    if request.target_id:
        request = await _apply_saved_target(request, repo)
    job_id = str(uuid4())
    payload = _encrypt_request_payload(request)
    await repo.create_scan_job(job_id, request.tool_id, payload)
    try:
        # Only the job_id ever goes on the queue - the worker re-fetches (and decrypts)
        # the full request from Postgres when it picks the task up. Runs off the event
        # loop since Celery's publish call is blocking I/O. Kombu's own connection-retry
        # settings (broker_connection_timeout / task_publish_retry_policy in
        # worker/celery_app.py) turned out NOT to bound how long .send_task() blocks
        # when the broker is unreachable (verified empirically: it hung 20s+ even with
        # those set - connection *establishment* retries separately from publish
        # retries and ignores task_publish_retry_policy). asyncio.wait_for is a hard
        # backstop instead: the request fails fast even if the background thread
        # (which can't be killed) keeps retrying until Celery's own retries exhaust.
        await asyncio.wait_for(
            asyncio.to_thread(celery_app.send_task, "run_scan_job", args=[job_id]),
            timeout=_PUBLISH_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        # Without this, a broker outage would leave the job permanently stuck at
        # "queued" - no worker will ever pick up a task that was never published.
        await repo.update_scan_job_status(job_id, "failed", error=f"Could not queue scan: {exc}")
        raise
    return ScanJobSubmitResponse(job_id=job_id, status="queued")


async def get_scan_job(job_id: str, repo: Repository) -> ScanJobStatusResponse:
    job = await repo.get_scan_job(job_id)
    return ScanJobStatusResponse.from_record(job)


async def list_scan_jobs(status: str | None, repo: Repository) -> list[ScanJobStatusResponse]:
    jobs = await repo.list_scan_jobs(status)
    return [ScanJobStatusResponse.from_record(job) for job in jobs]


async def list_scan_findings(job_id: str, offset: int, limit: int, repo: Repository) -> list[ScanFindingResponse]:
    findings = await repo.list_scan_findings(job_id, offset=offset, limit=limit)
    return [ScanFindingResponse.from_record(finding) for finding in findings]


def _validate_garak_target_credentials(target_type: str, credentials: dict[str, str]) -> None:
    if target_type not in GARAK_TARGET_TYPES:
        raise ValueError(f"Unsupported Garak target type: {target_type}")
    allowed = set(GARAK_CREDENTIAL_SCHEMA.get(target_type, ()))
    unknown = set(credentials) - allowed
    if unknown:
        raise ValueError(f"Unsupported credential field(s) for target type '{target_type}': {', '.join(sorted(unknown))}.")
    missing = [key for key in GARAK_CREDENTIAL_REQUIRED.get(target_type, ()) if not credentials.get(key, "").strip()]
    if missing:
        raise ValueError(f"Missing required credential field(s) for target type '{target_type}': {', '.join(missing)}.")


def _validate_pyrit_target_credentials(credentials: dict[str, str]) -> None:
    unknown = set(credentials) - set(PYRIT_CREDENTIAL_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported PyRIT credential field(s): {', '.join(sorted(unknown))}.")
    missing = [key for key in PYRIT_CREDENTIAL_FIELDS if not credentials.get(key, "").strip()]
    if missing:
        raise ValueError(f"Missing required PyRIT credential field(s): {', '.join(missing)}.")


def _validate_deepteam_target_credentials(provider: str, credentials: dict[str, str]) -> None:
    if provider not in DEEPTEAM_PROVIDERS:
        raise ValueError(f"Unsupported DeepTeam provider: {provider}")
    allowed = set(DEEPTEAM_CREDENTIAL_SCHEMA.get(provider, ()))
    unknown = set(credentials) - allowed
    if unknown:
        raise ValueError(f"Unsupported credential field(s) for provider '{provider}': {', '.join(sorted(unknown))}.")
    missing = [key for key in DEEPTEAM_CREDENTIAL_REQUIRED.get(provider, ()) if not credentials.get(key, "").strip()]
    if missing:
        raise ValueError(f"Missing required credential field(s) for provider '{provider}': {', '.join(missing)}.")


async def save_scan_target(request: ScanTargetCreateRequest, repo: Repository) -> ScanTargetResponse:
    if request.tool_id == "garak":
        _validate_garak_target_credentials(request.target_type, request.credentials)
    elif request.tool_id == "pyrit":
        _validate_pyrit_target_credentials(request.credentials)
    elif request.tool_id == "deepteam":
        _validate_deepteam_target_credentials(request.target_type, request.credentials)
    else:
        raise ValueError(f"Saved connections aren't supported yet for tool '{request.tool_id}'.")
    secret_keys = _secret_keys_for(request.tool_id)
    encrypted = {
        key: (encrypt(value) if key in secret_keys and value else value)
        for key, value in request.credentials.items()
    }
    record = await repo.create_scan_target(
        request.tool_id, request.target_type, request.name, encrypted, target_name=request.target_name
    )
    return ScanTargetResponse.from_record(record)


async def list_scan_targets(tool_id: str | None, repo: Repository) -> list[ScanTargetResponse]:
    targets = await repo.list_scan_targets(tool_id)
    return [ScanTargetResponse.from_record(target) for target in targets]
