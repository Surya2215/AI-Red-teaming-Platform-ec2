"""FastAPI application for programmatic scans and the React frontend."""

from __future__ import annotations

import hmac
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field, SecretStr, model_validator
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import (
    ALL_ROLES,
    ROLE_ADMIN,
    ROLE_MEMBER,
    ROLE_VIEWER,
    AuthenticatedUser,
    bootstrap_admin,
    check_login_rate_limit,
    create_access_token,
    decode_access_token,
    get_current_user,
    hash_password,
    record_audit,
    require_role,
    require_user,
    verify_password,
)
from core.config import ROOT_DIR, enforce_production_auth_guard, get_settings
from core.llm_client import AzureOpenAIClient
from core.schemas import ScanRequest, ScanResult, TargetConfig
from database.repository import Repository
from database.session import AsyncSessionLocal, get_session, init_db
from engine.pdf_report import export_pdf, export_tool_scan_pdf, report_file_name, tool_scan_report_file_name
from engine.report_generator import generate_enterprise_report
from engine.scan_orchestrator import ScanOrchestrator
from engine.scenario_loader import PluginLoader
from engine.target_executor import TargetExecutor
from engine.tool_scan import TOOL_DEFINITIONS, ToolScanRequest, ToolScanResult, list_tool_connectors, run_tool_scan
from engine.tool_scan_jobs import (
    ScanFindingResponse,
    ScanJobStatusResponse,
    ScanJobSubmitResponse,
    ScanTargetCreateRequest,
    ScanTargetResponse,
    get_scan_job,
    list_scan_findings,
    list_scan_jobs,
    list_scan_targets,
    save_scan_target,
    submit_scan_job,
)


app = FastAPI(title="AI Red Teaming Platform", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_AUTH_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/auth/login"}


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    # Opt-in: only enforced once an operator sets API_KEY (see core/config.py). Local
    # dev with no API_KEY configured stays exactly as open as it is today.
    # CORS preflight (OPTIONS) is always exempt: browsers send it without any custom
    # headers (X-API-Key/Authorization included) by spec, so gating it here would
    # fail the preflight itself and the browser would never even attempt the real
    # request - surfacing as an opaque "Failed to fetch" with no HTTP status to act on.
    expected = get_settings().api_key
    if expected and request.method != "OPTIONS" and request.url.path not in _AUTH_EXEMPT_PATHS:
        provided = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(provided, expected):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key."})
    return await call_next(request)


@app.middleware("http")
async def require_user_auth(request: Request, call_next):
    # Opt-in via AUTH_ENABLED - unset means today's no-login behavior is unchanged
    # (core/config.py's auth_enabled docstring). Once on, every request needs EITHER
    # a valid X-API-Key (unattributed, for CI/automation - checked by the middleware
    # above, already ran by this point) OR a valid Authorization: Bearer <jwt>
    # (attributed to a real user, for the web UI and for RBAC/audit-log purposes).
    # See require_api_key's comment above for why OPTIONS is always exempt.
    settings = get_settings()
    if not settings.auth_enabled or request.method == "OPTIONS" or request.url.path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)

    api_key_ok = bool(settings.api_key) and hmac.compare_digest(request.headers.get("x-api-key", ""), settings.api_key)
    if api_key_ok:
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
    if not token or not decode_access_token(token):
        return JSONResponse(status_code=401, content={"detail": "Sign in required."})
    return await call_next(request)


TARGET_DIR = ROOT_DIR / "targets"
REPORT_DIR = ROOT_DIR / "reports"

OWASP_CATEGORY_OPTIONS = [
    "LLM01-Prompt Injection",
    "LLM02-Sensitive Information Disclosure",
    "LLM03-Supply Chain",
    "LLM04-Data_model_poisoning",
    "LLM05-Improper_output_handling",
    "LLM06-Excessive_agency",
    "LLM07-Insecure Plugin Design",
    "LLM08-Vector_Embedding_Weaknesses",
    "LLM09-Misinformation",
    "LLM10-Unbounded_Consumption",
    # OWASP Top 10 for Agentic Applications (ASI01-ASI10)
    "ASI01-Agent_Goal_Hijack",
    "ASI02-Tool_Misuse_Exploitation",
    "ASI03-Identity_Privilege_Abuse",
    "ASI04-Agentic_Supply_Chain_Vulnerabilities",
    "ASI05-Unexpected_Code_Execution",
    "ASI06-Memory_Context_Poisoning",
    "ASI07-Insecure_Inter_Agent_Communication",
    "ASI08-Cascading_Failures",
    "ASI09-Human_Agent_Trust_Exploitation",
    "ASI10-Rogue_Agents",
]


class TargetAssistantMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str


class TargetAssistantRequest(BaseModel):
    messages: list[TargetAssistantMessage] = Field(default_factory=list)
    current_target: dict[str, Any] = Field(default_factory=dict)
    delivery_template: dict[str, Any] = Field(default_factory=dict)
    auth_template: dict[str, Any] = Field(default_factory=dict)
    combination: str = ""


class TargetAssistantResponse(BaseModel):
    reply: str
    target: dict[str, Any] | None = None
    provider: str


class RuntimeLLMSettingsRequest(BaseModel):
    llm_provider: str
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None
    anthropic_base_url: str | None = None
    huggingface_api_key: str | None = None
    huggingface_model: str | None = None
    huggingface_base_url: str | None = None
    ollama_model: str | None = None
    ollama_base_url: str | None = None
    aws_region: str | None = None
    aws_bedrock_model_id: str | None = None


@app.on_event("startup")
async def startup() -> None:
    enforce_production_auth_guard(get_settings())
    await init_db()
    try:
        async with AsyncSessionLocal() as session:
            await bootstrap_admin(session)
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Failed to bootstrap the initial admin user on startup.")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
#  Production auth: login, current user, user management, audit log
#  (core/auth.py - opt-in via AUTH_ENABLED, see that module's docstring)
# ─────────────────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None = None

    @classmethod
    def from_record(cls, record) -> "UserOut":
        return cls(
            id=record.id,
            email=record.email,
            display_name=record.display_name,
            role=record.role,
            is_active=record.is_active,
            created_at=record.created_at,
            last_login_at=record.last_login_at,
        )


class LoginResponse(BaseModel):
    access_token: str
    user: UserOut


class CreateUserRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    display_name: str = ""
    role: str = ROLE_ADMIN

    @model_validator(mode="after")
    def _validate_role(self) -> "CreateUserRequest":
        if self.role not in ALL_ROLES:
            raise ValueError(f"role must be one of {ALL_ROLES}")
        return self


class UpdateUserRequest(BaseModel):
    display_name: str | None = None
    role: str | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=8)

    @model_validator(mode="after")
    def _validate_role(self) -> "UpdateUserRequest":
        if self.role is not None and self.role not in ALL_ROLES:
            raise ValueError(f"role must be one of {ALL_ROLES}")
        return self


@app.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest, request: Request, session: AsyncSession = Depends(get_session)) -> LoginResponse:
    if not get_settings().auth_enabled:
        raise HTTPException(status_code=404, detail="User auth is not enabled on this deployment.")
    client_ip = request.client.host if request.client else ""
    check_login_rate_limit(f"ip:{client_ip}", f"email:{payload.email.strip().lower()}")
    repo = Repository(session)
    record = await repo.get_user_by_email(payload.email)
    if record is None or not record.is_active or not verify_password(payload.password, record.password_hash):
        await record_audit(
            session, user=None, action="login.failed", resource_type="user", resource_id=payload.email.strip().lower(),
            request=request, unauthenticated_actor=payload.email.strip().lower(),
        )
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    await repo.record_login(record.id)
    await record_audit(session, user=AuthenticatedUser(record.id, record.email, record.role), action="login.success", resource_type="user", resource_id=record.id, request=request)
    token = create_access_token(record)
    return LoginResponse(access_token=token, user=UserOut.from_record(record))


@app.get("/auth/me", response_model=UserOut)
async def get_me(user: AuthenticatedUser = Depends(require_user), session: AsyncSession = Depends(get_session)) -> UserOut:
    record = await Repository(session).get_user(user.id)
    if record is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserOut.from_record(record)


@app.get("/auth/users", response_model=list[UserOut])
async def list_users(_: AuthenticatedUser = Depends(require_role(ROLE_ADMIN)), session: AsyncSession = Depends(get_session)) -> list[UserOut]:
    records = await Repository(session).list_users()
    return [UserOut.from_record(record) for record in records]


@app.post("/auth/users", response_model=UserOut)
async def create_user(
    payload: CreateUserRequest,
    request: Request,
    admin: AuthenticatedUser = Depends(require_role(ROLE_ADMIN)),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    repo = Repository(session)
    if await repo.get_user_by_email(payload.email) is not None:
        raise HTTPException(status_code=409, detail="A user with this email already exists.")
    record = await repo.create_user(
        email=payload.email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
        role=payload.role,
    )
    await record_audit(
        session, user=admin, action="user.create", resource_type="user", resource_id=record.id,
        detail={"email": record.email, "role": record.role}, request=request,
    )
    return UserOut.from_record(record)


@app.patch("/auth/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: str,
    payload: UpdateUserRequest,
    request: Request,
    admin: AuthenticatedUser = Depends(require_role(ROLE_ADMIN)),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    repo = Repository(session)
    if payload.role is not None and user_id == admin.id and payload.role != ROLE_ADMIN:
        raise HTTPException(status_code=400, detail="You cannot remove your own admin role.")
    if payload.is_active is False and user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account.")
    record = await repo.update_user(
        user_id,
        display_name=payload.display_name,
        role=payload.role,
        is_active=payload.is_active,
        password_hash=hash_password(payload.password) if payload.password else None,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="User not found.")
    await record_audit(
        session, user=admin, action="user.update", resource_type="user", resource_id=user_id,
        detail=payload.model_dump(exclude={"password"}, exclude_none=True), request=request,
    )
    return UserOut.from_record(record)


@app.delete("/auth/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    admin: AuthenticatedUser = Depends(require_role(ROLE_ADMIN)),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    if not await Repository(session).delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    await record_audit(session, user=admin, action="user.delete", resource_type="user", resource_id=user_id, request=request)
    return {"deleted": True, "id": user_id}


@app.get("/audit-log")
async def get_audit_log(
    offset: int = 0,
    limit: int = 100,
    action_prefix: str | None = None,
    user_email: str | None = None,
    _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN)),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    limit = max(1, min(500, limit))
    entries = await Repository(session).list_audit_entries(
        offset=max(0, offset), limit=limit, action_prefix=action_prefix, user_email=user_email
    )
    return [
        {
            "id": entry.id,
            "user_id": entry.user_id,
            "user_email": entry.user_email,
            "action": entry.action,
            "resource_type": entry.resource_type,
            "resource_id": entry.resource_id,
            "detail": entry.detail,
            "ip_address": entry.ip_address,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
        }
        for entry in entries
    ]


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip(".-")
    return stem or "target"


def _safe_json_path(directory: Path, filename: str) -> Path:
    path = (directory / filename).resolve()
    root = directory.resolve()
    if path.parent != root or path.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Invalid JSON filename.")
    return path


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"{path.name} is not a JSON object.")
    return payload


@app.post("/targets")
async def save_target(
    target: TargetConfig,
    session: AsyncSession = Depends(get_session),
    _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER)),
) -> dict[str, str]:
    await Repository(session).upsert_target(target)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_stem(target.name)}.json"
    (TARGET_DIR / filename).write_text(json.dumps(target.model_dump(mode="json"), indent=2), encoding="utf-8")
    return {"status": "saved", "target": target.name, "filename": filename}


@app.put("/targets/{filename}")
async def update_target(
    filename: str,
    target: TargetConfig,
    session: AsyncSession = Depends(get_session),
    _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER)),
) -> dict[str, str]:
    old_path = _safe_json_path(TARGET_DIR, filename)
    await Repository(session).upsert_target(target)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    new_filename = f"{_safe_stem(target.name)}.json"
    new_path = TARGET_DIR / new_filename
    new_path.write_text(json.dumps(target.model_dump(mode="json"), indent=2), encoding="utf-8")
    if old_path.exists() and old_path.resolve() != new_path.resolve():
        old_path.unlink()
    return {"status": "saved", "target": target.name, "filename": new_filename}


@app.get("/targets")
async def list_targets() -> list[dict[str, Any]]:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    targets: list[dict[str, Any]] = []
    for path in sorted(TARGET_DIR.glob("*.json")):
        payload = _read_json_file(path)
        if str(payload.get("name") or "").startswith("[REFERENCE]"):
            continue
        targets.append({"filename": path.name, "target": payload})
    return targets


@app.get("/targets/{filename}")
async def get_target(filename: str) -> dict[str, Any]:
    path = _safe_json_path(TARGET_DIR, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Target not found.")
    return {"filename": path.name, "target": _read_json_file(path)}


@app.delete("/targets/{filename}")
async def delete_target(
    filename: str, _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER))
) -> dict[str, object]:
    path = _safe_json_path(TARGET_DIR, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Target not found.")
    path.unlink()
    return {"deleted": True, "filename": path.name}


@app.post("/targets/test")
async def test_target(target: TargetConfig) -> dict[str, object]:
    executor = TargetExecutor()
    return await executor.test_connection(target)


@app.post("/targets/assistant", response_model=TargetAssistantResponse)
async def target_assistant(request: TargetAssistantRequest) -> TargetAssistantResponse:
    """Generate/refine target JSON using Azure OpenAI configuration when available."""

    settings = get_settings()
    client = AzureOpenAIClient(settings=settings)
    system_prompt = """You are a target-configuration assistant for an AI red-teaming platform.
Return only valid JSON. Do not include markdown.
Your response schema:
{
  "reply": "short user-facing explanation",
  "target": {
    "name": "string",
    "url": "string",
    "method": "POST|GET|PUT|PATCH",
    "headers": {},
    "request_template": {},
    "auth": {},
    "timeout_seconds": 30
  }
}
The target must remain compatible with the scanner:
- request_template must include the attack placeholder "{{prompt}}" wherever the user message belongs.
- Preserve delivery/auth template metadata under target.auth.template_preview when present.
- For no auth, use target.auth.type = "none".
- For bearer env auth, use target.auth.type = "bearer" and token_env.
- For OAuth2 client credentials, use target.auth.type = "session" and a workflow with credential_authentication and next_turn.
- Keep JSON minimal and executable."""
    user_prompt = json.dumps(
        {
            "task": "Update or generate the best target JSON from the conversation and selected templates.",
            "messages": [message.model_dump() for message in request.messages],
            "current_target": request.current_target,
            "selected_delivery_template": request.delivery_template,
            "selected_auth_template": request.auth_template,
            "runtime_combination": request.combination,
        },
        ensure_ascii=False,
    )
    raw = await client.complete_json(system_prompt, user_prompt)
    target_payload = raw.get("target") if isinstance(raw.get("target"), dict) else None
    reply = str(raw.get("reply") or "I updated the target JSON draft. Review it in Manual Form before saving.")
    if target_payload is not None:
        try:
            target_payload = TargetConfig.model_validate(target_payload).model_dump(mode="json")
        except Exception:
            target_payload = None
            reply = "I could not produce a valid target JSON. Please add the application name, endpoint URL, method, auth type, and response format."
    return TargetAssistantResponse(
        reply=reply,
        target=target_payload,
        provider=str(raw.get("_provider") or settings.llm_provider if settings.llm_ready else "local_fallback"),
    )


@app.post("/scans", response_model=ScanResult)
async def run_scan(
    request: ScanRequest,
    session: AsyncSession = Depends(get_session),
    _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER)),
) -> ScanResult:
    orchestrator = ScanOrchestrator(repository=Repository(session))
    return await orchestrator.run_scan(request)


@app.post("/scans/{scan_id}/cancel")
async def cancel_scan(
    scan_id: str, _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER))
) -> dict[str, str]:
    TargetExecutor.request_cancel(scan_id)
    return {"status": "cancel_requested", "scan_id": scan_id}


@app.get("/tool-scans/tools")
async def tool_scan_tools() -> dict[str, Any]:
    return {"tools": list_tool_connectors()}


@app.post("/tool-scans/jobs", response_model=ScanJobSubmitResponse)
async def create_tool_scan_job(
    request: ToolScanRequest,
    session: AsyncSession = Depends(get_session),
    _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER)),
) -> ScanJobSubmitResponse:
    """Queue a Garak/PyRIT/DeepTeam scan for async execution on the Celery worker.
    Returns immediately with a job_id - never blocks on tool execution (replaces the
    old synchronous /tool-scans/run, which called run_tool_scan() inline)."""

    try:
        return await submit_scan_job(request, Repository(session))
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail="Saved connection not found.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Could not queue scan: the task queue is unreachable.") from exc


@app.get("/tool-scans/jobs/{job_id}", response_model=ScanJobStatusResponse)
async def get_tool_scan_job(job_id: str, session: AsyncSession = Depends(get_session)) -> ScanJobStatusResponse:
    try:
        return await get_scan_job(job_id, Repository(session))
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail="Scan job not found.") from exc


@app.get("/tool-scans/jobs/{job_id}/findings", response_model=list[ScanFindingResponse])
async def get_tool_scan_job_findings(
    job_id: str, offset: int = 0, limit: int = 100, session: AsyncSession = Depends(get_session)
) -> list[ScanFindingResponse]:
    return await list_scan_findings(job_id, offset, limit, Repository(session))


@app.get("/tool-scans/jobs", response_model=list[ScanJobStatusResponse])
async def get_tool_scan_jobs(
    status: str | None = None, session: AsyncSession = Depends(get_session)
) -> list[ScanJobStatusResponse]:
    return await list_scan_jobs(status, Repository(session))


@app.delete("/tool-scans/jobs/{job_id}")
async def delete_tool_scan_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
    _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER)),
) -> dict[str, object]:
    deleted = await Repository(session).delete_scan_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scan job not found.")
    return {"deleted": True, "job_id": job_id}


@app.post("/tool-scans/validate", response_model=ToolScanResult)
async def validate_tool_scan(request: ToolScanRequest) -> ToolScanResult:
    """Synchronous, subprocess-free command validation only - runs entirely inline
    since dry_run scans never shell out (see engine/tool_scan.py::run_tool_scan's
    dry_run branch). Real (non-dry-run) scans must go through POST /tool-scans/jobs
    instead so the request never blocks on tool execution."""

    if not request.dry_run:
        raise HTTPException(status_code=400, detail="Use POST /tool-scans/jobs for non-dry-run scans.")
    return run_tool_scan(request)


@app.get("/tool-scans/jobs/{job_id}/pdf")
async def get_tool_scan_job_pdf(job_id: str, session: AsyncSession = Depends(get_session)) -> Response:
    try:
        job = await get_scan_job(job_id, Repository(session))
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail="Scan job not found.") from exc
    tool_name = TOOL_DEFINITIONS.get(job.tool_id).name if job.tool_id in TOOL_DEFINITIONS else job.tool_id
    result = {
        "scan_id": job.job_id,
        "tool_id": job.tool_id,
        "tool_name": tool_name,
        "status": job.status.upper(),
        "error": job.error,
        "llm_analysis": job.llm_analysis,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
    }
    pdf_bytes = export_tool_scan_pdf(result)
    pdf_filename = tool_scan_report_file_name(result, ".pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{pdf_filename}"'},
    )


@app.post("/tool-scans/targets", response_model=ScanTargetResponse)
async def create_tool_scan_target(
    request: ScanTargetCreateRequest,
    session: AsyncSession = Depends(get_session),
    _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER)),
) -> ScanTargetResponse:
    try:
        return await save_scan_target(request, Repository(session))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tool-scans/targets", response_model=list[ScanTargetResponse])
async def get_tool_scan_targets(
    tool_id: str | None = None, session: AsyncSession = Depends(get_session)
) -> list[ScanTargetResponse]:
    return await list_scan_targets(tool_id, Repository(session))


@app.delete("/tool-scans/targets/{target_id}")
async def delete_tool_scan_target(
    target_id: str,
    session: AsyncSession = Depends(get_session),
    _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER)),
) -> dict[str, object]:
    deleted = await Repository(session).delete_scan_target(target_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scan target not found.")
    return {"deleted": True, "target_id": target_id}


@app.get("/reports")
async def list_reports() -> list[dict[str, Any]]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for path in sorted(REPORT_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = _read_json_file(path)
        reports.append({"filename": path.name, "report": payload})
    return reports


@app.get("/reports/{filename}")
async def get_report(filename: str) -> dict[str, Any]:
    path = _safe_json_path(REPORT_DIR, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return {"filename": path.name, "report": _read_json_file(path)}


@app.delete("/reports/{filename}")
async def delete_report(
    filename: str, _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN, ROLE_MEMBER))
) -> dict[str, object]:
    path = _safe_json_path(REPORT_DIR, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    path.unlink()
    return {"deleted": True, "filename": path.name}


@app.get("/reports/{filename}/pdf")
async def get_report_pdf(filename: str) -> Response:
    path = _safe_json_path(REPORT_DIR, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    report = _read_json_file(path)
    pdf_bytes = export_pdf(report)
    pdf_filename = report_file_name(report, ".pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{pdf_filename}"'},
    )


@app.get("/scenarios")
async def list_scenarios(category: str | None = None) -> dict[str, Any]:
    loader = PluginLoader()
    categories = [category] if category else OWASP_CATEGORY_OPTIONS
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in categories:
        scenarios = loader.discover_scenarios(item)
        grouped[item] = [
            {
                "id": plugin.metadata.id,
                "name": plugin.metadata.name,
                "description": plugin.metadata.description,
                "owasp_category": plugin.metadata.owasp_category,
                "type": plugin.metadata.type,
                "version": plugin.metadata.version,
                "attack_options": _scenario_attack_options(plugin),
                "turn_counts": _scenario_turn_counts(plugin),
            }
            for plugin in scenarios.values()
        ]
    return {"categories": OWASP_CATEGORY_OPTIONS, "scenarios": grouped}


@app.get("/settings/runtime")
async def runtime_settings() -> dict[str, Any]:
    settings = get_settings()
    return _runtime_settings_payload(settings)


@app.put("/settings/runtime")
async def update_runtime_settings(
    request: RuntimeLLMSettingsRequest, _: AuthenticatedUser = Depends(require_role(ROLE_ADMIN))
) -> dict[str, Any]:
    settings = get_settings()
    allowed = {"azure_openai", "aws_bedrock", "ollama", "openai", "huggingface", "anthropic"}
    if request.llm_provider not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported LLM provider.")

    settings.llm_provider = request.llm_provider  # type: ignore[assignment]
    _set_if_present(settings, "azure_openai_endpoint", request.azure_openai_endpoint)
    _set_secret_if_present(settings, "azure_openai_api_key", request.azure_openai_api_key)
    _set_if_present(settings, "azure_openai_deployment", request.azure_openai_deployment)
    _set_if_present(settings, "azure_openai_api_version", request.azure_openai_api_version)
    _set_secret_if_present(settings, "openai_api_key", request.openai_api_key)
    _set_if_present(settings, "openai_model", request.openai_model)
    _set_if_present(settings, "openai_base_url", request.openai_base_url)
    _set_secret_if_present(settings, "anthropic_api_key", request.anthropic_api_key)
    _set_if_present(settings, "anthropic_model", request.anthropic_model)
    _set_if_present(settings, "anthropic_base_url", request.anthropic_base_url)
    _set_secret_if_present(settings, "huggingface_api_key", request.huggingface_api_key)
    _set_if_present(settings, "huggingface_model", request.huggingface_model)
    _set_if_present(settings, "huggingface_base_url", request.huggingface_base_url)
    _set_if_present(settings, "ollama_model", request.ollama_model)
    _set_if_present(settings, "ollama_base_url", request.ollama_base_url)
    _set_if_present(settings, "aws_region", request.aws_region)
    _set_if_present(settings, "aws_bedrock_model_id", request.aws_bedrock_model_id)
    return _runtime_settings_payload(settings)


def _set_if_present(settings: Any, key: str, value: str | None) -> None:
    if value is not None and value != "":
        setattr(settings, key, value)


def _set_secret_if_present(settings: Any, key: str, value: str | None) -> None:
    if value and value != "***configured***":
        setattr(settings, key, SecretStr(value))


def _runtime_settings_payload(settings: Any) -> dict[str, Any]:
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "database_url": settings.database_url,
        "log_level": settings.log_level,
        "report_dir": str(settings.report_dir),
        "llm_provider": settings.llm_provider,
        "llm_ready": settings.llm_ready,
        "azure_openai_endpoint": settings.azure_openai_endpoint or "",
        "azure_openai_deployment": settings.azure_openai_deployment or "",
        "azure_openai_api_version": settings.azure_openai_api_version,
        "azure_ready": settings.azure_ready,
        "openai_model": settings.openai_model,
        "openai_base_url": settings.openai_base_url,
        "openai_ready": settings.openai_ready,
        "anthropic_model": settings.anthropic_model,
        "anthropic_base_url": settings.anthropic_base_url,
        "anthropic_ready": settings.anthropic_ready,
        "huggingface_model": settings.huggingface_model,
        "huggingface_base_url": settings.huggingface_base_url,
        "huggingface_ready": settings.huggingface_ready,
        "ollama_model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_ready": settings.ollama_ready,
        "aws_region": settings.aws_region,
        "aws_bedrock_model_id": settings.aws_bedrock_model_id,
        "aws_bedrock_ready": settings.aws_bedrock_ready,
        "default_temperature": settings.default_temperature,
        "default_timeout_seconds": settings.default_timeout_seconds,
        "default_retry_count": settings.default_retry_count,
    }


def _scenario_attack_options(plugin: Any) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    if callable(getattr(plugin, "attack_definitions", None)):
        for attack in plugin.attack_definitions():
            options.append(
                {
                    "id": str(attack.get("attack_id") or attack.get("chain_id") or attack.get("name")),
                    "label": str(attack.get("name") or attack.get("attack_type") or attack.get("attack_id")),
                    "kind": "attack",
                }
            )
    if callable(getattr(plugin, "multi_turn_attack_definitions", None)):
        for attack in plugin.multi_turn_attack_definitions():
            options.append(
                {
                    "id": str(attack.get("chain_id")),
                    "label": str(attack.get("attack_type") or attack.get("chain_id")),
                    "kind": "chain",
                }
            )
    if not options:
        options.append({"id": plugin.metadata.id, "label": plugin.metadata.name, "kind": "scenario"})
    return options


def _scenario_turn_counts(plugin: Any) -> dict[str, int]:
    scenario_id = plugin.metadata.id
    single = 0
    multi = 0

    if callable(getattr(plugin, "attack_definitions", None)):
        count = len(plugin.attack_definitions())
        if scenario_id in {
            "llm06.excessive_agency",
            "llm07.system_prompt_leakage",
            "llm08.vector_embedding_weaknesses",
            "llm09.misinformation",
            "llm10.unbounded_consumption",
        }:
            single += count
        else:
            single += count

    if callable(getattr(plugin, "multi_turn_attack_definitions", None)):
        multi += len(plugin.multi_turn_attack_definitions())

    if scenario_id == "llm01.crescendo_attack":
        try:
            module = __import__(plugin.__class__.__module__, fromlist=["CRESCENDO_PROFILES"])
            multi += len(getattr(module, "CRESCENDO_PROFILES", {}) or {})
        except Exception:
            multi += 5

    if scenario_id == "llm01.prompt_injection":
        try:
            from core.schemas import ScanSettings

            target_stub = type("TargetStub", (), {"name": "Target"})()
            payloads = plugin.build_payloads(target_stub, ScanSettings(max_turns=25))
            single += len({payload.category for payload in payloads})
        except Exception:
            single += 9

    return {"single_turn": single, "multi_turn": multi, "total": single + multi}
