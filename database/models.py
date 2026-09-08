"""Database models designed to migrate from SQLite to PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class TargetRecord(Base):
    __tablename__ = "targets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    url: Mapped[str] = mapped_column(String(2048))
    method: Mapped[str] = mapped_column(String(12))
    headers: Mapped[dict] = mapped_column(JSON, default=dict)
    request_template: Mapped[dict] = mapped_column(JSON, default=dict)
    auth: Mapped[dict] = mapped_column(JSON, default=dict)
    timeout_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ScanRecord(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    scan_name: Mapped[str] = mapped_column(String(180), index=True)
    target_name: Mapped[str] = mapped_column(String(160), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)

    results: Mapped[list[ScanResultRecord]] = relationship(back_populates="scan", cascade="all, delete-orphan")
    logs: Mapped[list[AttackLogRecord]] = relationship(back_populates="scan", cascade="all, delete-orphan")


class ScanResultRecord(Base):
    __tablename__ = "scan_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[str] = mapped_column(String(64), ForeignKey("scans.scan_id"), index=True)
    scenario_id: Mapped[str] = mapped_column(String(160), index=True)
    scenario_name: Mapped[str] = mapped_column(String(180))
    owasp_category: Mapped[str] = mapped_column(String(120), index=True)
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    scan: Mapped[ScanRecord] = relationship(back_populates="results")
    detector_results: Mapped[list[DetectorResultRecord]] = relationship(back_populates="scan_result", cascade="all, delete-orphan")


class AttackLogRecord(Base):
    __tablename__ = "attack_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[str] = mapped_column(String(64), ForeignKey("scans.scan_id"), index=True)
    scenario_id: Mapped[str] = mapped_column(String(160), index=True)
    turn: Mapped[int] = mapped_column(Integer)
    stage: Mapped[str] = mapped_column(String(120))
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    scan: Mapped[ScanRecord] = relationship(back_populates="logs")


class DetectorResultRecord(Base):
    __tablename__ = "detector_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_result_id: Mapped[int] = mapped_column(Integer, ForeignKey("scan_results.id"), index=True)
    detector_id: Mapped[str] = mapped_column(String(180), index=True)
    vulnerable: Mapped[bool] = mapped_column()
    confidence: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(32), index=True)
    reason: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    scan_result: Mapped[ScanResultRecord] = relationship(back_populates="detector_results")


class ScanTargetRecord(Base):
    """A saved, reusable Garak/PyRIT/DeepTeam target. Credentials are Fernet-encrypted
    (see core/crypto.py) before being stored in encrypted_credentials - never plaintext."""

    __tablename__ = "scan_targets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    tool_id: Mapped[str] = mapped_column(String(32), index=True)
    target_type: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(160))
    # Garak's --target_name (model/deployment identifier) - a distinct concept from the
    # credential fields below (e.g. Azure's AZURE_MODEL_NAME env var). Not used by PyRIT,
    # whose equivalent (deployment_name) already lives inside encrypted_credentials.
    target_name: Mapped[str] = mapped_column(String(200), default="")
    encrypted_credentials: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class ScanJobRecord(Base):
    """One async Tool Scan run. request_payload is the validated ToolScanRequest with
    any credential fields Fernet-encrypted before storage - decrypted only inside the
    Celery worker task, never logged, never returned to the API caller."""

    __tablename__ = "scan_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    tool_id: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    request_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    report_paths: Mapped[list] = mapped_column(JSON, default=list)
    llm_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    findings: Mapped[list[ScanFindingRecord]] = relationship(back_populates="job", cascade="all, delete-orphan")


class UserRecord(Base):
    """A logged-in platform user. Only exists once AUTH_ENABLED is turned on
    (core/config.py) - see core/auth.py for password hashing/JWT issuance and the
    bootstrap-admin flow that creates the first row on startup. Distinct from the
    legacy X-API-Key middleware (still supported for machine/CI clients) - that
    grants full access with no identity attached, this grants a role-scoped,
    attributable identity used for RBAC checks and the audit log below."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    display_name: Mapped[str] = mapped_column(String(160), default="")
    # "admin" | "member" | "viewer" - see core/auth.py's ROLE_* constants for what
    # each can do. Not a DB enum so a new role can be added without a migration.
    role: Mapped[str] = mapped_column(String(20), default="member", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLogRecord(Base):
    """Immutable trail of who did what, for production/multi-user deployments.
    Rows are append-only (never updated or deleted by the app itself) so this
    stays a trustworthy record even if a user's account is later disabled.
    user_email is denormalized (kept even if the user row is later removed) so
    history remains readable after account deletion."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_email: Mapped[str] = mapped_column(String(320), default="")
    # e.g. "login.success", "login.failed", "scan.create", "scan.delete",
    # "target.update", "user.role_change", "settings.update" - dotted
    # "<resource>.<verb>" convention, kept free-text so new actions don't need a schema change.
    action: Mapped[str] = mapped_column(String(80), index=True)
    resource_type: Mapped[str] = mapped_column(String(40), default="", index=True)
    resource_id: Mapped[str] = mapped_column(String(160), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    ip_address: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)


class ScanFindingRecord(Base):
    __tablename__ = "scan_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("scan_jobs.id"), index=True)
    probe: Mapped[str] = mapped_column(String(200))
    detector: Mapped[str] = mapped_column(String(200))
    passed: Mapped[bool] = mapped_column(Boolean)
    score: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    severity: Mapped[str] = mapped_column(String(16), default="info", index=True)
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    # Groups findings that belong to the same attack/exchange so the UI can render them
    # as a single conversation (see worker/result_parser.py). Garak: each attempt gets
    # its own synthetic conversation_id (always exactly one turn). PyRIT: all turns of
    # one multi-turn attack (e.g. crescendo) share a conversation_id, ordered by `turn`.
    conversation_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    turn: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    job: Mapped[ScanJobRecord] = relationship(back_populates="findings")
