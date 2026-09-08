"""Production user auth, RBAC, and audit logging - see database.models.UserRecord
and AuditLogRecord for the underlying tables.

Opt-in via AUTH_ENABLED (core/config.py), and additive: it coexists with the
existing X-API-Key middleware (core/api.py's require_api_key). API-key requests
stay full-access and unattributed (for CI/automation clients that have no notion
of a human user); Authorization: Bearer <jwt> requests carry a real user
identity, which is what RBAC checks and the audit log key off of.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from database.models import UserRecord
from database.repository import Repository
from database.session import get_session

ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"
ROLE_VIEWER = "viewer"
ALL_ROLES = (ROLE_ADMIN, ROLE_MEMBER, ROLE_VIEWER)

_JWT_ALGORITHM = "HS256"

logger = logging.getLogger(__name__)


class LoginRateLimiter:
    """Fixed-window brute-force guard for /auth/login, keyed independently by
    client IP and by attempted email so neither a single IP hammering many
    accounts nor many IPs hammering one account slips through. In-memory and
    per-process by design (no new infra dependency for a first line of
    defense) - on a multi-replica deployment each replica enforces its own
    window rather than a shared one, which is a real gap; a Redis/Valkey-backed
    limiter (both already in the stack as the Celery broker/backend) would
    close it if this ever runs behind a load balancer with >1 API replica."""

    def __init__(self, max_attempts: int = 10, window_seconds: float = 300.0) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits[key]
        cutoff = now - self.window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()
        return hits

    def check(self, *keys: str) -> None:
        """Raises HTTPException(429) if any of the given keys is already over
        the limit; otherwise records this attempt against every key."""

        now = time.monotonic()
        for key in keys:
            if not key:
                continue
            hits = self._prune(key, now)
            if len(hits) >= self.max_attempts:
                raise HTTPException(
                    status_code=429,
                    detail="Too many login attempts. Please wait a few minutes and try again.",
                )
        for key in keys:
            if key:
                self._hits[key].append(now)


_login_rate_limiter = LoginRateLimiter()


def check_login_rate_limit(*keys: str) -> None:
    _login_rate_limiter.check(*keys)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # A malformed/legacy hash should fail closed, not raise 500s at login.
        return False


def create_access_token(user: UserRecord) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any] | None:
    settings = get_settings()
    if not settings.jwt_secret:
        return None
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[_JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


class AuthenticatedUser:
    """Lightweight identity attached to a request - deliberately not the SQLAlchemy
    UserRecord itself, since request handlers shouldn't hold a live ORM object
    tied to a session that may have already been closed by the time they run."""

    __slots__ = ("id", "email", "role")

    def __init__(self, id: str, email: str, role: str) -> None:
        self.id = id
        self.email = email
        self.role = role


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip() or None


async def get_current_user(request: Request, session: AsyncSession = Depends(get_session)) -> AuthenticatedUser | None:
    """Best-effort resolution of the caller's identity from an Authorization:
    Bearer header - returns None (not a 401) when there's no/invalid token, so
    it's safe to use on endpoints that work for both authenticated users and
    plain API-key callers. Endpoints that must have a real user use
    require_user()/require_role() below instead."""

    token = _bearer_token(request)
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    record = await Repository(session).get_user(user_id)
    if record is None or not record.is_active:
        return None
    return AuthenticatedUser(id=record.id, email=record.email, role=record.role)


# Synthetic identity used in place of a real user whenever AUTH_ENABLED is off -
# keeps require_user()/require_role() as pure no-ops on a deployment that hasn't
# turned auth on, so adding these dependencies to existing routes never breaks
# today's default (no-login) behavior. Also gives audit-log entries written while
# auth is off an honest, obviously-synthetic actor rather than a blank/None user.
_AUTH_DISABLED_USER = AuthenticatedUser(id="", email="system (auth disabled)", role=ROLE_ADMIN)


async def require_user(user: AuthenticatedUser | None = Depends(get_current_user)) -> AuthenticatedUser:
    if not get_settings().auth_enabled:
        return _AUTH_DISABLED_USER
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def require_role(*roles: str):
    """Dependency factory: raises 403 unless the current user's role is one of
    `roles`. Apply to the specific mutating/sensitive endpoints that need it -
    plain require_user (any authenticated user, any role) covers everything
    else. A no-op (like require_user) while AUTH_ENABLED is off, so retrofitting
    this onto existing routes never breaks a deployment that hasn't turned auth on."""

    async def _dependency(user: AuthenticatedUser = Depends(require_user)) -> AuthenticatedUser:
        if user is _AUTH_DISABLED_USER:
            return user
        if user.role not in roles:
            raise HTTPException(status_code=403, detail=f"This action requires one of these roles: {', '.join(roles)}.")
        return user

    return _dependency


async def record_audit(
    session: AsyncSession,
    *,
    user: AuthenticatedUser | None,
    action: str,
    resource_type: str = "",
    resource_id: str = "",
    detail: dict[str, Any] | None = None,
    request: Request | None = None,
    unauthenticated_actor: str = "api-key",
) -> None:
    """Best-effort audit write - a formatting/DB hiccup here must never break the
    request it's logging. `user=None` means there's no logged-in identity for this
    entry; `unauthenticated_actor` labels who/what acted instead - defaults to
    "api-key" (the legacy X-API-Key path), but callers logging a failed login
    should pass the attempted email so the entry doesn't misleadingly claim an
    API key was involved. A no-op while AUTH_ENABLED is off, so a default/dev
    deployment's audit_log table doesn't fill up with noise for a feature it
    hasn't turned on."""

    if not get_settings().auth_enabled:
        return
    try:
        ip_address = request.client.host if request and request.client else ""
        await Repository(session).add_audit_entry(
            user_id=user.id if user else None,
            user_email=user.email if user else unauthenticated_actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            detail=detail,
            ip_address=ip_address,
        )
    except Exception:
        logger.exception("Failed to write audit log entry for action=%s", action)


async def bootstrap_admin(session: AsyncSession) -> None:
    """Creates the first admin user on startup from INITIAL_ADMIN_EMAIL/
    INITIAL_ADMIN_PASSWORD if the users table is still empty. No-op once any
    user exists, and a no-op entirely when AUTH_ENABLED is off - there is no
    public signup, so this is the only way to get a first login."""

    settings = get_settings()
    if not settings.auth_enabled:
        return
    repo = Repository(session)
    if await repo.count_users() > 0:
        return
    if not settings.initial_admin_email or not settings.initial_admin_password:
        logger.warning(
            "AUTH_ENABLED is true but no users exist yet and INITIAL_ADMIN_EMAIL/"
            "INITIAL_ADMIN_PASSWORD are not set - no one will be able to log in. "
            "Set both env vars and restart to create the first admin account."
        )
        return
    await repo.create_user(
        email=settings.initial_admin_email,
        password_hash=hash_password(settings.initial_admin_password),
        display_name="Administrator",
        role=ROLE_ADMIN,
    )
    logger.info("Bootstrapped initial admin user %s", settings.initial_admin_email)
