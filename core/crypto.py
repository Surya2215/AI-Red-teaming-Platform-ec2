"""Fernet symmetric encryption for credentials persisted at rest (scan_targets.encrypted_credentials,
scan_jobs.request_payload). Decrypted values must never be logged or returned to API callers -
callers should decrypt only inside the Celery worker task, immediately before injecting them as
subprocess-scoped env vars (see engine/tool_scan.py::_tool_env for the existing scoping pattern)."""

from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from core.config import get_settings


class EncryptionKeyMissingError(RuntimeError):
    pass


@lru_cache
def _fernet() -> Fernet:
    key = get_settings().encryption_key.strip()
    if not key:
        raise EncryptionKeyMissingError(
            "ENCRYPTION_KEY is not configured. Generate one with: "
            'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        )
    return Fernet(key.encode("utf-8"))


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt(value: str) -> str:
    try:
        return _fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Could not decrypt value: invalid token or wrong ENCRYPTION_KEY.") from exc
