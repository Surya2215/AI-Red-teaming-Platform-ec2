"""Structured JSON logging configuration."""

import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from pythonjsonlogger import json as jsonlogger

from core.config import get_settings


class RedactingJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter that masks common secret fields."""

    SECRET_KEYS = {"api_key", "authorization", "token", "password", "secret", "AZURE_OPENAI_API_KEY"}

    def process_log_record(self, log_record: dict[str, Any]) -> dict[str, Any]:
        log_record["timestamp"] = datetime.now(UTC).isoformat()
        for key in list(log_record):
            if key.lower() in {item.lower() for item in self.SECRET_KEYS}:
                log_record[key] = "***REDACTED***"
        return log_record


def configure_logging() -> None:
    """Configure root JSON logging once."""

    settings = get_settings()
    root = logging.getLogger()
    if getattr(root, "_redteam_configured", False):
        return

    root.setLevel(settings.log_level)
    fmt = "%(timestamp)s %(levelname)s %(name)s %(message)s %(scan_id)s %(target)s %(scenario)s"
    formatter = RedactingJsonFormatter(fmt)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.log_dir / "platform.jsonl",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.handlers.clear()
    root.addHandler(stream)
    root.addHandler(file_handler)
    setattr(root, "_redteam_configured", True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""

    configure_logging()
    return logging.getLogger(name)
