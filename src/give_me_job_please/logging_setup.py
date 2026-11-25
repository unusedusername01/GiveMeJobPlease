"""Structured JSON logging (structlog) with trace_id binding and secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog

# Patterns that must never appear in stdout or error logs
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-or-v1-[A-Za-z0-9_-]{12,}"), "[REDACTED_OPENROUTER_KEY]"),
    (re.compile(r"sk-or-[A-Za-z0-9_-]{12,}"), "[REDACTED_OPENROUTER_KEY]"),
    (re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"), "[REDACTED_TELEGRAM_TOKEN]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{12,}", re.IGNORECASE), r"\1[REDACTED]"),
]

_SENSITIVE_KEYS = {
    "api_key",
    "openrouter_key",
    "openrouter_key_enc",
    "code_verifier",
    "encryption_key",
    "telegram_bot_token",
    "authorization",
    "token",
    "secret",
}


def redact_string(value: str) -> str:
    """Replace known secret token patterns in arbitrary strings."""
    result = value
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _redact_recursive(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_string(obj)
    if isinstance(obj, dict):
        cleaned: dict[Any, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in _SENSITIVE_KEYS:
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = _redact_recursive(v)
        return cleaned
    if isinstance(obj, list):
        return [_redact_recursive(item) for item in obj]
    return obj


def redact_secrets_processor(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor that strips secrets from all logged fields and tracebacks."""
    return _redact_recursive(event_dict)


def configure_logging(log_dir: Path, log_level: str = "INFO") -> Path:
    """Configure structlog JSON output to stdout and a rotating errors.jsonl file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = log_dir / "errors.jsonl"

    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
        redact_secrets_processor,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(ensure_ascii=False),
        foreign_pre_chain=shared_processors,
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)

    error_file_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=2 * 1024 * 1024,  # 2 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(error_file_handler)

    # Quiet noisy third-party loggers unless debugging
    for noisy in ("httpx", "httpcore", "aiogram.event", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return error_log_path


def read_recent_error_logs(error_log_path: Path, limit: int = 5) -> list[dict[str, Any]]:
    """Read the last `limit` structured JSON error entries for the /logs owner command."""
    if not error_log_path.exists():
        return []
    lines = error_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: list[dict[str, Any]] = []
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError:
            entries.append({"event": raw})
        if len(entries) >= limit:
            break
    return entries
