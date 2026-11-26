"""Structured JSON logging and JSONL audit log writer for MCP calls."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def write_audit_log(
    audit_log_path: str | Path,
    *,
    tool: str,
    args: dict[str, Any],
    status: str,
    duration_ms: float,
    error: str | None = None,
    row_count: int | None = None,
) -> None:
    """Append a single JSONL audit entry without contending on SQLite locks."""
    path = Path(audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool": tool,
        "args": args,
        "status": status,
        "duration_ms": round(duration_ms, 2),
    }
    if row_count is not None:
        entry["row_count"] = row_count
    if error is not None:
        entry["error"] = error
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
