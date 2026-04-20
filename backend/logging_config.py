"""Structured JSON logging with per-request context propagation.

Every log record automatically includes:
  - request_id  : ULID — time-sortable, globally unique per HTTP request
  - trace_id    : propagated from X-Trace-ID header or auto-generated UUID4
  - span_id     : per service-call UUID4 for distributed tracing correlation
  - span_name   : human label of the current pipeline stage

Context propagates automatically into ``asyncio.to_thread()`` tasks via
Python's ``contextvars`` module — no manual plumbing required.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

# ── Per-request context vars ───────────────────────────────────────────────────
_request_id: ContextVar[str] = ContextVar("request_id", default="")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_span_id: ContextVar[str] = ContextVar("span_id", default="")
_span_name: ContextVar[str] = ContextVar("span_name", default="")
_service_name: ContextVar[str] = ContextVar("service_name", default="sovereign-sql-backend")
_environment: ContextVar[str] = ContextVar("environment", default="development")


# ── Context helpers ────────────────────────────────────────────────────────────

def set_request_context(request_id: str, trace_id: str) -> None:
    _request_id.set(request_id)
    _trace_id.set(trace_id)


def set_span_context(span_id: str, span_name: str) -> None:
    _span_id.set(span_id)
    _span_name.set(span_name)


def clear_span_context() -> None:
    _span_id.set("")
    _span_name.set("")


def get_request_id() -> str:
    return _request_id.get()


def get_trace_id() -> str:
    return _trace_id.get()


def new_request_id() -> str:
    """ULID (time-sortable) with UUID4 fallback."""
    try:
        from ulid import ULID  # type: ignore[import-untyped]
        return str(ULID())
    except ImportError:
        return uuid.uuid4().hex.upper()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def set_service_context(service_name: str, environment: str) -> None:
    _service_name.set(service_name)
    _environment.set(environment)


# ── JSON log formatter ─────────────────────────────────────────────────────────

# Fields that are built-in LogRecord attributes — never re-emitted as extras
_LOGRECORD_BUILTIN_KEYS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)))


class JsonFormatter(logging.Formatter):
    """Emit each log record as a compact single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        payload: dict[str, Any] = {
            "timestamp": self._iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "service_name": _service_name.get(),
            "environment": _environment.get(),
            "request_id": _request_id.get(),
            "trace_id": _trace_id.get(),
            "span_id": _span_id.get(),
            "span_name": _span_name.get(),
            "filename": record.filename,
            "lineno": record.lineno,
        }

        # Attach any caller-supplied extra fields (skip built-ins + already set)
        for key, value in record.__dict__.items():
            if key in _LOGRECORD_BUILTIN_KEYS or key in payload or key.startswith("_"):
                continue
            payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)

    @staticmethod
    def _iso(ts: float) -> str:
        t = time.gmtime(ts)
        ms = int((ts - int(ts)) * 1000)
        return (
            f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T"
            f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}Z"
        )


# ── Root logger setup ──────────────────────────────────────────────────────────

def configure_logging(
    level: str = "INFO",
    use_json: bool = True,
    service_name: str = "sovereign-sql-backend",
    environment: str = "development",
) -> None:
    """Configure root logger once at application startup."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    set_service_context(service_name, environment)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(
        JsonFormatter() if use_json
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for lib in ("urllib3", "httpx", "neo4j", "pinecone", "httpcore", "uvicorn.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
