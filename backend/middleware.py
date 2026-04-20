"""ASGI middleware stack.

Applied outermost → innermost (so RequestContext runs before Timing):
  1. RequestContextMiddleware — generates/propagates request_id + trace_id
  2. TimingMiddleware         — records E2E duration, emits structured access log

Both propagate IDs through contextvars so every downstream log line
within the same async context carries the correct correlation fields.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logging_config import (
    get_logger,
    new_request_id,
    new_trace_id,
    set_request_context,
)

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Inject request_id and trace_id into async context + response headers.

    - X-Request-ID : ULID generated per request (time-sortable)
    - X-Trace-ID   : echoes caller header or auto-generates UUID4
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = new_request_id()
        trace_id = request.headers.get("x-trace-id") or new_trace_id()

        set_request_context(request_id, trace_id)

        # Attach to request.state so route handlers can read them directly
        request.state.request_id = request_id
        request.state.trace_id = trace_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure end-to-end latency and emit a structured access-log line.

    SSE streams (Content-Type: text/event-stream) are long-lived connections,
    so we log immediately at response creation time rather than waiting for the
    stream to close (which would never emit a log line until disconnect).
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - t0) * 1_000, 2)

        is_sse = "text/event-stream" in response.headers.get("content-type", "")

        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

        logger.info(
            "access",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                # For SSE, elapsed_ms is only handshake time, not stream duration
                "duration_ms": elapsed_ms,
                "stream": is_sse,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
            },
        )
        return response

