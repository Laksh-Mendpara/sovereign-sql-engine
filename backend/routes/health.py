"""Health, readiness, and metrics endpoints.

GET /health   — liveness probe (no external calls, always fast)
GET /ready    — readiness probe (pings all three external services in parallel)
GET /metrics  — current in-process metrics snapshot
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse

from ..services.metadata_service import MetadataService
from ..services.neo4j_service import Neo4jService
from ..services.pinecone_service import PineconeService

from ..config import BackendSettings
from ..dependencies import (
    get_metadata_service,
    get_neo4j_service,
    get_pinecone_service,
    get_settings,
)
from ..logging_config import get_logger
from ..metrics import get_metrics, render_prometheus_metrics
from ..models import HealthResponse, ReadinessResponse, ServiceStatus

router = APIRouter(tags=["observability"])
logger = get_logger(__name__)

_PROCESS_START = time.time()


# ── Liveness ───────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 immediately. Load balancers use this to confirm the process is alive.",
)
async def health(settings: BackendSettings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        uptime_seconds=round(time.time() - _PROCESS_START, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Readiness ──────────────────────────────────────────────────────────────────

@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Pings Pinecone, Neo4j, and the metadata DB concurrently. "
        "Returns 200 if all services are reachable, 503 if any critical service is down."
    ),
)
async def ready(
    pinecone_svc: PineconeService = Depends(get_pinecone_service),
    neo4j_svc: Neo4jService = Depends(get_neo4j_service),
    metadata_svc: MetadataService = Depends(get_metadata_service),
) -> JSONResponse:
    results = await asyncio.gather(
        _ping_pinecone(pinecone_svc),
        _ping_neo4j(neo4j_svc),
        _ping_metadata(metadata_svc),
        return_exceptions=True,
    )

    services: list[ServiceStatus] = []
    for r in results:
        if isinstance(r, ServiceStatus):
            services.append(r)
        else:
            services.append(ServiceStatus(service="unknown", status="down", detail=str(r)))

    all_ok = all(s.status == "ok" for s in services)
    any_down = any(s.status == "down" for s in services)

    if all_ok:
        overall, http_code = "ready", 200
    elif any_down:
        overall, http_code = "not_ready", 503
    else:
        overall, http_code = "degraded", 200

    return JSONResponse(
        content=ReadinessResponse(status=overall, services=services).model_dump(),
        status_code=http_code,
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

@router.get(
    "/metrics",
    summary="Metrics snapshot",
    description="In-process request counts and per-stage latency histograms (p50/p95/p99).",
)
async def metrics_snapshot() -> dict[str, Any]:
    return get_metrics().snapshot()


@router.get(
    "/metrics/prometheus",
    summary="Prometheus metrics",
    description="Prometheus exposition format for scraping and Grafana Cloud remote_write.",
)
async def metrics_prometheus() -> PlainTextResponse:
    try:
        return PlainTextResponse(render_prometheus_metrics(), media_type="text/plain; version=0.0.4")
    except Exception as exc:
        logger.exception("metrics.prometheus_render_failed", extra={"error": str(exc)})
        fallback = (
            "# HELP sovereign_sql_exporter_up Exporter health.\n"
            "# TYPE sovereign_sql_exporter_up gauge\n"
            "sovereign_sql_exporter_up 0\n"
        )
        return PlainTextResponse(fallback, media_type="text/plain; version=0.0.4")


# ── Service ping helpers (each runs in a thread to stay non-blocking) ──────────

async def _ping_pinecone(svc: PineconeService) -> ServiceStatus:
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(svc.index.describe_index_stats),
            timeout=5.0,
        )
        lat = round((time.perf_counter() - t0) * 1_000, 2)
        return ServiceStatus(service="pinecone", status="ok", latency_ms=lat)
    except Exception as exc:
        lat = round((time.perf_counter() - t0) * 1_000, 2)
        logger.warning("ready.pinecone_fail", extra={"error": str(exc)})
        return ServiceStatus(service="pinecone", status="down", latency_ms=lat, detail=str(exc))


async def _ping_neo4j(svc: Neo4jService) -> ServiceStatus:
    t0 = time.perf_counter()
    try:
        def _ping() -> None:
            with svc.driver.session() as session:
                session.run("RETURN 1")

        await asyncio.wait_for(asyncio.to_thread(_ping), timeout=5.0)
        lat = round((time.perf_counter() - t0) * 1_000, 2)
        return ServiceStatus(service="neo4j", status="ok", latency_ms=lat)
    except Exception as exc:
        lat = round((time.perf_counter() - t0) * 1_000, 2)
        logger.warning("ready.neo4j_fail", extra={"error": str(exc)})
        return ServiceStatus(service="neo4j", status="down", latency_ms=lat, detail=str(exc))


async def _ping_metadata(svc: MetadataService) -> ServiceStatus:
    t0 = time.perf_counter()
    try:
        def _ping() -> None:
            svc.conn.execute("SELECT 1")

        await asyncio.wait_for(asyncio.to_thread(_ping), timeout=5.0)
        lat = round((time.perf_counter() - t0) * 1_000, 2)
        return ServiceStatus(service="metadata_db", status="ok", latency_ms=lat)
    except Exception as exc:
        lat = round((time.perf_counter() - t0) * 1_000, 2)
        logger.warning("ready.metadata_fail", extra={"error": str(exc)})
        return ServiceStatus(service="metadata_db", status="down", latency_ms=lat, detail=str(exc))
