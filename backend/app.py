"""FastAPI application factory with lifespan, middleware, and exception handlers.

Start the server
────────────────
  # Development (auto-reload, plain-text logs)
  BACKEND_DEBUG=true uvicorn backend.app:app --reload --port 8000

  # Production
  uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1

  # Via pyproject script
  python -m backend.app
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .services.classifier_service import ClassifierService
from .services.guard_service import GuardService
from .services.metadata_service import MetadataService
from .services.modal_chat import ModalChatClient
from .services.observability_service import ObservabilityService
from .services.neo4j_service import Neo4jService
from .services.pinecone_service import PineconeService
from .services.sql_execution_service import SqlExecutionService
from .services.runpod_service import RunpodService

from .config import BackendSettings, get_settings
from .logging_config import configure_logging, get_logger, get_request_id, get_trace_id, set_service_context
from .metrics import get_metrics
from .middleware import RequestContextMiddleware, TimingMiddleware
from .routes.health import router as health_router
from .routes.pipeline import router as pipeline_router

logger = get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise all service singletons once at startup; tear-down at shutdown."""
    settings: BackendSettings = app.state.settings
    log = get_logger("lifespan")
    log.info("startup.begin", extra={"app_version": settings.app_version})

    # Build all services in a thread (they use blocking drivers)
    def _init() -> dict:
        log.info("startup.initialising_services")
        guard_client = ModalChatClient(settings.llama_guard_url, settings.llama_guard_model, logger)
        classifier_client = ModalChatClient(settings.phi4_url, settings.phi4_model, logger)

        return {
            "guard_service": GuardService(guard_client, logger),
            "classifier_service": ClassifierService(classifier_client, logger),
            "metadata_service": MetadataService(settings.sqlite_conn_str, logger),
            "sql_execution_service": SqlExecutionService(settings.sqlite_data_conn_str, logger),
            "observability_service": ObservabilityService(settings.sqlite_observability_conn_str, logger),
            "pinecone_service": PineconeService(
                api_key=settings.pinecone_api_key,
                index_name=settings.pinecone_index_name,
                index_host=settings.pinecone_index_host,
                namespace=settings.pinecone_namespace,
                embed_model=settings.pinecone_embed_model,
                rerank_model=settings.pinecone_rerank_model,
                db_name=settings.db_name,
                logger=logger,
            ),
            "neo4j_service": Neo4jService(
                settings.neo4j_url,
                settings.neo4j_username,
                settings.neo4j_password,
                settings.db_name,
                logger,
            ),
            "runpod_service": RunpodService(
                api_key=settings.runpod_api_key,
                endpoint_id=settings.runpod_endpoint_id,
                base_url=settings.runpod_base_url,
                poll_interval=settings.runpod_status_poll_interval,
                timeout_seconds=settings.runpod_status_timeout,
                logger=logger,
            ),
        }

    svcs = await asyncio.to_thread(_init)
    for key, obj in svcs.items():
        setattr(app.state, key, obj)

    # Warm up the metrics singleton
    get_metrics()

    log.info("startup.complete")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("shutdown.begin")

    def _cleanup() -> None:
        for name, closer in [
            ("neo4j", "neo4j_service"),
            ("metadata", "metadata_service"),
            ("sqlite_data", "sql_execution_service"),
            ("observability", "observability_service"),
        ]:
            try:
                getattr(svcs[closer], "close")()
                log.info(f"shutdown.{name}_closed")
            except Exception as exc:
                log.warning(f"shutdown.{name}_close_error", extra={"error": str(exc)})

    await asyncio.to_thread(_cleanup)
    log.info("shutdown.complete")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app(settings: BackendSettings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    configure_logging(
        level=settings.log_level,
        use_json=not settings.debug,
        service_name=settings.app_name,
        environment="development" if settings.debug else "production",
    )
    set_service_context(settings.app_name, "development" if settings.debug else "production")

    app = FastAPI(
        title="Sovereign SQL Engine — Pipeline API",
        description=(
            "Production-grade Text2SQL pipeline with async parallel orchestration, "
            "structured JSON logging, distributed trace context (request_id / trace_id / span_id), "
            "and per-stage latency metrics."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store settings on app.state for Depends() injection
    app.state.settings = settings

    # ── Middleware (registered top-down = applied outer-to-inner) ────────────
    app.add_middleware(TimingMiddleware)         # inner  → runs 2nd on ingress, 1st on egress
    app.add_middleware(RequestContextMiddleware) # outer  → runs 1st on ingress, 2nd on egress
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Trace-ID", "X-Response-Time"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(pipeline_router)

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(StarletteHTTPException)
    async def http_exc_handler(req: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        detail.setdefault("request_id", getattr(req.state, "request_id", get_request_id()))
        detail.setdefault("trace_id", getattr(req.state, "trace_id", get_trace_id()))
        return JSONResponse(status_code=exc.status_code, content=detail)

    @app.exception_handler(RequestValidationError)
    async def validation_exc_handler(req: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("request.validation_error", extra={"errors": exc.errors()})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": exc.errors(),
                "request_id": getattr(req.state, "request_id", get_request_id()),
                "trace_id": getattr(req.state, "trace_id", get_trace_id()),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exc_handler(req: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", extra={"path": req.url.path})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "detail": str(exc),
                "request_id": getattr(req.state, "request_id", get_request_id()),
                "trace_id": getattr(req.state, "trace_id", get_trace_id()),
            },
        )

    logger.info(
        "app.created",
        extra={"app_version": settings.app_version, "debug": settings.debug},
    )
    return app


# ── Module-level instance (used by uvicorn / gunicorn) ────────────────────────
app = create_app()


# ── Script entry ──────────────────────────────────────────────────────────────

def run_server() -> None:
    """Entry point: ``python -m backend.app`` or pyproject ``[project.scripts]``."""
    cfg = get_settings()
    configure_logging(
        level=cfg.log_level,
        use_json=not cfg.debug,
        service_name=cfg.app_name,
        environment="development" if cfg.debug else "production",
    )
    set_service_context(cfg.app_name, "development" if cfg.debug else "production")
    uvicorn.run(
        "backend.app:app",
        host=cfg.host,
        port=cfg.port,
        workers=cfg.workers,
        log_level=cfg.log_level.lower(),
        access_log=False,       # TimingMiddleware handles access logs
        reload=cfg.debug,
    )


if __name__ == "__main__":
    run_server()
