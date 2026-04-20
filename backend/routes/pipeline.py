import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from ..services.classifier_service import ClassifierService
from ..services.guard_service import GuardService
from ..services.metadata_service import MetadataService
from ..services.observability_service import ObservabilityService
from ..services.neo4j_service import Neo4jService
from ..services.pinecone_service import PineconeService
from ..services.sql_execution_service import SqlExecutionService
from ..services.runpod_service import RunpodService

from ..config import BackendSettings
from ..dependencies import (
    get_classifier_service,
    get_guard_service,
    get_metadata_service,
    get_observability_service,
    get_neo4j_service,
    get_pinecone_service,
    get_sql_execution_service,
    get_runpod_service,
    get_settings,
)
from ..logging_config import get_logger, get_request_id, get_trace_id
from ..models import ErrorResponse, FeedbackRequest, FeedbackResponse, PipelineRequest, PipelineResponse
from ..pipeline_executor import AsyncPipelineExecutor
from ..sse_executor import SSEPipelineExecutor

router = APIRouter(prefix="/v1/pipeline", tags=["pipeline"])
logger = get_logger(__name__)


@router.post(
    "/query",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the full Text2SQL pipeline",
    description=(
        "Executes the full Text2SQL pipeline with maximum parallelism:\n\n"
        "- **Guard** (safety check) and **Classifier** run *concurrently* with **Pinecone** retrieval\n"
        "- **Neo4j** join expansion follows Pinecone\n"
        "- **Schema SQL** generation follows Neo4j\n"
        "- **Arctic** (RunPod) SQL generation follows schema\n\n"
        "Every response includes a `metrics` block with per-stage wall-clock timings.\n"
        "Every log line carries `request_id`, `trace_id`, and `span_id` for full observability."
    ),
    responses={
        200: {"description": "Pipeline completed successfully"},
        422: {"model": ErrorResponse, "description": "Request validation error"},
        504: {"model": ErrorResponse, "description": "Pipeline timed out"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def run_pipeline(
    body: PipelineRequest,
    request: Request,
    settings: BackendSettings = Depends(get_settings),
    guard_svc: GuardService = Depends(get_guard_service),
    classifier_svc: ClassifierService = Depends(get_classifier_service),
    pinecone_svc: PineconeService = Depends(get_pinecone_service),
    neo4j_svc: Neo4jService = Depends(get_neo4j_service),
    metadata_svc: MetadataService = Depends(get_metadata_service),
    sql_execution_svc: SqlExecutionService = Depends(get_sql_execution_service),
    runpod_svc: RunpodService = Depends(get_runpod_service),
    observability_svc: ObservabilityService = Depends(get_observability_service),
) -> PipelineResponse:
    # Pull correlation IDs set by RequestContextMiddleware
    request_id: str = getattr(request.state, "request_id", None) or get_request_id()
    trace_id: str = getattr(request.state, "trace_id", None) or get_trace_id()

    # Caller may override trace_id via request body
    if body.trace_id:
        trace_id = body.trace_id

    logger.info(
        "pipeline.request",
        extra={
            "query_preview": body.query[:80],
            "query_length": len(body.query),
        },
    )

    executor = AsyncPipelineExecutor(
        settings=settings,
        guard_service=guard_svc,
        classifier_service=classifier_svc,
        pinecone_service=pinecone_svc,
        neo4j_service=neo4j_svc,
        metadata_service=metadata_svc,
        sql_execution_service=sql_execution_svc,
        runpod_service=runpod_svc,
        observability_service=observability_svc,
    )

    try:
        return await executor.execute(
            query=body.query,
            request_id=request_id,
            trace_id=trace_id,
        )

    except asyncio.TimeoutError:
        logger.error("pipeline.request.timeout")
        await asyncio.to_thread(
            observability_svc.upsert_request_record,
            {
                "request_id": request_id,
                "trace_id": trace_id,
                "query": body.query,
                "execution_status": "timeout",
                "terminal_state": "timeout",
                "error_type": "timeout",
                "error_message": f"Pipeline did not complete within {settings.total_pipeline_timeout}s",
                "metadata_json": {"source": "pipeline.query"},
            },
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": "pipeline_timeout",
                "detail": f"Pipeline did not complete within {settings.total_pipeline_timeout}s",
                "request_id": request_id,
                "trace_id": trace_id,
            },
        )

    except Exception as exc:
        logger.exception("pipeline.request.error", extra={"error": str(exc)})
        await asyncio.to_thread(
            observability_svc.upsert_request_record,
            {
                "request_id": request_id,
                "trace_id": trace_id,
                "query": body.query,
                "execution_status": "error",
                "terminal_state": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "metadata_json": {"source": "pipeline.query"},
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "detail": str(exc),
                "request_id": request_id,
                "trace_id": trace_id,
            },
        )


# ── SSE Streaming endpoint ────────────────────────────────────────────────────

@router.post(
    "/stream",
    summary="Stream the Text2SQL pipeline results via SSE",
    description=(
        "Same pipeline as `/query`, but streams each stage result to the client "
        "immediately as it completes using **Server-Sent Events (SSE)**.\n\n"
        "### Event sequence\n"
        "| `event:` | Fires when | Key fields |\n"
        "|---|---|---|\n"
        "| `pipeline.start` | Request accepted | `request_id`, `trace_id` |\n"
        "| `guard` | Guard check done | `allowed`, `reason`, `latency_ms` |\n"
        "| `classification` | Classifier done | `label`, `reason`, `latency_ms` |\n"
        "| `pinecone` | Vector retrieval done | `columns`, `tables`, `latency_ms` |\n"
        "| `neo4j` | Join expansion done | `schema_tables`, `latency_ms` |\n"
        "| `schema` | Schema SQL ready | `schema_sql`, `latency_ms` |\n"
        "| `runpod` | SQL generated | `generated_sql`, `latency_ms` |\n"
        "| `execution.remark` | Execution policy applied | `remark`, `execution_sql`, `blocked_by_firewall` |\n"
        "| `execution.data` | Fetched rows returned | `execution_sql`, `execution_data` |\n"
        "| `pipeline.complete` | All done (or early exit) | `metrics`, `skipped` |\n"
        "| `pipeline.error` | Unrecoverable error | `error`, `detail` |\n\n"
        "Use `curl -N` to see events as they stream:\n"
        "```bash\n"
        "curl -N -X POST http://localhost:8000/v1/pipeline/stream \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '{\"query\": \"Which drivers won the most races?\"}'\n"
        "```"
    ),
    responses={
        200: {"description": "SSE stream — text/event-stream"},
        422: {"model": ErrorResponse, "description": "Request validation error"},
    },
)
async def stream_pipeline(
    body: PipelineRequest,
    request: Request,
    settings: BackendSettings = Depends(get_settings),
    guard_svc: GuardService = Depends(get_guard_service),
    classifier_svc: ClassifierService = Depends(get_classifier_service),
    pinecone_svc: PineconeService = Depends(get_pinecone_service),
    neo4j_svc: Neo4jService = Depends(get_neo4j_service),
    metadata_svc: MetadataService = Depends(get_metadata_service),
    sql_execution_svc: SqlExecutionService = Depends(get_sql_execution_service),
    runpod_svc: RunpodService = Depends(get_runpod_service),
    observability_svc: ObservabilityService = Depends(get_observability_service),
) -> EventSourceResponse:
    request_id: str = getattr(request.state, "request_id", None) or get_request_id()
    trace_id: str = getattr(request.state, "trace_id", None) or get_trace_id()

    if body.trace_id:
        trace_id = body.trace_id

    logger.info(
        "sse.pipeline.request",
        extra={
            "query_preview": body.query[:80],
            "query_length": len(body.query),
        },
    )

    executor = SSEPipelineExecutor(
        settings=settings,
        guard_service=guard_svc,
        classifier_service=classifier_svc,
        pinecone_service=pinecone_svc,
        neo4j_service=neo4j_svc,
        metadata_service=metadata_svc,
        sql_execution_service=sql_execution_svc,
        runpod_service=runpod_svc,
        observability_service=observability_svc,
    )

    async def event_generator():
        try:
            async for event in executor.execute_streaming(
                query=body.query,
                request_id=request_id,
                trace_id=trace_id,
            ):
                # Check if client disconnected between events
                if await request.is_disconnected():
                    logger.info("sse.client_disconnected", extra={"request_id": request_id})
                    break
                yield event
        except asyncio.CancelledError:
            logger.info("sse.generator_cancelled", extra={"request_id": request_id})
        except Exception as exc:
            logger.exception("sse.generator_error", extra={"error": str(exc)})

    return EventSourceResponse(
        event_generator(),
        headers={
            "X-Request-ID": request_id,
            "X-Trace-ID": trace_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering for SSE
        },
    )


@router.post(
    "/feedback",
    status_code=status.HTTP_200_OK,
    summary="Store thumbs-down feedback for a pipeline response",
)
async def submit_feedback(
    body: FeedbackRequest,
    observability_svc: ObservabilityService = Depends(get_observability_service),
) -> FeedbackResponse:
    await asyncio.to_thread(
        observability_svc.record_user_feedback,
        request_id=body.request_id,
        trace_id=body.trace_id,
        query=body.query,
        response=body.response,
        feedback_type=body.feedback_type,
        comment=body.comment,
        rating=body.rating,
        user_id=body.user_id,
        session_id=body.session_id,
    )
    await asyncio.to_thread(
        observability_svc.record_logical_failure,
        request_id=body.request_id,
        trace_id=body.trace_id,
        query=body.query,
        model_output=body.response,
        expected_output=None,
        is_correct=False,
        failure_reason=body.comment or "thumbs_down",
        review_status="new",
        correction_json=None,
        notes=body.comment,
    )
    return FeedbackResponse(status="ok", detail="feedback recorded")
