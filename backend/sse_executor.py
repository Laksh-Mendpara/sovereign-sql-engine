"""Streaming SSE pipeline executor.

Publishes each completed stage result to an ``asyncio.Queue`` immediately,
which is drained by the route handler and forwarded to the SSE stream.

Event sequence
──────────────
  pipeline.start        → emitted immediately on entry
  guard                 → when guard check completes (Task A)
  classification        → when classifier completes (Task B)
  pinecone              → when Pinecone retrieval completes (Task C, step 1)
  neo4j                 → when Neo4j expansion completes (Task C, step 2)
  schema                → when metadata schema SQL is ready (Task C, step 3)
  runpod                → when Arctic SQL generation completes (final step)
  execution.remark      → firewall / limit remark for execution SQL
  execution.data        → fetched rows from the SQLite data DB
  pipeline.complete     → always the last event (even on early-exit)

  On any unrecoverable error:
  pipeline.error        → followed by stream close

The existing ``AsyncPipelineExecutor`` (synchronous JSON response) is not
modified; this class is purely additive.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, AsyncGenerator

if TYPE_CHECKING:
    from .config import BackendSettings
    from .services.classifier_service import ClassifierService
    from .services.guard_service import GuardService
    from .services.metadata_service import MetadataService
    from .services.observability_service import ObservabilityService
    from .services.neo4j_service import Neo4jService
    from .services.pinecone_service import PineconeService
    from .services.sql_execution_service import SqlExecutionService
    from .services.runpod_service import RunpodService

from .logic_models import (
    ClassificationResult,
    GuardResult,
    RetrievedColumn,
    RetrievedTable,
)
from .prompts import build_arctic_runpod_input
from .logging_config import get_logger
from .metrics import (
    MetricsCollector,
    StageTimer,
    get_metrics,
    timed_span,
)
from .models import (
    ClassificationResultSchema,
    GuardResultSchema,
    SSEExecutionDataPayload,
    SSEExecutionRemarkPayload,
    RetrievedColumnSchema,
    RetrievedTableSchema,
    SSEClassificationPayload,
    SSECompletePayload,
    SSEErrorPayload,
    SSEGuardPayload,
    SSENeo4jPayload,
    SSEPineconePayload,
    SSERunpodPayload,
    SSESchemaPayload,
    SSEStartPayload,
    StageMetrics,
)
from .sql_utils import plan_sql_execution

logger = get_logger(__name__)

# Sentinel that signals the event queue has been fully drained
_STREAM_DONE = object()


class SSEPipelineExecutor:
    """Streaming variant of the pipeline executor.

    Each stage pushes a ``(event_name, payload_model)`` tuple into a shared
    ``asyncio.Queue``.  The route handler draws from that queue and forwards
    events to the client as fast as they arrive.
    """

    def __init__(
        self,
        settings: BackendSettings,
        guard_service: GuardService,
        classifier_service: ClassifierService,
        pinecone_service: PineconeService,
        neo4j_service: Neo4jService,
        metadata_service: MetadataService,
        sql_execution_service: SqlExecutionService,
        runpod_service: RunpodService,
        observability_service: ObservabilityService,
    ) -> None:
        self.cfg = settings
        self.guard = guard_service
        self.classifier = classifier_service
        self.pinecone = pinecone_service
        self.neo4j = neo4j_service
        self.metadata = metadata_service
        self.sql_executor = sql_execution_service
        self.runpod = runpod_service
        self.observability = observability_service
        self.metrics: MetricsCollector = get_metrics()
        self._audit_state: dict[str, Any] = {}

    # ── Public streaming entry point ───────────────────────────────────────────

    async def execute_streaming(
        self,
        query: str,
        request_id: str,
        trace_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Async generator that yields SSE-compatible dicts:
            {"event": "<name>", "data": "<json string>", "id": "<seq>"}

        The generator completes (returns) after the ``pipeline.complete`` or
        ``pipeline.error`` event is yielded.
        """
        queue: asyncio.Queue[tuple[str, Any] | object] = asyncio.Queue()
        timer = StageTimer()
        seq = 0
        self.metrics.requests_total.inc()
        self._audit_state = {
            "request_id": request_id,
            "trace_id": trace_id,
            "query": query,
            "execution_status": "running",
            "terminal_state": "running",
        }

        def _emit(event_name: str, payload: Any) -> None:
            """Thread-safe: can be called from to_thread callbacks or coroutines."""
            queue.put_nowait((event_name, payload))

        # ── Immediately emit pipeline.start ────────────────────────────────────
        start_payload = SSEStartPayload(
            request_id=request_id,
            trace_id=trace_id,
            query=query[:256],
        )
        seq += 1
        yield {"event": "pipeline.start", "data": start_payload.model_dump_json(), "id": str(seq)}

        # ── Launch all three parallel branches ─────────────────────────────────
        classify_task: asyncio.Task[ClassificationResult] = asyncio.create_task(
            self._classify_and_emit(query, timer, _emit)
        )
        guard_task: asyncio.Task[GuardResult] = asyncio.create_task(
            self._guard_and_emit(query, timer, _emit)
        )
        retrieval_task: asyncio.Task[None] = asyncio.create_task(
            self._retrieval_and_emit(query, classify_task, guard_task, timer, _emit)
        )

        # ── Wait for all tasks, then push sentinel ─────────────────────────────
        async def _wait_all() -> None:
            """Wait for all work to finish, then signal the queue."""
            try:
                await asyncio.gather(guard_task, classify_task, retrieval_task)
            except Exception:
                pass  # Errors have already been emitted as pipeline.error
            finally:
                queue.put_nowait(_STREAM_DONE)

        watcher = asyncio.create_task(_wait_all())

        # ── Drain queue and yield events until sentinel ─────────────────────────
        try:
            while True:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=self.cfg.total_pipeline_timeout + 5,
                )
                if item is _STREAM_DONE:
                    break

                event_name, payload = item  # type: ignore[misc]
                seq += 1
                data = payload.model_dump_json() if hasattr(payload, "model_dump_json") else str(payload)
                yield {"event": event_name, "data": data, "id": str(seq)}

        except asyncio.TimeoutError:
            # Hard timeout — cancel all outstanding work
            for t in (guard_task, classify_task, retrieval_task, watcher):
                t.cancel()
            self._audit_state.update(
                {
                    "execution_status": "timeout",
                    "terminal_state": "timeout",
                    "error_type": "timeout",
                    "error_message": f"Pipeline did not complete within {self.cfg.total_pipeline_timeout}s",
                }
            )
            await asyncio.to_thread(self.observability.upsert_request_record, dict(self._audit_state))
            error_payload = SSEErrorPayload(
                error="pipeline_timeout",
                detail=f"Pipeline did not complete within {self.cfg.total_pipeline_timeout}s",
            )
            seq += 1
            yield {"event": "pipeline.error", "data": error_payload.model_dump_json(), "id": str(seq)}
            self.metrics.requests_failed.inc()
            return

        except asyncio.CancelledError:
            # Client disconnected — cancel remaining work
            for t in (guard_task, classify_task, retrieval_task, watcher):
                t.cancel()
            await asyncio.to_thread(self.observability.upsert_request_record, dict(self._audit_state))
            logger.info("sse.client_disconnected", extra={"request_id": request_id})
            return

        # ── Emit final pipeline.complete ────────────────────────────────────────
        t = timer.as_dict()
        complete_payload = SSECompletePayload(
            skipped=getattr(watcher, "_sse_skipped", False),
            skip_reason=getattr(watcher, "_sse_skip_reason", None),
            metrics=StageMetrics(
                guard_ms=t.get("guard_ms"),
                classifier_ms=t.get("classifier_ms"),
                pinecone_ms=t.get("pinecone_ms"),
                neo4j_ms=t.get("neo4j_ms"),
                metadata_ms=t.get("metadata_ms"),
                runpod_ms=t.get("runpod_ms"),
                total_ms=t["total_ms"],
            ),
        )
        seq += 1
        yield {"event": "pipeline.complete", "data": complete_payload.model_dump_json(), "id": str(seq)}

        await asyncio.to_thread(self.observability.upsert_request_record, dict(self._audit_state))
        self.metrics.total_latency.observe(timer.total_ms)
        logger.info(
            "sse.pipeline.done",
            extra={"total_ms": timer.total_ms, "request_id": request_id},
        )

    # ── Task A: Guard ─────────────────────────────────────────────────────────

    async def _guard_and_emit(
        self,
        query: str,
        timer: StageTimer,
        emit: Any,
    ) -> GuardResult:
        try:
            with timed_span("guard_ms", timer, self.metrics.guard_latency) as _span:
                logger.info("sse.guard.start")
                result: GuardResult = await asyncio.wait_for(
                    asyncio.to_thread(self.guard.check, query),
                    timeout=self.cfg.guard_timeout,
                )
                logger.info("sse.guard.done", extra={"allowed": result.allowed})

            elapsed = timer.as_dict().get("guard_ms", 0.0)
            emit("guard", SSEGuardPayload(
                allowed=result.allowed,
                reason=result.reason,
                latency_ms=elapsed,
            ))
            self._audit_state.update(
                {
                    "guard_allowed": result.allowed,
                    "guard_reason": result.reason,
                }
            )
            return result
        except Exception as exc:
            logger.exception("sse.guard.error")
            self._audit_state.update(
                {
                    "execution_status": "error",
                    "terminal_state": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            emit("pipeline.error", SSEErrorPayload(error="guard_error", detail=str(exc)))
            raise

    # ── Task B: Classifier ────────────────────────────────────────────────────

    async def _classify_and_emit(
        self,
        query: str,
        timer: StageTimer,
        emit: Any,
    ) -> ClassificationResult:
        try:
            with timed_span("classifier_ms", timer, self.metrics.classifier_latency):
                logger.info("sse.classifier.start")
                result: ClassificationResult = await asyncio.wait_for(
                    asyncio.to_thread(self.classifier.classify, query),
                    timeout=self.cfg.classifier_timeout,
                )
                logger.info("sse.classifier.done", extra={"label": result.label})

            elapsed = timer.as_dict().get("classifier_ms", 0.0)
            emit("classification", SSEClassificationPayload(
                label=result.label,
                reason=result.reason,
                latency_ms=elapsed,
            ))
            self._audit_state.update(
                {
                    "classification_label": result.label,
                    "classification_reason": result.reason,
                }
            )
            return result
        except Exception as exc:
            logger.exception("sse.classifier.error")
            self._audit_state.update(
                {
                    "execution_status": "error",
                    "terminal_state": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            emit("pipeline.error", SSEErrorPayload(error="classifier_error", detail=str(exc)))
            raise

    # ── Task C: Retrieval Chain (Pinecone → Neo4j → Schema → RunPod) ─────────

    async def _retrieval_and_emit(
        self,
        query: str,
        classify_task: asyncio.Task[ClassificationResult],
        guard_task: asyncio.Task[GuardResult],
        timer: StageTimer,
        emit: Any,
    ) -> None:
        try:
            # 1. Pinecone — runs in parallel with guard + classify
            cols, tbls = await self._pinecone_and_emit(query, timer, emit)

            # 2. Await classification (likely already done by the time Pinecone finishes)
            classification = await classify_task

            # 3. Await guard (check before expensive downstream work)
            guard = await guard_task

            # Early-exit: guard blocked
            if not guard.allowed:
                self.metrics.requests_guard_blocked.inc()
                logger.warning("sse.guard_blocked", extra={"reason": guard.reason})
                self._audit_state.update(
                    {
                        "execution_status": "blocked_guardrail",
                        "terminal_state": "guardrail_blocked",
                    }
                )
                return  # pipeline.complete will be assembled by the watcher

            # Early-exit: out of topic
            if classification.label == "out_of_topic":
                self.metrics.requests_out_of_topic.inc()
                self.metrics.inc_label("out_of_topic")
                logger.info("sse.out_of_topic")
                self._audit_state.update(
                    {
                        "execution_status": "blocked_out_of_topic",
                        "terminal_state": "out_of_topic",
                    }
                )
                return

            self.metrics.inc_label(classification.label)

            # 4. Neo4j expansion
            seed: set[str] = {c.table_name for c in cols}
            seed.update(t.table_name for t in tbls)
            selected = sorted(seed)

            if not selected:
                self._audit_state.update(
                    {
                        "execution_status": "skipped",
                        "terminal_state": "skipped",
                    }
                )
                return

            schema_tables = await self._neo4j_and_emit(selected, classification.label, timer, emit)

            # 5. Schema SQL
            schema_sql = await self._schema_and_emit(schema_tables, timer, emit)

            if not schema_sql:
                self._audit_state.update(
                    {
                        "execution_status": "skipped",
                        "terminal_state": "skipped",
                    }
                )
                return

            # 6. RunPod — terminal stage
            await self._runpod_and_emit(query, schema_sql, timer, emit)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("sse.retrieval_chain.error", extra={"error": str(exc)})
            emit("pipeline.error", SSEErrorPayload(error="retrieval_error", detail=str(exc)))
            self._audit_state.update(
                {
                    "execution_status": "error",
                    "terminal_state": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            return

    # ── Pinecone sub-step ─────────────────────────────────────────────────────

    async def _pinecone_and_emit(
        self,
        query: str,
        timer: StageTimer,
        emit: Any,
    ) -> tuple[list[RetrievedColumn], list[RetrievedTable]]:
        try:
            with timed_span("pinecone_ms", timer, self.metrics.pinecone_latency):
                logger.info("sse.pinecone.start")
                col_task = asyncio.create_task(
                    asyncio.wait_for(
                        asyncio.to_thread(
                            self.pinecone.fetch_top_columns,
                            query, self.metadata,
                            self.cfg.top_k_columns,
                            self.cfg.initial_retrieval_multiplier,
                        ),
                        timeout=self.cfg.pinecone_timeout,
                    )
                )
                tbl_task = asyncio.create_task(
                    asyncio.wait_for(
                        asyncio.to_thread(
                            self.pinecone.fetch_top_tables,
                            query, self.metadata,
                            self.cfg.top_k_tables,
                        ),
                        timeout=self.cfg.pinecone_timeout,
                    )
                )
                cols, tbls = await asyncio.gather(col_task, tbl_task)
                logger.info("sse.pinecone.done", extra={"cols": len(cols), "tbls": len(tbls)})
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            logger.warning("sse.pinecone.timeout", extra={"error": str(exc)})
            cols, tbls = [], []
        except Exception as exc:
            logger.warning("sse.pinecone.error", extra={"error": str(exc)})
            cols, tbls = [], []

        elapsed = timer.as_dict().get("pinecone_ms", 0.0)
        emit("pinecone", SSEPineconePayload(
            columns=[RetrievedColumnSchema(**asdict(c)) for c in cols],
            tables=[RetrievedTableSchema(**asdict(t)) for t in tbls],
            latency_ms=elapsed,
        ))
        self._audit_state["metadata_json"] = {
            "retrieved_columns": [asdict(c) for c in cols],
            "retrieved_tables": [asdict(t) for t in tbls],
        }
        return cols, tbls

    # ── Neo4j sub-step ────────────────────────────────────────────────────────

    async def _neo4j_and_emit(
        self,
        selected: list[str],
        difficulty: str,
        timer: StageTimer,
        emit: Any,
    ) -> list[str]:
        with timed_span("neo4j_ms", timer, self.metrics.neo4j_latency):
            logger.info("sse.neo4j.start", extra={"seed_count": len(selected)})
            result: list[str] = await asyncio.wait_for(
                asyncio.to_thread(self.neo4j.expand_tables, selected, difficulty),
                timeout=self.cfg.neo4j_timeout,
            )
            logger.info("sse.neo4j.done", extra={"schema_table_count": len(result)})

        elapsed = timer.as_dict().get("neo4j_ms", 0.0)
        emit("neo4j", SSENeo4jPayload(
            schema_tables=result,
            latency_ms=elapsed,
        ))
        metadata = self._audit_state.get("metadata_json") or {}
        metadata["schema_tables"] = result
        self._audit_state["metadata_json"] = metadata
        return result

    # ── Schema sub-step ───────────────────────────────────────────────────────

    async def _schema_and_emit(
        self,
        schema_tables: list[str],
        timer: StageTimer,
        emit: Any,
    ) -> str:
        with timed_span("metadata_ms", timer, self.metrics.metadata_latency):
            logger.info("sse.schema.start", extra={"table_count": len(schema_tables)})
            schema_sql: str = await asyncio.wait_for(
                asyncio.to_thread(self.metadata.get_schema_sql, schema_tables),
                timeout=self.cfg.metadata_timeout,
            )
            logger.info("sse.schema.done", extra={"schema_chars": len(schema_sql)})

        elapsed = timer.as_dict().get("metadata_ms", 0.0)
        emit("schema", SSESchemaPayload(
            schema_sql=schema_sql,
            latency_ms=elapsed,
        ))
        self._audit_state["schema_sql"] = schema_sql
        return schema_sql

    # ── RunPod sub-step ───────────────────────────────────────────────────────

    async def _runpod_and_emit(
        self,
        query: str,
        schema_sql: str,
        timer: StageTimer,
        emit: Any,
    ) -> None:
        with timed_span("runpod_ms", timer, self.metrics.runpod_latency):
            payload = build_arctic_runpod_input(query, schema_sql)
            logger.info("sse.runpod.start")
            response: dict = await asyncio.wait_for(
                asyncio.to_thread(self.runpod.run_request, payload),
                timeout=self.cfg.runpod_timeout,
            )
            logger.info("sse.runpod.done", extra={"status": response.get("status", "unknown")})

        elapsed = timer.as_dict().get("runpod_ms", 0.0)
        generated_sql = self._extract_sql(response)
        emit("runpod", SSERunpodPayload(
            generated_sql=generated_sql,
            runpod_response=response,
            latency_ms=elapsed,
        ))
        self._audit_state.update(
            {
                "generated_sql": generated_sql,
                "raw_runpod_response_json": response,
            }
        )
        execution_plan = plan_sql_execution(generated_sql)
        emit("execution.remark", SSEExecutionRemarkPayload(
            remark=execution_plan.remark,
            execution_sql=execution_plan.execution_sql,
            blocked_by_firewall=execution_plan.blocked_by_firewall,
        ))
        self._audit_state.update(
            {
                "execution_sql": execution_plan.execution_sql,
                "execution_status": "blocked_firewall" if execution_plan.blocked_by_firewall else "success",
                "terminal_state": "firewall_blocked" if execution_plan.blocked_by_firewall else "complete",
            }
        )
        execution_data = None
        if execution_plan.execution_sql and not execution_plan.blocked_by_firewall:
            try:
                execution_data = await self._execute_sql(execution_plan.execution_sql)
                if not execution_data:
                    execution_data = None
            except Exception as exc:
                logger.warning("sse.sql_execute.failed", extra={"error": str(exc)})
                self._audit_state.update(
                    {
                        "execution_status": "error",
                        "terminal_state": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
        emit("execution.data", SSEExecutionDataPayload(
            execution_sql=execution_plan.execution_sql,
            execution_data=execution_data,
        ))
        self._audit_state.update(
            {
                "execution_result_json": execution_data,
                "execution_row_count": len(execution_data or []),
            }
        )

    async def _execute_sql(self, sql: str) -> list[dict[str, Any]]:
        logger.info("sse.sql_execute.start")
        result: list[dict[str, Any]] = await asyncio.wait_for(
            asyncio.to_thread(self.sql_executor.execute, sql),
            timeout=self.cfg.sqlite_query_timeout,
        )
        logger.info("sse.sql_execute.done", extra={"row_count": len(result)})
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_sql(runpod_response: dict) -> str | None:
        raw_text = None
        try:
            output = runpod_response.get("output")
            if isinstance(output, list) and len(output) > 0:
                first_item = output[0]
                if isinstance(first_item, dict):
                    choices = first_item.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        tokens = choices[0].get("tokens", [])
                        if tokens and isinstance(tokens, list):
                            raw_text = "".join(tokens)
                        elif choices[0].get("text"):
                            raw_text = choices[0].get("text")
            elif isinstance(output, dict):
                choices = output.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {})
                    if isinstance(msg, dict) and msg.get("content"):
                        raw_text = msg.get("content")
                    elif choices[0].get("tokens") and isinstance(choices[0].get("tokens"), list):
                        raw_text = "".join(choices[0].get("tokens", []))
                    elif choices[0].get("text"):
                        raw_text = choices[0].get("text")
            elif isinstance(output, str):
                raw_text = output
        except Exception:
            pass

        if not raw_text:
            return None

        raw_text = raw_text.strip()
        import re
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return raw_text or None
