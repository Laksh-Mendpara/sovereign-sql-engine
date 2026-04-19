"""Async pipeline orchestrator with maximum service-level parallelism.

Execution DAG (Ultra-Concurrent)
────────────────────────────────

  asyncio.gather (concurrent fan-out)
  ┌──────────────────┬──────────────────┬──────────────────────────────────┐
  │ Task A: Guard    │ Task B: Classify │ Task C: Retrieval Chain          │
  │ (Modal LLM)      │ (Modal LLM)      │ 1. Pinecone (Col + Tbl)          │
  │                  │                  │ 2. await Task B (difficulty)     │
  │                  │                  │ 3. Neo4j (join expansion)        │
  │                  │                  │ 4. Metadata (schema SQL)         │
  └──────────────────┴────────┬─────────┴──────────────────────────────────┘
                              │
                    Combine All & Check Gates
                    (guard.allowed? classification.out_of_topic?)
                              │
                       ┌──────▼──────────┐
                       │  runpod arctic  │
                       └──────┬──────────┘
                              │
                       PipelineResponse

All synchronous service calls are offloaded via ``asyncio.to_thread()``.
Each stage is wrapped with ``timed_span`` (sets span_id in log context)
and ``asyncio.wait_for`` (per-stage timeout).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import BackendSettings
    from .services.classifier_service import ClassifierService
    from .services.guard_service import GuardService
    from .services.metadata_service import MetadataService
    from .services.neo4j_service import Neo4jService
    from .services.pinecone_service import PineconeService
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
    PipelineResponse,
    RetrievedColumnSchema,
    RetrievedTableSchema,
    StageMetrics,
)

logger = get_logger(__name__)


class AsyncPipelineExecutor:
    """Stateless async orchestrator — service objects are injected per request."""

    def __init__(
        self,
        settings: BackendSettings,
        guard_service: GuardService,
        classifier_service: ClassifierService,
        pinecone_service: PineconeService,
        neo4j_service: Neo4jService,
        metadata_service: MetadataService,
        runpod_service: RunpodService,
    ) -> None:
        self.cfg = settings
        self.guard = guard_service
        self.classifier = classifier_service
        self.pinecone = pinecone_service
        self.neo4j = neo4j_service
        self.metadata = metadata_service
        self.runpod = runpod_service
        self.metrics: MetricsCollector = get_metrics()

    # ── Public entry point ─────────────────────────────────────────────────────

    async def execute(self, query: str, request_id: str, trace_id: str) -> PipelineResponse:
        timer = StageTimer()
        self.metrics.requests_total.inc()

        logger.info(
            "pipeline.start",
            extra={"query_length": len(query), "query_preview": query[:80]},
        )

        try:
            response = await asyncio.wait_for(
                self._run(query, request_id, trace_id, timer),
                timeout=self.cfg.total_pipeline_timeout,
            )
        except asyncio.TimeoutError:
            self.metrics.requests_failed.inc()
            logger.error(
                "pipeline.timeout",
                extra={"timeout_s": self.cfg.total_pipeline_timeout},
            )
            # Re-raise so global handler can return 504
            raise
        except Exception:
            self.metrics.requests_failed.inc()
            logger.exception("pipeline.error")
            raise

        self.metrics.total_latency.observe(timer.total_ms)
        logger.info(
            "pipeline.done",
            extra={
                "total_ms": timer.total_ms,
                "skipped": response.skipped,
                "classification": response.classification.label,
            },
        )
        return response

    # ── Orchestration ──────────────────────────────────────────────────────────

    async def _run(
        self,
        query: str,
        request_id: str,
        trace_id: str,
        timer: StageTimer,
    ) -> PipelineResponse:

        # ── Stage 1: Ultra-Parallel Fan-out ───────────────────────────────────
        # We start Guard, Classifier, and the Retrieval Flow (Pinecone+Neo4j+Metadata)
        # all at the same time.

        logger.info("pipeline.stage", extra={"stage": "fan_out"})

        # Task A: Guard
        guard_task = asyncio.create_task(self._guard(query, timer))

        # Task B: Classification
        classify_task = asyncio.create_task(self._classify(query, timer))

        # Task C: Retrieval Chain (Pinecone -> await B -> Neo4j -> Metadata)
        retrieval_task = asyncio.create_task(self._retrieval_flow(query, classify_task, timer))

        # Wait for all three branches to finish
        guard, classification, retrieval_data = await asyncio.gather(
            guard_task, classify_task, retrieval_task
        )

        cols, tbls, schema_tables, schema_sql = retrieval_data or ([], [], [], "")

        # ── Stage 2: Guard gate ───────────────────────────────────────────────
        if not guard.allowed:
            self.metrics.requests_guard_blocked.inc()
            logger.warning("pipeline.guard_blocked", extra={"reason": guard.reason})
            return self._make_response(
                request_id=request_id, trace_id=trace_id, query=query,
                guard=guard, classification=classification,
                cols=cols, tbls=tbls, selected=[], schema_tables=[],
                schema_sql="",
                runpod_response={"skipped": True, "reason": f"guard_failed: {guard.reason}"},
                timer=timer, skipped=True,
                skip_reason=f"guard_failed: {guard.reason}",
            )

        # ── Stage 3: Classification gate ──────────────────────────────────────
        if classification.label == "out_of_topic":
            self.metrics.requests_out_of_topic.inc()
            self.metrics.inc_label("out_of_topic")
            logger.info("pipeline.out_of_topic", extra={"reason": classification.reason})
            return self._make_response(
                request_id=request_id, trace_id=trace_id, query=query,
                guard=guard, classification=classification,
                cols=cols, tbls=tbls, selected=[], schema_tables=[],
                schema_sql="",
                runpod_response={"skipped": True, "reason": classification.reason},
                timer=timer, skipped=True, skip_reason=classification.reason,
            )

        self.metrics.inc_label(classification.label)

        # ── Stage 4: RunPod Arctic SQL generation ─────────────────────────────
        # Note: selected tables are derived from cols/tbls
        seed: set[str] = {c.table_name for c in cols}
        seed.update(t.table_name for t in tbls)
        selected = sorted(seed)

        if not schema_sql:
            logger.warning("pipeline.no_schema_generated", extra={"seed_count": len(selected)})
            return self._make_response(
                request_id=request_id, trace_id=trace_id, query=query,
                guard=guard, classification=classification,
                cols=cols, tbls=tbls, selected=selected,
                schema_tables=[], schema_sql="",
                runpod_response={"skipped": True, "reason": "no_schema_generated"},
                timer=timer, skipped=True, skip_reason="no_schema_generated",
            )

        runpod_response = await self._runpod(query, schema_sql, timer)
        generated_sql = self._extract_sql(runpod_response)

        return self._make_response(
            request_id=request_id, trace_id=trace_id, query=query,
            guard=guard, classification=classification,
            cols=cols, tbls=tbls, selected=selected,
            schema_tables=schema_tables, schema_sql=schema_sql,
            runpod_response=runpod_response,
            timer=timer, generated_sql=generated_sql,
        )

    # ── Task C: Retrieval Flow (Pipeline) ─────────────────────────────────────

    async def _retrieval_flow(
        self,
        query: str,
        classify_task: asyncio.Task[ClassificationResult],
        timer: StageTimer,
    ) -> tuple[list[RetrievedColumn], list[RetrievedTable], list[str], str]:
        """
        Executes Pinecone -> Neo4j -> Metadata in a single async chain.
        Wait for classify_task result before starting Neo4j.
        """
        # 1. Pinecone (Runs in parallel with Task A and Task B)
        cols, tbls = await self._pinecone_group(query, timer)

        # 2. Wait for Classification (Task B) which is already running
        # This is the "join point" where Task C waits for Task B.
        classification = await classify_task

        # If out of topic, don't waste time on Neo4j or Metadata
        if classification.label == "out_of_topic":
            return cols, tbls, [], ""

        # 3. Seed tables for Neo4j
        seed: set[str] = {c.table_name for c in cols}
        seed.update(t.table_name for t in tbls)
        selected = sorted(seed)

        if not selected:
            return cols, tbls, [], ""

        # 4. Neo4j join expansion
        schema_tables = await self._neo4j_expand(selected, classification.label, timer)

        # 5. Metadata schema SQL
        schema_sql = await self._get_schema(schema_tables, timer)

        return cols, tbls, schema_tables, schema_sql

    # ── Service Primitives ────────────────────────────────────────────────────

    async def _guard(self, query: str, timer: StageTimer) -> GuardResult:
        with timed_span("guard_ms", timer, self.metrics.guard_latency):
            logger.info("guard.start")
            result: GuardResult = await asyncio.wait_for(
                asyncio.to_thread(self.guard.check, query),
                timeout=self.cfg.guard_timeout,
            )
            logger.info("guard.done", extra={"allowed": result.allowed, "reason": result.reason or ""})
            return result

    async def _classify(self, query: str, timer: StageTimer) -> ClassificationResult:
        with timed_span("classifier_ms", timer, self.metrics.classifier_latency):
            logger.info("classifier.start")
            result: ClassificationResult = await asyncio.wait_for(
                asyncio.to_thread(self.classifier.classify, query),
                timeout=self.cfg.classifier_timeout,
            )
            logger.info("classifier.done", extra={"label": result.label})
            return result

    async def _pinecone_group(
        self, query: str, timer: StageTimer
    ) -> tuple[list[RetrievedColumn], list[RetrievedTable]]:
        with timed_span("pinecone_ms", timer, self.metrics.pinecone_latency):
            logger.info("pinecone.start")
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
            logger.info("pinecone.done", extra={"columns": len(cols), "tables": len(tbls)})
            return cols, tbls

    async def _neo4j_expand(
        self, selected: list[str], difficulty: str, timer: StageTimer
    ) -> list[str]:
        with timed_span("neo4j_ms", timer, self.metrics.neo4j_latency):
            logger.info("neo4j.start", extra={"seed_count": len(selected)})
            result: list[str] = await asyncio.wait_for(
                asyncio.to_thread(self.neo4j.expand_tables, selected, difficulty),
                timeout=self.cfg.neo4j_timeout,
            )
            logger.info("neo4j.done", extra={"schema_table_count": len(result)})
            return result

    async def _get_schema(self, schema_tables: list[str], timer: StageTimer) -> str:
        with timed_span("metadata_ms", timer, self.metrics.metadata_latency):
            logger.info("metadata.start", extra={"table_count": len(schema_tables)})
            schema_sql: str = await asyncio.wait_for(
                asyncio.to_thread(self.metadata.get_schema_sql, schema_tables),
                timeout=self.cfg.metadata_timeout,
            )
            logger.info("metadata.done", extra={"schema_chars": len(schema_sql)})
            return schema_sql

    async def _runpod(self, query: str, schema_sql: str, timer: StageTimer) -> dict:
        with timed_span("runpod_ms", timer, self.metrics.runpod_latency):
            payload = build_arctic_runpod_input(query, schema_sql)
            logger.info("runpod.start")
            response: dict = await asyncio.wait_for(
                asyncio.to_thread(self.runpod.run_request, payload),
                timeout=self.cfg.runpod_timeout,
            )
            logger.info("runpod.done", extra={"status": response.get("status", "unknown")})
            return response

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

    @staticmethod
    def _make_response(
        *,
        request_id: str,
        trace_id: str,
        query: str,
        guard: GuardResult,
        classification: ClassificationResult,
        cols: list[RetrievedColumn],
        tbls: list[RetrievedTable],
        selected: list[str],
        schema_tables: list[str],
        schema_sql: str,
        runpod_response: dict,
        timer: StageTimer,
        skipped: bool = False,
        skip_reason: str | None = None,
        generated_sql: str | None = None,
    ) -> PipelineResponse:
        t = timer.as_dict()
        return PipelineResponse(
            request_id=request_id,
            trace_id=trace_id,
            query=query,
            guard=GuardResultSchema(**asdict(guard)),
            classification=ClassificationResultSchema(**asdict(classification)),
            retrieved_columns=[RetrievedColumnSchema(**asdict(c)) for c in cols],
            retrieved_tables=[RetrievedTableSchema(**asdict(t_)) for t_ in tbls],
            selected_tables=selected,
            schema_tables=schema_tables,
            schema_sql=schema_sql,
            generated_sql=generated_sql,
            runpod_response=runpod_response,
            metrics=StageMetrics(
                guard_ms=t.get("guard_ms"),
                classifier_ms=t.get("classifier_ms"),
                pinecone_ms=t.get("pinecone_ms"),
                neo4j_ms=t.get("neo4j_ms"),
                metadata_ms=t.get("metadata_ms"),
                runpod_ms=t.get("runpod_ms"),
                total_ms=t["total_ms"],
            ),
            skipped=skipped,
            skip_reason=skip_reason,
        )
