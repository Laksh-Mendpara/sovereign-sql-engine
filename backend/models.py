"""Pydantic wire-level schemas for the API.

Separate from the internal dataclasses in ``pipeline_test/models.py``
which remain unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request ────────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    """Body for POST /v1/pipeline/query."""

    query: str = Field(..., min_length=1, max_length=4096, description="Natural-language query")
    trace_id: Optional[str] = Field(
        default=None,
        description="Optional caller-supplied trace ID (propagated to logs + response)",
    )

    @field_validator("query")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be blank after stripping whitespace")
        return stripped


class FeedbackRequest(BaseModel):
    request_id: str
    trace_id: Optional[str] = None
    query: str
    response: str
    feedback_type: str = Field(default="thumbs_down")
    comment: Optional[str] = None
    rating: Optional[int] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


# ── Inner result schemas ───────────────────────────────────────────────────────

class GuardResultSchema(BaseModel):
    allowed: bool
    reason: str
    raw_response: str


class ClassificationResultSchema(BaseModel):
    label: str
    reason: str
    raw_response: str


class RetrievedColumnSchema(BaseModel):
    table_name: str
    column_name: str
    text: str
    vector_score: float
    rerank_score: float


class RetrievedTableSchema(BaseModel):
    table_name: str
    text: str
    vector_score: float


class StageMetrics(BaseModel):
    """Per-stage wall-clock timings in milliseconds."""

    guard_ms: Optional[float] = None
    classifier_ms: Optional[float] = None
    pinecone_ms: Optional[float] = None
    neo4j_ms: Optional[float] = None
    metadata_ms: Optional[float] = None
    runpod_ms: Optional[float] = None
    total_ms: float


# ── Main pipeline response ─────────────────────────────────────────────────────

class PipelineResponse(BaseModel):
    """Full result returned by POST /v1/pipeline/query."""

    # Correlation
    request_id: str
    trace_id: str

    # Pipeline output
    query: str
    guard: GuardResultSchema
    classification: ClassificationResultSchema
    retrieved_columns: list[RetrievedColumnSchema]
    retrieved_tables: list[RetrievedTableSchema]
    selected_tables: list[str]
    schema_tables: list[str]
    schema_sql: str
    generated_sql: Optional[str] = None
    execution_sql: Optional[str] = None
    execution_data: Optional[list[dict[str, Any]]] = None
    runpod_response: dict[str, Any]

    # Observability
    metrics: StageMetrics

    # Early-exit metadata
    skipped: bool = False
    skip_reason: Optional[str] = None


# ── Health / Readiness ─────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    service: str
    status: str                   # "ok" | "degraded" | "down"
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                   # "ok"
    version: str
    uptime_seconds: float
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str                   # "ready" | "degraded" | "not_ready"
    services: list[ServiceStatus]


# ── Error ──────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    detail: Optional[str] = None


# ── SSE Event payloads ─────────────────────────────────────────────────────────

class SSEStartPayload(BaseModel):
    """Emitted immediately when the request is accepted."""
    request_id: str
    trace_id: str
    query: str


class SSEGuardPayload(BaseModel):
    """Emitted when the guard check finishes."""
    allowed: bool
    reason: str
    latency_ms: float


class SSEClassificationPayload(BaseModel):
    """Emitted when the classifier finishes."""
    label: str          # "easy" | "difficult" | "out_of_topic"
    reason: str
    latency_ms: float


class SSEPineconePayload(BaseModel):
    """Emitted when Pinecone retrieval finishes."""
    columns: list[RetrievedColumnSchema]
    tables: list[RetrievedTableSchema]
    latency_ms: float


class SSENeo4jPayload(BaseModel):
    """Emitted when Neo4j join expansion finishes."""
    schema_tables: list[str]
    latency_ms: float


class SSESchemaPayload(BaseModel):
    """Emitted when metadata schema SQL is generated."""
    schema_sql: str
    latency_ms: float


class SSERunpodPayload(BaseModel):
    """Emitted when RunPod SQL generation finishes."""
    generated_sql: Optional[str] = None
    runpod_response: dict[str, Any]
    latency_ms: float


class SSEExecutionRemarkPayload(BaseModel):
    """Emitted after RunPod, before execution data."""
    remark: str
    execution_sql: Optional[str] = None
    blocked_by_firewall: bool = False


class SSEExecutionDataPayload(BaseModel):
    """Emitted after the execution remark with fetched rows, if any."""
    execution_sql: Optional[str] = None
    execution_data: Optional[list[dict[str, Any]]] = None


class SSECompletePayload(BaseModel):
    """Final event — always emitted, even on early-exit (guard/out_of_topic)."""
    skipped: bool = False
    skip_reason: Optional[str] = None
    metrics: StageMetrics


class SSEErrorPayload(BaseModel):
    """Emitted on timeout or unrecoverable error."""
    error: str
    detail: str
