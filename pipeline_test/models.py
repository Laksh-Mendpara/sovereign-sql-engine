"""Dataclasses shared across the pipeline checker."""

from dataclasses import asdict, dataclass


@dataclass
class GuardResult:
    allowed: bool
    reason: str
    raw_response: str


@dataclass
class ClassificationResult:
    label: str
    reason: str
    raw_response: str


@dataclass
class RetrievedColumn:
    table_name: str
    column_name: str
    text: str
    vector_score: float
    rerank_score: float


@dataclass
class RetrievedTable:
    table_name: str
    text: str
    vector_score: float


@dataclass
class PipelineResult:
    query: str
    guard: GuardResult
    classification: ClassificationResult
    retrieved_columns: list[RetrievedColumn]
    retrieved_tables: list[RetrievedTable]
    selected_tables: list[str]
    schema_tables: list[str]
    schema_sql: str
    runpod_response: dict
    generated_sql: str | None = None
    execution_sql: str | None = None
    execution_data: list[dict] | None = None

    def as_dict(self) -> dict:
        return asdict(self)
