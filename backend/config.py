"""Pydantic-based settings for the backend server.

Reads from ``pipeline_test/.env`` (same file used by the CLI pipeline),
with additional backend-specific settings layered on top.
"""

from __future__ import annotations

import os
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve path: backend/ → ../ → pipeline_test/.env
_ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")


class BackendSettings(BaseSettings):
    """All settings for the production FastAPI backend."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    port: int = Field(default=8000, alias="BACKEND_PORT")
    workers: int = Field(default=1, alias="BACKEND_WORKERS")
    debug: bool = Field(default=False, alias="BACKEND_DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── Per-stage timeouts (seconds) ──────────────────────────────────────────
    guard_timeout: float = Field(default=30.0, alias="GUARD_TIMEOUT")
    classifier_timeout: float = Field(default=30.0, alias="CLASSIFIER_TIMEOUT")
    pinecone_timeout: float = Field(default=45.0, alias="PINECONE_TIMEOUT")
    neo4j_timeout: float = Field(default=30.0, alias="NEO4J_TIMEOUT")
    metadata_timeout: float = Field(default=20.0, alias="METADATA_TIMEOUT")
    sqlite_query_timeout: float = Field(default=20.0, alias="SQLITE_QUERY_TIMEOUT")
    runpod_timeout: float = Field(default=360.0, alias="RUNPOD_TIMEOUT")
    total_pipeline_timeout: float = Field(default=420.0, alias="PIPELINE_TIMEOUT")

    # ── Pipeline / Database (mirrors pipeline_test config) ────────────────────
    db_name: str = Field(alias="DB_NAME")
    sqlite_host: str = Field(alias="SQLITE_HOST")
    sqlite_port: str = Field(alias="SQLITE_PORT")
    sqlite_db: str = Field(alias="SQLITE_DB")
    sqlite_metadata_db: str = Field(alias="SQLITE_METADATA_DB")
    sqlite_observability_db: str = Field(alias="SQLITE_LLM_OBSERVABILITY_DB")
    sqlite_api_key: str = Field(alias="SQLITE_API_KEY")

    pinecone_api_key: str = Field(alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="", alias="PINECONE_INDEX_NAME")
    pinecone_index_host: str = Field(default="", alias="PINECONE_INDEX_HOST")
    pinecone_namespace: str = Field(default="metadata", alias="PINECONE_NAMESPACE")
    pinecone_embed_model: str = Field(default="llama-text-embed-v2", alias="PINECONE_EMBED_MODEL")
    pinecone_rerank_model: str = Field(default="pinecone-rerank-v0", alias="PINECONE_RERANK_MODEL")

    llama_guard_url: str = Field(alias="LLAMA_GUARD_URL")
    llama_guard_model: str = Field(default="llm", alias="LLAMA_GUARD_MODEL")
    phi4_url: str = Field(alias="PHI4_URL")
    phi4_model: str = Field(default="llm", alias="PHI4_MODEL")

    neo4j_url: str = Field(alias="NEO4J_URL")
    neo4j_username: str = Field(alias="NEO4J_USERNAME")
    neo4j_password: str = Field(alias="NEO4J_PASSWORD")

    runpod_api_key: str = Field(alias="RUNPOD_API_KEY")
    runpod_endpoint_id: str = Field(alias="RUNPOD_ENDPOINT_ID")
    runpod_base_url: str = Field(default="https://api.runpod.ai/v2", alias="RUNPOD_BASE_URL")
    runpod_status_poll_interval: float = Field(default=2.0, alias="RUNPOD_STATUS_POLL_INTERVAL")
    runpod_status_timeout: int = Field(default=300, alias="RUNPOD_STATUS_TIMEOUT")

    top_k_columns: int = Field(default=8, alias="TOP_K_COLUMNS")
    top_k_tables: int = Field(default=5, alias="TOP_K_TABLES")
    initial_retrieval_multiplier: int = Field(default=3, alias="INITIAL_RETRIEVAL_MULTIPLIER")

    # ── App metadata ──────────────────────────────────────────────────────────
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    app_name: str = Field(default="sovereign-sql-backend", alias="APP_NAME")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @property
    def sqlite_conn_str(self) -> str:
        return (
            f"sqlitecloud://{self.sqlite_host}:{self.sqlite_port}/"
            f"{self.sqlite_metadata_db}?apikey={self.sqlite_api_key}"
        )

    @property
    def sqlite_data_conn_str(self) -> str:
        return (
            f"sqlitecloud://{self.sqlite_host}:{self.sqlite_port}/"
            f"{self.sqlite_db}?apikey={self.sqlite_api_key}"
        )

    @property
    def sqlite_observability_conn_str(self) -> str:
        return (
            f"sqlitecloud://{self.sqlite_host}:{self.sqlite_port}/"
            f"{self.sqlite_observability_db}?apikey={self.sqlite_api_key}"
        )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# Cached singleton
_settings: BackendSettings | None = None


def get_settings() -> BackendSettings:
    global _settings
    if _settings is None:
        _settings = BackendSettings()
    return _settings
