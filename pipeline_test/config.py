"""Configuration loading for the pipeline checker."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    db_name: str
    sqlite_metadata_conn_str: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_index_host: str
    pinecone_namespace: str
    pinecone_embed_model: str
    pinecone_rerank_model: str
    llama_guard_url: str
    llama_guard_model: str
    phi4_url: str
    phi4_model: str
    neo4j_url: str
    neo4j_username: str
    neo4j_password: str
    runpod_api_key: str
    runpod_endpoint_id: str
    runpod_base_url: str
    runpod_status_poll_interval: float
    runpod_status_timeout: int
    top_k_columns: int
    top_k_tables: int
    initial_retrieval_multiplier: int


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Environment variable {name} is required.")
    return value


def load_settings() -> Settings:
    load_dotenv()

    sqlite_metadata_conn_str = (
        f"sqlitecloud://{_required('SQLITE_HOST')}:{_required('SQLITE_PORT')}/"
        f"{_required('SQLITE_METADATA_DB')}?apikey={_required('SQLITE_API_KEY')}"
    )

    return Settings(
        db_name=_required("DB_NAME"),
        sqlite_metadata_conn_str=sqlite_metadata_conn_str,
        pinecone_api_key=_required("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "").strip(),
        pinecone_index_host=os.getenv("PINECONE_INDEX_HOST", "").strip(),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "metadata").strip(),
        pinecone_embed_model=os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2").strip(),
        pinecone_rerank_model=os.getenv("PINECONE_RERANK_MODEL", "pinecone-rerank-v0").strip(),
        llama_guard_url=_required("LLAMA_GUARD_URL"),
        llama_guard_model=os.getenv("LLAMA_GUARD_MODEL", "llm").strip(),
        phi4_url=_required("PHI4_URL"),
        phi4_model=os.getenv("PHI4_MODEL", "llm").strip(),
        neo4j_url=_required("NEO4J_URL"),
        neo4j_username=_required("NEO4J_USERNAME"),
        neo4j_password=_required("NEO4J_PASSWORD"),
        runpod_api_key=_required("RUNPOD_API_KEY"),
        runpod_endpoint_id=_required("RUNPOD_ENDPOINT_ID"),
        runpod_base_url=os.getenv("RUNPOD_BASE_URL", "https://api.runpod.ai/v2").strip(),
        runpod_status_poll_interval=float(os.getenv("RUNPOD_STATUS_POLL_INTERVAL", "2")),
        runpod_status_timeout=int(os.getenv("RUNPOD_STATUS_TIMEOUT", "300")),
        top_k_columns=int(os.getenv("TOP_K_COLUMNS", "8")),
        top_k_tables=int(os.getenv("TOP_K_TABLES", "5")),
        initial_retrieval_multiplier=int(os.getenv("INITIAL_RETRIEVAL_MULTIPLIER", "3")),
    )

