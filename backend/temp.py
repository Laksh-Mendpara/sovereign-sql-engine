"""Temporary helper for creating the observability schema in SQLite Cloud."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os

import sqlitecloud

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS request_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL UNIQUE,
        trace_id TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        query TEXT NOT NULL,
        prompt_json TEXT,
        guard_allowed INTEGER,
        guard_reason TEXT,
        classification_label TEXT,
        classification_reason TEXT,
        generated_sql TEXT,
        execution_sql TEXT,
        execution_result_json TEXT,
        execution_row_count INTEGER,
        execution_status TEXT,
        terminal_state TEXT,
        error_type TEXT,
        error_message TEXT,
        raw_runpod_response_json TEXT,
        metadata_json TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_request_records_trace_id ON request_records(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_request_records_status ON request_records(execution_status)",
    """
    CREATE TABLE IF NOT EXISTS validation_failures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT,
        trace_id TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        input_text TEXT,
        expected_schema TEXT,
        actual_value TEXT,
        validator_name TEXT,
        error_type TEXT NOT NULL,
        error_message TEXT,
        stage TEXT,
        raw_payload_json TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_validation_failures_trace_id ON validation_failures(trace_id)",
    """
    CREATE TABLE IF NOT EXISTS logical_failures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT,
        trace_id TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        query TEXT,
        model_output TEXT,
        expected_output TEXT,
        is_correct INTEGER NOT NULL DEFAULT 0,
        failure_reason TEXT,
        review_status TEXT NOT NULL DEFAULT 'new',
        correction_json TEXT,
        notes TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_logical_failures_trace_id ON logical_failures(trace_id)",
    """
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT,
        trace_id TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        query TEXT,
        response TEXT,
        feedback_type TEXT NOT NULL,
        comment TEXT,
        rating INTEGER,
        user_id TEXT,
        session_id TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_user_feedback_trace_id ON user_feedback(trace_id)",
)


def build_observability_conn_str() -> str:
    conn_str = os.environ.get("SQLITE_OBSERVABILITY_CONN_STR")
    if conn_str:
        return conn_str

    host = os.environ.get("SQLITE_HOST")
    port = os.environ.get("SQLITE_PORT")
    db = os.environ.get("SQLITE_LLM_OBSERVABILITY_DB")
    api_key = os.environ.get("SQLITE_API_KEY")

    if not all([host, port, db, api_key]):
        raise SystemExit(
            "Set SQLITE_OBSERVABILITY_CONN_STR or SQLITE_HOST/SQLITE_PORT/"
            "SQLITE_LLM_OBSERVABILITY_DB/SQLITE_API_KEY"
        )

    return f"sqlitecloud://{host}:{port}/{db}?apikey={api_key}"


def create_observability_tables(conn_str: str) -> None:
    conn = sqlitecloud.connect(conn_str)
    try:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        if hasattr(conn, "commit"):
            conn.commit()
    finally:
        conn.close()


def main() -> None:
    create_observability_tables(build_observability_conn_str())


if __name__ == "__main__":
    main()
