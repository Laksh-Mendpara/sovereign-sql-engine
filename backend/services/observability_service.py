"""Durable observability writer for audit and feedback records."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import sqlitecloud

from ..temp import create_observability_tables


class ObservabilityService:
    def __init__(self, conn_str: str, logger: logging.Logger) -> None:
        self.conn = sqlitecloud.connect(conn_str)
        self._lock = threading.Lock()
        self.logger = logger
        create_observability_tables(conn_str)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    @staticmethod
    def _json(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        with self._lock:
            self.conn.execute(sql, params)
            if hasattr(self.conn, "commit"):
                self.conn.commit()

    def upsert_request_record(self, record: dict[str, Any]) -> None:
        self._execute(
            """
            INSERT INTO request_records (
                request_id, trace_id, query, prompt_json, guard_allowed, guard_reason,
                classification_label, classification_reason, generated_sql, execution_sql,
                execution_result_json, execution_row_count, execution_status, terminal_state,
                error_type, error_message, raw_runpod_response_json, metadata_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(request_id) DO UPDATE SET
                trace_id=excluded.trace_id,
                query=excluded.query,
                prompt_json=excluded.prompt_json,
                guard_allowed=excluded.guard_allowed,
                guard_reason=excluded.guard_reason,
                classification_label=excluded.classification_label,
                classification_reason=excluded.classification_reason,
                generated_sql=excluded.generated_sql,
                execution_sql=excluded.execution_sql,
                execution_result_json=excluded.execution_result_json,
                execution_row_count=excluded.execution_row_count,
                execution_status=excluded.execution_status,
                terminal_state=excluded.terminal_state,
                error_type=excluded.error_type,
                error_message=excluded.error_message,
                raw_runpod_response_json=excluded.raw_runpod_response_json,
                metadata_json=excluded.metadata_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                record.get("request_id"),
                record.get("trace_id"),
                record.get("query"),
                self._json(record.get("prompt_json")),
                int(bool(record["guard_allowed"])) if record.get("guard_allowed") is not None else None,
                record.get("guard_reason"),
                record.get("classification_label"),
                record.get("classification_reason"),
                record.get("generated_sql"),
                record.get("execution_sql"),
                self._json(record.get("execution_result_json")),
                record.get("execution_row_count"),
                record.get("execution_status"),
                record.get("terminal_state"),
                record.get("error_type"),
                record.get("error_message"),
                self._json(record.get("raw_runpod_response_json")),
                self._json(record.get("metadata_json")),
            ),
        )

    def record_validation_failure(
        self,
        *,
        request_id: str | None,
        trace_id: str | None,
        input_text: str | None,
        expected_schema: Any = None,
        actual_value: Any = None,
        validator_name: str | None = None,
        error_type: str,
        error_message: str | None = None,
        stage: str | None = None,
        raw_payload_json: Any = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO validation_failures (
                request_id, trace_id, input_text, expected_schema, actual_value,
                validator_name, error_type, error_message, stage, raw_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                trace_id,
                input_text,
                self._json(expected_schema),
                self._json(actual_value),
                validator_name,
                error_type,
                error_message,
                stage,
                self._json(raw_payload_json),
            ),
        )

    def record_logical_failure(
        self,
        *,
        request_id: str | None,
        trace_id: str | None,
        query: str | None,
        model_output: Any = None,
        expected_output: Any = None,
        is_correct: bool = False,
        failure_reason: str | None = None,
        review_status: str = "new",
        correction_json: Any = None,
        notes: str | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO logical_failures (
                request_id, trace_id, query, model_output, expected_output, is_correct,
                failure_reason, review_status, correction_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                trace_id,
                query,
                self._json(model_output),
                self._json(expected_output),
                1 if is_correct else 0,
                failure_reason,
                review_status,
                self._json(correction_json),
                notes,
            ),
        )

    def record_user_feedback(
        self,
        *,
        request_id: str | None,
        trace_id: str | None,
        query: str | None,
        response: Any = None,
        feedback_type: str,
        comment: str | None = None,
        rating: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO user_feedback (
                request_id, trace_id, query, response, feedback_type, comment,
                rating, user_id, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                trace_id,
                query,
                self._json(response),
                feedback_type,
                comment,
                rating,
                user_id,
                session_id,
            ),
        )
