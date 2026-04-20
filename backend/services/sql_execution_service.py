"""SQLite Cloud query execution service for read-only SQL."""

from __future__ import annotations

import logging
import threading
from typing import Any

import sqlitecloud


class SqlExecutionService:
    def __init__(self, conn_str: str, logger: logging.Logger) -> None:
        self.conn = sqlitecloud.connect(conn_str)
        self._lock = threading.Lock()
        self.logger = logger

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def execute(self, sql: str, limit: int = 30) -> list[dict[str, Any]]:
        """Execute a read-only SQL query and return JSON-friendly rows."""
        with self._lock:
            cursor = self.conn.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in (cursor.description or [])]

        result: list[dict[str, Any]] = []
        for row in rows[:limit]:
            if isinstance(row, dict):
                result.append({k: self._serialize_value(v) for k, v in row.items()})
                continue

            values = list(row)
            result.append(
                {
                    columns[i] if i < len(columns) else f"col_{i}": self._serialize_value(value)
                    for i, value in enumerate(values)
                }
            )

        self.logger.info(
            "Executed SQL query against SQLite DB",
            extra={"row_count": len(result), "column_count": len(columns)},
        )
        return result
