"""SQLite Cloud metadata store client.

This client uses a threading.Lock to ensure thread-safety when accessed
via asyncio.to_thread in the parallel pipeline executor.
"""

from __future__ import annotations

import logging
import threading

import sqlitecloud


class MetadataService:
    def __init__(self, conn_str: str, logger: logging.Logger) -> None:
        self.conn = sqlitecloud.connect(conn_str)
        self._lock = threading.Lock()
        self.logger = logger

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def get_column_documents(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        documents: dict[tuple[str, str], str] = {}
        with self._lock:
            for table_name, column_name in pairs:
                cursor = self.conn.execute(
                    """
                    SELECT cm.description, cm.value_description
                    FROM column_metadata cm
                    JOIN table_metadata tm ON tm.id = cm.table_id
                    WHERE tm.table_name = ? AND cm.original_name = ?
                    """,
                    (table_name, column_name),
                )
                row = cursor.fetchone()
                if not row:
                    continue
                parts = [part.strip() for part in row if isinstance(part, str) and part.strip()]
                documents[(table_name, column_name)] = "\n".join(parts)
        return documents

    def get_table_documents(self, table_names: list[str]) -> dict[str, str]:
        documents: dict[str, str] = {}
        with self._lock:
            for table_name in table_names:
                cursor = self.conn.execute(
                    "SELECT description FROM table_metadata WHERE table_name = ?",
                    (table_name,),
                )
                row = cursor.fetchone()
                if row and isinstance(row[0], str):
                    documents[table_name] = row[0].strip()
        return documents

    def get_schema_sql(self, table_names: list[str]) -> str:
        """Retrieves SQL CREATE TABLE statements for all requested tables.

        Optimized for LLM context limits:
        - Omits friendly_name and value_description (wordy).
        - Truncates examples to 40 chars max.
        - Keeps column names, types, and primary descriptions.
        """
        blocks: list[str] = []
        with self._lock:
            for table_name in table_names:
                table_cursor = self.conn.execute(
                    "SELECT description, relationships FROM table_metadata WHERE table_name = ?",
                    (table_name,),
                )
                table_row = table_cursor.fetchone()
                table_description = table_row[0] if table_row else ""
                relationships = table_row[1] if table_row else ""

                column_cursor = self.conn.execute(
                    """
                    SELECT
                        cm.original_name,
                        cm.description,
                        cm.data_format,
                        cm.example
                    FROM column_metadata cm
                    JOIN table_metadata tm ON tm.id = cm.table_id
                    WHERE tm.table_name = ?
                    ORDER BY cm.id
                    """,
                    (table_name,),
                )
                columns = column_cursor.fetchall()
                column_lines: list[str] = []
                for original_name, description, data_format, example in columns:
                    col_type = data_format or "TEXT"
                    comments = []

                    # Only keep the core description if it's short, or truncate
                    if description:
                        desc = description.strip()
                        if len(desc) > 60:
                            desc = desc[:57] + "..."
                        comments.append(desc)

                    # Only keep one short example
                    if example:
                        ex = example.strip()
                        if len(ex) > 40:
                            ex = ex[:37] + "..."
                        comments.append(f"ex: {ex}")

                    line = f"    {original_name} {col_type}"
                    if comments:
                        line += f", -- {'; '.join(comments)}"
                    column_lines.append(line)

                block = [f"CREATE TABLE {table_name} ("]
                block.append(",\n".join(column_lines))
                block.append(");")

                # Keep table-level metadata concise
                if table_description:
                    desc = table_description.strip()
                    if len(desc) > 100:
                        desc = desc[:97] + "..."
                    block.append(f"-- table description: {desc}")
                if relationships:
                    block.append(f"-- relationships: {relationships}")

                blocks.append("\n".join(block))

        self.logger.info("Built schema SQL for %d tables", len(table_names))
        return "\n\n".join(blocks)
