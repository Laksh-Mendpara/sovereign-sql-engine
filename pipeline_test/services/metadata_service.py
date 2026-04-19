"""SQLite Cloud metadata store client."""

from __future__ import annotations

import logging

import sqlitecloud


class MetadataService:
    def __init__(self, conn_str: str, logger: logging.Logger) -> None:
        self.conn = sqlitecloud.connect(conn_str)
        self.logger = logger

    def close(self) -> None:
        self.conn.close()

    def get_column_documents(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        documents: dict[tuple[str, str], str] = {}
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
        blocks: list[str] = []
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
                    cm.friendly_name,
                    cm.description,
                    cm.data_format,
                    cm.value_description,
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
            for original_name, friendly_name, description, data_format, value_description, example in columns:
                col_type = data_format or "TEXT"
                comments = []
                if friendly_name:
                    comments.append(f"friendly name: {friendly_name}")
                if description:
                    comments.append(f"description: {description}")
                if value_description:
                    comments.append(f"value description: {value_description}")
                if example:
                    comments.append(f"examples: {example}")

                line = f"    {original_name} {col_type}"
                if comments:
                    line += f", -- {'; '.join(comments)}"
                column_lines.append(line)

            block = [f"CREATE TABLE {table_name} ("]
            block.append(",\n".join(column_lines))
            block.append(");")
            if table_description:
                block.append(f"-- table description: {table_description}")
            if relationships:
                block.append(f"-- relationships: {relationships}")
            blocks.append("\n".join(block))

        schema_sql = "\n\n".join(blocks)
        self.logger.info("Built schema SQL for %d tables", len(table_names))
        return schema_sql
