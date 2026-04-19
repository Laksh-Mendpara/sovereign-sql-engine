"""Neo4j graph retrieval service."""

from __future__ import annotations

import logging

from neo4j import GraphDatabase


class Neo4jService:
    def __init__(self, url: str, username: str, password: str, db_name: str, logger: logging.Logger) -> None:
        self.logger = logger
        self.db_name = db_name
        self.driver = GraphDatabase.driver(url, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def expand_tables(self, seed_tables: list[str], difficulty: str) -> list[str]:
        if not seed_tables:
            return []

        max_hops = 2 if difficulty == "easy" else 4
        final_tables = set(seed_tables)

        self.logger.info(
            "Resolving Neo4j join tables from %d seed tables with max_hops=%d",
            len(seed_tables),
            max_hops,
        )

        with self.driver.session() as session:
            for i, start in enumerate(seed_tables):
                for end in seed_tables[i + 1 :]:
                    result = session.run(
                        f"""
                        MATCH p = shortestPath(
                            (a:Table {{name: $start, db: $db}})
                            -[:REFERENCES*..{max_hops}]-
                            (b:Table {{name: $end, db: $db}})
                        )
                        RETURN [node IN nodes(p) | node.name] AS names
                        """,
                        start=start,
                        end=end,
                        db=self.db_name,
                    )
                    row = result.single()
                    if row and row["names"]:
                        final_tables.update(row["names"])

        return sorted(final_tables)

