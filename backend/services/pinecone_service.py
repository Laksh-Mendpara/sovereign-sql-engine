"""Pinecone retrieval service."""

from __future__ import annotations

import logging

from pinecone import Pinecone

from ..logic_models import RetrievedColumn, RetrievedTable

from .metadata_service import MetadataService


class PineconeService:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        index_host: str,
        namespace: str,
        embed_model: str,
        rerank_model: str,
        db_name: str,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.namespace = namespace
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.db_name = db_name
        self.client = Pinecone(api_key=api_key)
        if index_host:
            self.index = self.client.Index(host=index_host)
        else:
            self.index = self.client.Index(name=index_name)

    @staticmethod
    def _match_metadata(match) -> dict:
        metadata = getattr(match, "metadata", None)
        if metadata:
            return dict(metadata)
        if isinstance(match, dict):
            return dict(match.get("metadata", {}) or {})
        return {}

    @staticmethod
    def _match_score(match) -> float:
        score = getattr(match, "score", None)
        if score is not None:
            return float(score)
        if isinstance(match, dict):
            return float(match.get("score", 0.0))
        return 0.0

    def _query_with_db_fallback(self, vector: list[float], top_k: int, category: str):
        strict_filter = {"category": {"$eq": category}, "db": {"$eq": self.db_name}}
        self.logger.debug(
            "Pinecone query namespace=%s category=%s top_k=%d filter=%s",
            self.namespace,
            category,
            top_k,
            strict_filter,
        )
        try:
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=self.namespace,
                filter=strict_filter,
                include_metadata=True,
            )
        except Exception as exc:
            self.logger.warning(
                "Pinecone query unavailable for category=%s; returning no matches. error=%s",
                category,
                exc,
            )
            return []
        matches = list(response.matches)
        self.logger.info(
            "Pinecone returned %d matches for category=%s with db filter db=%s",
            len(matches),
            category,
            self.db_name,
        )
        if matches:
            return matches

        fallback_filter = {"category": {"$eq": category}}
        self.logger.warning(
            "No Pinecone matches with db filter db=%s for category=%s; retrying without db filter",
            self.db_name,
            category,
        )
        fallback_response = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.namespace,
            filter=fallback_filter,
            include_metadata=True,
        )
        fallback_matches = list(fallback_response.matches)
        self.logger.info(
            "Pinecone returned %d fallback matches for category=%s without db filter",
            len(fallback_matches),
            category,
        )
        return fallback_matches

    def _embed_query(self, query: str) -> list[float]:
        self.logger.debug("Embedding query for Pinecone retrieval")
        try:
            embeddings = self.client.inference.embed(
                model=self.embed_model,
                inputs=[query],
                parameters={"input_type": "query", "truncate": "END"},
            )
        except Exception as exc:
            self.logger.warning("Pinecone embedding unavailable; returning no retrieval vector. error=%s", exc)
            return []
        return embeddings[0]["values"]

    def fetch_top_columns(
        self,
        query: str,
        metadata_service: MetadataService,
        top_k: int,
        initial_multiplier: int,
    ) -> list[RetrievedColumn]:
        self.logger.info("Fetching top similar columns from Pinecone")
        vector = self._embed_query(query)
        if not vector:
            return []
        matches = self._query_with_db_fallback(vector, top_k * initial_multiplier, "col")

        raw_matches = []
        for match in matches:
            metadata = self._match_metadata(match)
            table_name = metadata.get("table_name", "")
            column_name = metadata.get("name", "")
            if table_name and column_name:
                raw_matches.append(
                    {
                        "table_name": table_name,
                        "column_name": column_name,
                        "vector_score": self._match_score(match),
                    }
                )

        self.logger.info("Parsed %d column candidates from Pinecone metadata", len(raw_matches))

        documents = metadata_service.get_column_documents(
            [(item["table_name"], item["column_name"]) for item in raw_matches]
        )

        rerank_documents = []
        filtered_matches = []
        for item in raw_matches:
            text = documents.get((item["table_name"], item["column_name"]), "").strip()
            if not text:
                continue
            filtered_matches.append(item)
            rerank_documents.append(
                {
                    "text": text,
                    "table_name": item["table_name"],
                    "column_name": item["column_name"],
                }
            )

        if not rerank_documents:
            self.logger.warning("No rerankable column documents were found in metadata storage")
            return []

        rerank = self.client.inference.rerank(
            model=self.rerank_model,
            query=query,
            documents=rerank_documents,
            rank_fields=["text"],
            top_n=top_k,
            return_documents=True,
        )

        results: list[RetrievedColumn] = []
        for item in rerank.data:
            document = item["document"]
            table_name = document["table_name"]
            column_name = document["column_name"]
            vector_score = 0.0
            for candidate in filtered_matches:
                if candidate["table_name"] == table_name and candidate["column_name"] == column_name:
                    vector_score = candidate["vector_score"]
                    break
            results.append(
                RetrievedColumn(
                    table_name=table_name,
                    column_name=column_name,
                    text=document["text"],
                    vector_score=vector_score,
                    rerank_score=float(item["score"]),
                )
            )
        return results

    def fetch_top_tables(self, query: str, metadata_service: MetadataService, top_k: int) -> list[RetrievedTable]:
        self.logger.info("Fetching top similar tables from Pinecone")
        vector = self._embed_query(query)
        if not vector:
            return []
        matches = self._query_with_db_fallback(vector, top_k, "table")

        table_names = []
        table_scores = {}
        for match in matches:
            metadata = self._match_metadata(match)
            table_name = metadata.get("name", "")
            if table_name:
                table_names.append(table_name)
                table_scores[table_name] = self._match_score(match)

        self.logger.info("Parsed %d table candidates from Pinecone metadata", len(table_names))

        documents = metadata_service.get_table_documents(table_names)
        return [
            RetrievedTable(
                table_name=table_name,
                text=documents.get(table_name, ""),
                vector_score=table_scores.get(table_name, 0.0),
            )
            for table_name in table_names
        ]
