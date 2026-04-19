"""Top-level orchestration for the end-to-end pipeline check."""

from __future__ import annotations

import logging

from pipeline_test.config import Settings
from pipeline_test.models import PipelineResult
from pipeline_test.prompts import build_arctic_runpod_input
from pipeline_test.services.classifier_service import ClassifierService
from pipeline_test.services.guard_service import GuardService
from pipeline_test.services.metadata_service import MetadataService
from pipeline_test.services.modal_chat import ModalChatClient
from pipeline_test.services.neo4j_service import Neo4jService
from pipeline_test.services.pinecone_service import PineconeService
from pipeline_test.services.runpod_service import RunpodService


class PipelineChecker:
    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

        guard_client = ModalChatClient(settings.llama_guard_url, settings.llama_guard_model, logger)
        classifier_client = ModalChatClient(settings.phi4_url, settings.phi4_model, logger)

        self.guard_service = GuardService(guard_client, logger)
        self.classifier_service = ClassifierService(classifier_client, logger)
        self.metadata_service = MetadataService(settings.sqlite_metadata_conn_str, logger)
        self.pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            index_host=settings.pinecone_index_host,
            namespace=settings.pinecone_namespace,
            embed_model=settings.pinecone_embed_model,
            rerank_model=settings.pinecone_rerank_model,
            db_name=settings.db_name,
            logger=logger,
        )
        self.neo4j_service = Neo4jService(
            settings.neo4j_url,
            settings.neo4j_username,
            settings.neo4j_password,
            settings.db_name,
            logger,
        )
        self.runpod_service = RunpodService(
            api_key=settings.runpod_api_key,
            endpoint_id=settings.runpod_endpoint_id,
            base_url=settings.runpod_base_url,
            poll_interval=settings.runpod_status_poll_interval,
            timeout_seconds=settings.runpod_status_timeout,
            logger=logger,
        )

    def close(self) -> None:
        self.metadata_service.close()
        self.neo4j_service.close()

    def run(self, query: str) -> PipelineResult:
        self.logger.info("Starting end-to-end pipeline check")
        self.logger.info("Input query: %s", query)

        guard = self.guard_service.check(query)
        self.logger.info("Guard decision allowed=%s reason=%s", guard.allowed, guard.reason or "<none>")
        classification = self.classifier_service.classify(query)
        self.logger.info("Query classified as %s", classification.label)

        if not guard.allowed:
            self.logger.warning("Guardrail rejected query: %s", guard.reason)
            return PipelineResult(
                query=query,
                guard=guard,
                classification=classification,
                retrieved_columns=[],
                retrieved_tables=[],
                selected_tables=[],
                schema_tables=[],
                schema_sql="",
                runpod_response={"skipped": True, "reason": f"guard_failed: {guard.reason}"},
            )

        if classification.label == "out_of_topic":
            return PipelineResult(
                query=query,
                guard=guard,
                classification=classification,
                retrieved_columns=[],
                retrieved_tables=[],
                selected_tables=[],
                schema_tables=[],
                schema_sql="",
                runpod_response={"skipped": True, "reason": classification.reason},
            )

        retrieved_columns = self.pinecone_service.fetch_top_columns(
            query=query,
            metadata_service=self.metadata_service,
            top_k=self.settings.top_k_columns,
            initial_multiplier=self.settings.initial_retrieval_multiplier,
        )
        retrieved_tables = self.pinecone_service.fetch_top_tables(
            query=query,
            metadata_service=self.metadata_service,
            top_k=self.settings.top_k_tables,
        )

        union_tables = {item.table_name for item in retrieved_columns}
        union_tables.update(item.table_name for item in retrieved_tables)
        selected_tables = sorted(union_tables)
        self.logger.info("Selected %d seed tables from retrieval", len(selected_tables))

        schema_tables = self.neo4j_service.expand_tables(selected_tables, classification.label)
        self.logger.info("Expanded to %d schema tables after Neo4j join resolution", len(schema_tables))

        schema_sql = self.metadata_service.get_schema_sql(schema_tables)
        runpod_payload = build_arctic_runpod_input(query, schema_sql)
        self.logger.info(
            "Submitting final Arctic request with %d schema tables and schema length %d chars",
            len(schema_tables),
            len(schema_sql),
        )
        runpod_response = self.runpod_service.run_request(runpod_payload)

        return PipelineResult(
            query=query,
            guard=guard,
            classification=classification,
            retrieved_columns=retrieved_columns,
            retrieved_tables=retrieved_tables,
            selected_tables=selected_tables,
            schema_tables=schema_tables,
            schema_sql=schema_sql,
            runpod_response=runpod_response,
        )
