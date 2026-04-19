"""Top-level orchestration for the end-to-end pipeline check."""

from __future__ import annotations

import logging

from pipeline_test.config import Settings
from pipeline_test.models import PipelineResult
from pipeline_test.prompts import build_arctic_runpod_input
from pipeline_test.sql_utils import plan_sql_execution
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
        generated_sql = self._extract_sql(runpod_response)
        execution_plan = plan_sql_execution(generated_sql)

        return PipelineResult(
            query=query,
            guard=guard,
            classification=classification,
            retrieved_columns=retrieved_columns,
            retrieved_tables=retrieved_tables,
            selected_tables=selected_tables,
            schema_tables=schema_tables,
            schema_sql=schema_sql,
            generated_sql=generated_sql,
            execution_sql=execution_plan.execution_sql,
            execution_data=None,
            runpod_response=runpod_response,
        )

    @staticmethod
    def _extract_sql(runpod_response: dict) -> str | None:
        raw_text = None
        try:
            output = runpod_response.get("output")
            if isinstance(output, list) and len(output) > 0:
                first_item = output[0]
                if isinstance(first_item, dict):
                    choices = first_item.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        tokens = choices[0].get("tokens", [])
                        if tokens and isinstance(tokens, list):
                            raw_text = "".join(tokens)
                        elif choices[0].get("text"):
                            raw_text = choices[0].get("text")
            elif isinstance(output, dict):
                choices = output.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {})
                    if isinstance(msg, dict) and msg.get("content"):
                        raw_text = msg.get("content")
                    elif choices[0].get("tokens") and isinstance(choices[0].get("tokens"), list):
                        raw_text = "".join(choices[0].get("tokens", []))
                    elif choices[0].get("text"):
                        raw_text = choices[0].get("text")
            elif isinstance(output, str):
                raw_text = output
        except Exception:
            pass

        if not raw_text:
            return None

        raw_text = raw_text.strip()
        import re

        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return raw_text or None
