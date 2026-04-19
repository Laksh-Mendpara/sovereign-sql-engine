"""Phi-4 query classifier service."""

from __future__ import annotations

import logging
import re

from ..logic_models import ClassificationResult
from ..prompts import build_classifier_messages, parse_classifier_json

from .modal_chat import ModalChatClient


class ClassifierService:
    def __init__(self, client: ModalChatClient, logger: logging.Logger) -> None:
        self.client = client
        self.logger = logger

    @staticmethod
    def _looks_obviously_out_of_topic(query: str) -> bool:
        lowered = query.lower()
        math_pattern = r"\b\d+\s*[\+\-\*/x×]\s*\d+\b"
        db_keywords = (
            "sql",
            "select",
            "from",
            "where",
            "join",
            "group by",
            "order by",
            "database",
            "table",
            "column",
            "rows",
        )
        return bool(re.search(math_pattern, lowered)) and not any(
            keyword in lowered for keyword in db_keywords
        )

    def classify(self, query: str) -> ClassificationResult:
        if self._looks_obviously_out_of_topic(query):
            self.logger.info("Classified query as out_of_topic via heuristic")
            return ClassificationResult(
                label="out_of_topic",
                reason="Arithmetic question, not a database query",
                raw_response="heuristic: arithmetic question",
            )

        self.logger.info("Classifying query difficulty/topic")
        raw = self.client.chat(build_classifier_messages(query), max_tokens=128, temperature=0.0)
        self.logger.debug("Classifier response: %s", raw)
        parsed = parse_classifier_json(raw)
        label = str(parsed.get("label", "")).strip().lower().replace(" ", "_")
        if label not in {"easy", "difficult", "out_of_topic"}:
            lowered = raw.lower()
            if "out_of_topic" in lowered or "out of topic" in lowered:
                label = "out_of_topic"
            elif "difficult" in lowered or "hard" in lowered or "complex" in lowered:
                label = "difficult"
            else:
                label = "easy"

        reason = str(parsed.get("reason", "")).strip() or raw.strip()
        return ClassificationResult(label=label, reason=reason, raw_response=raw)
