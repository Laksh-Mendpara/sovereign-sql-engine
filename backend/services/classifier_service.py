"""Phi-4 query classifier service."""

from __future__ import annotations

import logging

from ..logic_models import ClassificationResult
from ..prompts import build_classifier_messages, parse_classifier_json

from .modal_chat import ModalChatClient


class ClassifierService:
    def __init__(self, client: ModalChatClient, logger: logging.Logger) -> None:
        self.client = client
        self.logger = logger

    def classify(self, query: str) -> ClassificationResult:
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
