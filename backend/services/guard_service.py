"""Llama Guard wrapper service."""

from __future__ import annotations

import logging
import re

from ..logic_models import GuardResult
from ..prompts import build_guard_messages, parse_guard_response

from .modal_chat import ModalChatClient


class GuardService:
    def __init__(self, client: ModalChatClient, logger: logging.Logger) -> None:
        self.client = client
        self.logger = logger

    @staticmethod
    def _looks_obviously_unsafe(query: str) -> bool:
        patterns = [
            r"\bmalware\b",
            r"\bransomware\b",
            r"\bexploit\b",
            r"\bhack\b",
            r"\bcredential\b",
            r"\bpassword\b",
            r"\bsteal\b",
            r"\bweapon\b",
            r"\bbomb\b",
            r"\bkill\b",
            r"\bself-harm\b",
            r"\bsuicide\b",
            r"\bchild sexual\b",
            r"\bcredit card\b",
            r"\bssn\b",
        ]
        lowered = query.lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    def check(self, query: str) -> GuardResult:
        self.logger.info("Running guardrail check")
        try:
            raw = self.client.chat(build_guard_messages(query), max_tokens=8, temperature=0.0)
            self.logger.debug("Guard response: %s", raw)
            allowed, reason = parse_guard_response(raw)
            if not allowed and not self._looks_obviously_unsafe(query):
                self.logger.warning(
                    "Guard model returned an unsafe-style response for a benign-looking query; allowing it. raw=%s",
                    raw,
                )
                return GuardResult(allowed=True, reason="", raw_response=raw)
            return GuardResult(allowed=allowed, reason=reason, raw_response=raw)
        except Exception as exc:
            self.logger.warning(
                "Guard service unavailable; failing closed. error=%s",
                exc,
            )
            return GuardResult(
                allowed=False,
                reason="Guard service unavailable; request blocked as fail-safe",
                raw_response=f"error: {exc}",
            )
