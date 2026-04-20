"""OpenAI-compatible Modal chat client."""

from __future__ import annotations

import logging

from .http_utils import post_json


class ModalChatClient:
    def __init__(self, base_url: str, model: str, logger: logging.Logger) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.logger = logger

    @staticmethod
    def _adapt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not messages:
            return messages

        if messages[0].get("role") != "system":
            return messages

        system_text = messages[0].get("content", "").strip()
        remaining = messages[1:]
        if not remaining:
            return [{"role": "user", "content": system_text}]

        first = remaining[0]
        if first.get("role") != "user":
            return [
                {
                    "role": "user",
                    "content": (
                        "Follow these instructions for this conversation:\n"
                        f"{system_text}\n\n"
                        "User request:"
                    ).strip(),
                },
                *remaining,
            ]

        adapted_first = {
            "role": "user",
            "content": (
                "Follow these instructions for this conversation:\n"
                f"{system_text}\n\n"
                f"User request: {first.get('content', '').strip()}"
            ),
        }
        return [adapted_first, *remaining[1:]]

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float = 0.0) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        adapted_messages = self._adapt_messages(messages)
        payload = {
            "model": self.model,
            "messages": adapted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        self.logger.debug("POST %s with %d adapted messages", url, len(adapted_messages))
        response = post_json(url, payload, timeout=180)
        return response["choices"][0]["message"]["content"].strip()
