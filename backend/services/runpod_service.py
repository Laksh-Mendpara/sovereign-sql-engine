"""Runpod client for Arctic inference."""

from __future__ import annotations

import logging
import time

from .http_utils import get_json, post_json


class RunpodService:
    def __init__(
        self,
        api_key: str,
        endpoint_id: str,
        base_url: str,
        poll_interval: float,
        timeout_seconds: int,
        logger: logging.Logger,
    ) -> None:
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
        self.logger = logger

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def run_request(self, payload: dict) -> dict:
        run_url = f"{self.base_url}/{self.endpoint_id}/run"
        self.logger.info("Submitting chat payload to Runpod endpoint %s", self.endpoint_id)
        response = post_json(run_url, payload, headers=self.headers, timeout=180)

        if response.get("status", "").upper() == "COMPLETED":
            return response

        job_id = response.get("id")
        if not job_id:
            return response

        status_url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            status_response = get_json(status_url, headers=self.headers, timeout=180)
            status = str(status_response.get("status", "")).upper()
            self.logger.debug("Runpod job %s status=%s", job_id, status)
            if status == "COMPLETED":
                return status_response
            if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                return status_response
            time.sleep(self.poll_interval)

        raise TimeoutError(f"Runpod job {job_id} did not complete within {self.timeout_seconds}s")
