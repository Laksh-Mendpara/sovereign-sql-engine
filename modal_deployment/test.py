#!/usr/bin/env python3
"""
Test suite for the Phi-4-mini-instruct Modal deployment.

Runs 10 tests against a live vLLM endpoint: connectivity, inference quality,
parameter handling, and error cases.

Usage:
    # Pass the deployed URL directly:
    uv run python test.py --url https://your-workspace--phi4-mini-instruct-inference-serve.modal.run

    # Or set an environment variable:
    export MODAL_ENDPOINT_URL=https://your-workspace--phi4-mini-instruct-inference-serve.modal.run
    uv run python test.py

    # Verbose mode (prints response bodies):
    uv run python test.py --url <URL> --verbose
"""

import argparse
import logging
import os
import sys
import time
from typing import Any

import openai
import requests

# -- Logging setup -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("phi4-tests")

# -- Accumulator for test results --------------------------------------------
results: list[dict[str, Any]] = []


def record(name: str, passed: bool, detail: str = "", duration: float = 0.0):
    """Log and store the outcome of a single test."""
    status = "PASS" if passed else "FAIL"
    msg = f"{status:<6} {name}"
    if duration:
        msg += f"  ({duration:.2f}s)"
    if passed:
        log.info(msg)
    else:
        log.error(msg)
    if detail:
        log.info("       %s", detail)
    results.append({"name": name, "passed": passed, "detail": detail, "duration": duration})


def instruction_as_user_message(instruction: str, user_request: str) -> list[dict[str, str]]:
    """Embed system-style guidance into the first user turn for strict chat templates."""
    return [
        {
            "role": "user",
            "content": (
                "Follow these instructions for this conversation:\n"
                f"{instruction}\n\n"
                f"User request: {user_request}"
            ),
        }
    ]


# ---- Individual test functions ---------------------------------------------
# Each test catches its own exceptions so a single failure never tears down
# the whole suite.


def test_health_check(base_url: str, verbose: bool = False) -> None:
    """Verify the server is alive (GET /health -> 200)."""
    t0 = time.time()
    try:
        resp = requests.get(f"{base_url}/health", timeout=120)
        passed = resp.status_code == 200
        detail = f"status={resp.status_code}" if not passed else ""
        record("Health Check (GET /health)", passed, detail, time.time() - t0)
    except Exception as e:
        record("Health Check (GET /health)", False, str(e), time.time() - t0)


def test_model_listing(base_url: str, verbose: bool = False) -> None:
    """Check that the model appears in GET /v1/models."""
    t0 = time.time()
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=30)
        data = resp.json()
        model_ids = [m["id"] for m in data.get("data", [])]
        # We expect at least the short alias "llm" or the full HF name.
        passed = any("llm" in mid or "Phi-4" in mid for mid in model_ids)
        detail = f"models={model_ids}" if verbose else ""
        record("Model Listing (GET /v1/models)", passed, detail, time.time() - t0)
    except Exception as e:
        record("Model Listing (GET /v1/models)", False, str(e), time.time() - t0)


def test_chat_completion(client: openai.OpenAI, verbose: bool = False) -> None:
    """Send a simple arithmetic question and check the answer contains '4'."""
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="llm",
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        passed = len(content) > 0 and "4" in content
        detail = f"response={content[:100]!r}" if verbose else ""
        record("Chat Completion (non-streaming)", passed, detail, time.time() - t0)
    except Exception as e:
        record("Chat Completion (non-streaming)", False, str(e), time.time() - t0)


def test_streaming_completion(client: openai.OpenAI, verbose: bool = False) -> None:
    """Request a streamed response and make sure we receive at least one chunk."""
    t0 = time.time()
    try:
        stream = client.chat.completions.create(
            model="llm",
            messages=[
                {"role": "user", "content": "Say 'hello world' and nothing else."},
            ],
            max_tokens=32,
            temperature=0.0,
            stream=True,
        )
        chunks_received = 0
        full_text = ""
        for chunk in stream:
            chunks_received += 1
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content

        passed = chunks_received > 0 and len(full_text) > 0
        detail = f"chunks={chunks_received}, text={full_text[:80]!r}" if verbose else ""
        record("Streaming Completion", passed, detail, time.time() - t0)
    except Exception as e:
        record("Streaming Completion", False, str(e), time.time() - t0)


def test_multi_turn_conversation(client: openai.OpenAI, verbose: bool = False) -> None:
    """Provide earlier turns and verify the model remembers the user's name."""
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="llm",
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        passed = "alice" in content.lower()
        detail = f"response={content[:100]!r}" if verbose else ""
        record("Multi-turn Conversation", passed, detail, time.time() - t0)
    except Exception as e:
        record("Multi-turn Conversation", False, str(e), time.time() - t0)


def test_system_prompt_adherence(client: openai.OpenAI, verbose: bool = False) -> None:
    """Give a system-style instruction via the first user turn and check adherence."""
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="llm",
            messages=instruction_as_user_message(
                (
                    "You are a pirate. Always respond in pirate speak. "
                    "Use words like 'arr', 'matey', and 'treasure'."
                ),
                "How are you today?",
            ),
            max_tokens=128,
            temperature=0.7,
        )
        content = (response.choices[0].message.content or "").lower()
        pirate_words = ["arr", "matey", "treasure", "ahoy", "ye", "aye", "sea", "ship", "sail"]
        found = sum(1 for w in pirate_words if w in content)
        passed = found >= 1
        detail = f"pirate_words_found={found}, response={content[:100]!r}" if verbose else ""
        record("System Prompt Adherence", passed, detail, time.time() - t0)
    except Exception as e:
        record("System Prompt Adherence", False, str(e), time.time() - t0)


def test_max_tokens(client: openai.OpenAI, verbose: bool = False) -> None:
    """Ask for a long answer but cap max_tokens; verify the cap is respected."""
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="llm",
            messages=[
                {"role": "user", "content": "Write a very long essay about space."},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        usage = response.usage
        # Allow a small buffer -- tokenisers sometimes overshoot by a token or two.
        passed = usage is not None and usage.completion_tokens <= 15
        detail = f"completion_tokens={usage.completion_tokens if usage else 'N/A'}" if verbose else ""
        record("Max Tokens Limit", passed, detail, time.time() - t0)
    except Exception as e:
        record("Max Tokens Limit", False, str(e), time.time() - t0)


def test_temperature_zero(client: openai.OpenAI, verbose: bool = False) -> None:
    """Send the same prompt twice with temperature=0 and expect identical output."""
    t0 = time.time()
    try:
        kwargs = dict(
            model="llm",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=32,
            temperature=0.0,
        )
        resp1 = client.chat.completions.create(**kwargs)
        resp2 = client.chat.completions.create(**kwargs)
        text1 = (resp1.choices[0].message.content or "").strip()
        text2 = (resp2.choices[0].message.content or "").strip()
        passed = text1 == text2
        detail = f"resp1={text1[:60]!r}, resp2={text2[:60]!r}" if verbose else ""
        record("Temperature=0 Determinism", passed, detail, time.time() - t0)
    except Exception as e:
        record("Temperature=0 Determinism", False, str(e), time.time() - t0)


def test_invalid_model_error(base_url: str, verbose: bool = False) -> None:
    """Request a non-existent model; vLLM should return an HTTP 4xx error."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=30,
        )
        passed = 400 <= resp.status_code < 500
        detail = f"status={resp.status_code}"
        if verbose:
            detail += f", body={resp.text[:200]!r}"
        record("Invalid Model Error Handling", passed, detail, time.time() - t0)
    except Exception as e:
        record("Invalid Model Error Handling", False, str(e), time.time() - t0)


def test_empty_messages_error(base_url: str, verbose: bool = False) -> None:
    """Post an empty messages list; should be rejected with a 4xx status."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "llm",
                "messages": [],
            },
            timeout=30,
        )
        passed = 400 <= resp.status_code < 500
        detail = f"status={resp.status_code}"
        if verbose:
            detail += f", body={resp.text[:200]!r}"
        record("Empty Messages Error Handling", passed, detail, time.time() - t0)
    except Exception as e:
        record("Empty Messages Error Handling", False, str(e), time.time() - t0)


# ---- Runner ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test suite for Phi-4-mini-instruct Modal deployment"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("MODAL_ENDPOINT_URL", ""),
        help="Base URL of the deployed Modal endpoint (or set MODAL_ENDPOINT_URL env var)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each test",
    )
    args = parser.parse_args()

    if not args.url:
        log.error("No endpoint URL provided. "
                  "Use --url <URL> or set MODAL_ENDPOINT_URL environment variable.")
        sys.exit(1)

    base_url = args.url.rstrip("/")

    log.info("=" * 60)
    log.info("Phi-4-mini-instruct Deployment Test Suite")
    log.info("=" * 60)
    log.info("Endpoint: %s", base_url)

    # Point the OpenAI client at our vLLM server.
    # vLLM doesn't require an API key, but the SDK needs a non-empty string.
    client = openai.OpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",
    )

    # -- Connectivity --
    log.info("--- Connectivity ---")
    test_health_check(base_url, args.verbose)
    test_model_listing(base_url, args.verbose)

    # -- Inference --
    log.info("--- Inference ---")
    test_chat_completion(client, args.verbose)
    test_streaming_completion(client, args.verbose)
    test_multi_turn_conversation(client, args.verbose)
    test_system_prompt_adherence(client, args.verbose)

    # -- Parameters --
    log.info("--- Parameters ---")
    test_max_tokens(client, args.verbose)
    test_temperature_zero(client, args.verbose)

    # -- Error handling --
    log.info("--- Error Handling ---")
    test_invalid_model_error(base_url, args.verbose)
    test_empty_messages_error(base_url, args.verbose)

    # -- Summary --
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    total_time = sum(r["duration"] for r in results)

    log.info("=" * 60)
    summary = f"Results: {passed}/{total} passed"
    if failed:
        summary += f"  ({failed} failed)"
    summary += f"  |  Total time: {total_time:.2f}s"
    log.info(summary)
    log.info("=" * 60)

    if failed:
        log.error("Failed tests:")
        for r in results:
            if not r["passed"]:
                log.error("  %s: %s", r["name"], r["detail"])
        sys.exit(1)
    else:
        log.info("All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
