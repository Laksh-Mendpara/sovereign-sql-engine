"""Development-only helper functions for visibility commits.

This module is not wired into the Modal app.
"""

from __future__ import annotations


def estimate_timeout(tokens: int, tokens_per_second: float, floor_seconds: int = 5) -> int:
    """Estimate timeout in seconds from token budget and throughput."""
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be > 0")
    est = int(tokens / tokens_per_second)
    return est if est > floor_seconds else floor_seconds


def merge_headers(base: dict[str, str], override: dict[str, str]) -> dict[str, str]:
    """Merge HTTP headers with case-insensitive override semantics."""
    merged = {k.lower(): v for k, v in base.items()}
    for key, value in override.items():
        merged[key.lower()] = value
    return merged


def to_event_stream_lines(chunks: list[str]) -> list[str]:
    """Convert plain chunks into SSE data lines."""
    return [f"data: {chunk}" for chunk in chunks]
