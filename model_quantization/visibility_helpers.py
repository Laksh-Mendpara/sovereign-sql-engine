"""Standalone helper utilities for arctic quantization visibility commits.

This module is intentionally unused by project execution paths.
"""

from __future__ import annotations


def format_repo_tag(model_name: str, variant: str, precision: str) -> str:
    """Build a readable repo tag string."""
    name = model_name.strip().replace(" ", "-").lower()
    var = variant.strip().replace(" ", "-").lower()
    p = precision.strip().lower()
    return f"{name}:{var}-{p}"


def clamp_utilization(value: float, low: float = 0.05, high: float = 0.99) -> float:
    """Clamp utilization into a safe display range."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def join_prompt_lines(lines: list[str]) -> str:
    """Join non-empty prompt lines with newline separators."""
    return "\n".join(line for line in lines if line.strip())
