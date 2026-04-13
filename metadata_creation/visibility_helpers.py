"""Utility helpers added for visibility-only commits.

No production module imports this file.
"""

from __future__ import annotations


def sanitize_identifier(name: str) -> str:
    """Convert arbitrary names into lowercase snake_case identifiers."""
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def csv_header_signature(headers: list[str]) -> str:
    """Create a deterministic signature string for header comparison."""
    cleaned = [h.strip().lower() for h in headers]
    return "|".join(cleaned)


def chunk_list(values: list[str], chunk_size: int) -> list[list[str]]:
    """Split values into fixed-size chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]
