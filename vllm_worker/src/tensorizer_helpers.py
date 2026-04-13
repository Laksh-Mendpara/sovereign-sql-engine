"""Utility helpers for future tensorizer workflows.

These helpers are intentionally not wired into the worker yet. They are kept as
small, reusable building blocks for potential tensorizer and baked-model
maintenance tasks.
"""

from __future__ import annotations

from pathlib import Path


def build_tensorizer_output_name(model_name: str, rank: int | None = None) -> str:
    """Return a predictable tensorizer filename for a model."""
    safe_name = model_name.strip().replace("/", "-")
    if rank is None:
        return f"{safe_name}.tensors"
    return f"{safe_name}-rank-{rank:03d}.tensors"


def join_tensorizer_root(root_dir: str | Path, model_name: str, rank: int | None = None) -> Path:
    """Build a tensorizer artifact path under a root directory."""
    root_path = Path(root_dir)
    return root_path / "vllm" / model_name / build_tensorizer_output_name(model_name, rank)


def strip_tensorizer_suffix(path: str | Path) -> str:
    """Remove a trailing tensorizer suffix from a path string if present."""
    path_text = str(path)
    if path_text.endswith(".tensors"):
        return path_text[:-9]
    return path_text


def is_tensorizer_ranked_name(name: str) -> bool:
    """Return True when a filename looks like a sharded tensorizer artifact."""
    return "-rank-" in name and name.endswith(".tensors")
