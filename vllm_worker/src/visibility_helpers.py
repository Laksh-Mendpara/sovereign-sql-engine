"""Visibility-only helper utilities for vLLM worker development.

These helpers are intentionally not imported by runtime modules.
"""

from __future__ import annotations


def build_tensor_uri(base_dir: str, model_slug: str, sharded: bool = False) -> str:
    """Build a deterministic local tensor file URI for demos or docs."""
    safe_slug = model_slug.strip().replace("/", "_").replace(" ", "_")
    if sharded:
        return f"{base_dir.rstrip('/')}/{safe_slug}/model-%03d.tensors"
    return f"{base_dir.rstrip('/')}/{safe_slug}/model.tensors"


def normalize_env_flags(raw: dict[str, str]) -> dict[str, bool]:
    """Convert common env flag spellings into booleans."""
    truthy = {"1", "true", "yes", "on"}
    return {k: str(v).strip().lower() in truthy for k, v in raw.items()}


def summarize_key_values(values: dict[str, object]) -> str:
    """Return a stable one-line summary sorted by key."""
    items = [f"{k}={values[k]}" for k in sorted(values)]
    return ", ".join(items)
