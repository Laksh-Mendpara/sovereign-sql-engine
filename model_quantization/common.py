"""Common configuration and utility functions for model quantization.

This module provides shared constants, paths, and helper functions used across
the quantization, tensorization, and deployment scripts.
"""
import logging
import os
import sys
from pathlib import Path

# Base model identifier from Hugging Face Hub (default fallback)
BASE_MODEL_ID = "Snowflake/Arctic-Text2SQL-R1-7B"

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Local model storage directories
MODELS_DIR = PROJECT_ROOT / "models"


def get_model_name_from_id(model_id: str) -> str:
    """Extract the model name from a Hugging Face repo ID."""
    return model_id.split("/")[-1]


def get_default_output_dir(model_id: str, quant_type: str) -> Path:
    """Get the default local output directory for a quantized model.
    quant_type should be 'gptq8', 'awq4', 'tensorized', etc.
    """
    model_name = get_model_name_from_id(model_id).lower()
    return MODELS_DIR / f"{model_name}-{quant_type}"


def get_default_repo_id(model_id: str, quant_type: str) -> str:
    """Get the default Hugging Face repository ID for publishing."""
    from huggingface_hub import HfApi
    username = HfApi(token=ensure_hf_token()).whoami()["name"]
    model_name = get_model_name_from_id(model_id).lower()
    return f"{username}/{model_name}-{quant_type}"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure process-wide logging for the quantization scripts.

    Args:
        level: Logging level name such as DEBUG, INFO, or WARNING.

    Returns:
        logging.Logger: Logger scoped to this package.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("model_quantization")


def load_hf_token() -> str | None:
    """Load Hugging Face API token from environment or .env file.
    
    Checks environment variable first, then searches for .env files in the
    current directory and project root.
    
    Returns:
        str or None: The HF_TOKEN value if found, otherwise None.
    """
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    for env_path in (Path(".env"), PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "HF_TOKEN":
                return value.strip().strip("'\"")

    return None


def ensure_hf_token() -> str:
    """Ensure Hugging Face token is available, raise error if not found.
    
    Returns:
        str: The HF_TOKEN value.
        
    Raises:
        RuntimeError: If HF_TOKEN is not found in environment or .env files.
    """
    token = load_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN not found in environment or .env.")
    return token
