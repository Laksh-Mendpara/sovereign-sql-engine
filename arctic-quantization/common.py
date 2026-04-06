"""Common configuration and utility functions for Arctic model quantization.

This module provides shared constants, paths, and helper functions used across
the quantization, tensorization, and deployment scripts.
"""
import os
from pathlib import Path

# Base model identifier from Hugging Face Hub
BASE_MODEL_ID = "Snowflake/Arctic-Text2SQL-R1-7B"
# Default Hugging Face Hub repository IDs for quantized and tensorized models
DEFAULT_GPTQ8_REPO_ID = "ByteMaster01/arctic-text2sql-r1-7b-gptq8"
DEFAULT_TENSORIZED_REPO_ID = "ByteMaster01/arctic-text2sql-r1-7b-gptq8-tensorized"

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Local model storage directories
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_GPTQ8_DIR = MODELS_DIR / "arctic-text2sql-r1-7b-gptq8"  # GPTQ 8-bit quantized model
DEFAULT_Q8_DIR = MODELS_DIR / "arctic-text2sql-r1-7b-q8"  # BitsAndBytes 8-bit quantized model
DEFAULT_TENSORIZED_DIR = MODELS_DIR / "arctic-text2sql-r1-7b-gptq8-tensorized"  # vLLM tensorized model


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
