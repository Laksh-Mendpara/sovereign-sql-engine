"""Upload local quantized models to Hugging Face Hub.

This script facilitates uploading GPTQ-8 or tensorized models to the Hugging Face Hub
with proper authentication and optional privacy settings.
"""
import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi

from common import (
    BASE_MODEL_ID,
    get_default_output_dir,
    get_default_repo_id,
    ensure_hf_token,
    setup_logging,
)


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for model upload.
    
    Returns:
        argparse.Namespace: Parsed arguments including source, local directory,
                          repo ID, privacy setting, and commit message.
    """
    parser = argparse.ArgumentParser(description="Push a local model folder to Hugging Face Hub.")
    parser.add_argument(
        "--source",
        choices=["gptq8", "awq4", "tensorized", "custom"],
        default="gptq8",
        help="Quantization type (gptq8, awq4, tensorized) or custom (requires --local-dir and --repo-id).",
    )
    parser.add_argument("--model-id", default=BASE_MODEL_ID, help="Hugging Face source model ID (used to generate default paths/repo names).")
    parser.add_argument("--local-dir", help="Local folder to upload.")
    parser.add_argument("--repo-id", help="Destination Hugging Face model repo.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--commit-message", default="Upload quantized model artifacts")
    parser.add_argument("--log-level", default="INFO", help="Logging level for debugging.")
    return parser.parse_args()


def resolve_paths(args):
    """Resolve local directory and repository ID based on source type.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        tuple: (local_dir: Path, repo_id: str)
        
    Raises:
        ValueError: If custom source without both --local-dir and --repo-id.
    """
    if args.source in ["gptq8", "awq4", "tensorized"]:
        local_dir = Path(args.local_dir) if args.local_dir else get_default_output_dir(args.model_id, args.source)
        repo_id = args.repo_id or get_default_repo_id(args.model_id, args.source)
        return local_dir, repo_id
    if not args.local_dir or not args.repo_id:
        raise ValueError("--source custom requires both --local-dir and --repo-id.")
    return Path(args.local_dir), args.repo_id


def main():
    """Main entry point: upload model to Hugging Face Hub.
    
    Validates local model exists, creates repository if needed, and uploads all files.
    """
    args = parse_args()
    setup_logging(args.log_level)
    local_dir, repo_id = resolve_paths(args)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local model folder not found at {local_dir.resolve()}.")

    logger.info("Uploading model artifacts from %s to %s", local_dir.resolve(), repo_id)
    logger.debug("Repository privacy: %s", "private" if args.private else "public")
    logger.debug("Commit message: %s", args.commit_message)

    # Initialize Hugging Face API client with authentication token
    api = HfApi(token=ensure_hf_token())
    logger.info("Ensuring repository exists")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=args.private)
    logger.info("Uploading folder contents")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    logger.info("Pushed model to https://huggingface.co/%s", repo_id)


if __name__ == "__main__":
    main()
