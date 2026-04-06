"""Upload local Arctic quantized models to Hugging Face Hub.

This script facilitates uploading GPTQ-8 or tensorized models to the Hugging Face Hub
with proper authentication and optional privacy settings.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

from common import (
    DEFAULT_GPTQ8_DIR,
    DEFAULT_GPTQ8_REPO_ID,
    DEFAULT_TENSORIZED_DIR,
    DEFAULT_TENSORIZED_REPO_ID,
    ensure_hf_token,
)


def parse_args():
    """Parse command-line arguments for model upload.
    
    Returns:
        argparse.Namespace: Parsed arguments including source, local directory,
                          repo ID, privacy setting, and commit message.
    """
    parser = argparse.ArgumentParser(description="Push a local Arctic model folder to Hugging Face Hub.")
    parser.add_argument(
        "--source",
        choices=["gptq8", "tensorized", "custom"],
        default="gptq8",
        help="Pick a default local folder/repo pair or use --local-dir and --repo-id with custom.",
    )
    parser.add_argument("--local-dir", help="Local folder to upload.")
    parser.add_argument("--repo-id", help="Destination Hugging Face model repo.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--commit-message", default="Upload Arctic model artifacts")
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
    if args.source == "gptq8":
        return Path(args.local_dir or DEFAULT_GPTQ8_DIR), args.repo_id or DEFAULT_GPTQ8_REPO_ID
    if args.source == "tensorized":
        return Path(args.local_dir or DEFAULT_TENSORIZED_DIR), args.repo_id or DEFAULT_TENSORIZED_REPO_ID
    if not args.local_dir or not args.repo_id:
        raise ValueError("--source custom requires both --local-dir and --repo-id.")
    return Path(args.local_dir), args.repo_id


def main():
    """Main entry point: upload model to Hugging Face Hub.
    
    Validates local model exists, creates repository if needed, and uploads all files.
    """
    args = parse_args()
    local_dir, repo_id = resolve_paths(args)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local model folder not found at {local_dir.resolve()}.")

    # Initialize Hugging Face API client with authentication token
    api = HfApi(token=ensure_hf_token())
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print(f"Pushed model to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
