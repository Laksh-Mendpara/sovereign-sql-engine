"""Tensorize a GPTQ-8 quantized Arctic model for vLLM inference.

This script converts a GPTQ-8 quantized model into vLLM's tensorized format,
which precompiles the model graph for optimized inference performance.
"""
import argparse
import os
import shutil
from pathlib import Path

from common import DEFAULT_GPTQ8_DIR, DEFAULT_GPTQ8_REPO_ID, DEFAULT_TENSORIZED_DIR


def parse_args():
    """Parse command-line arguments for tensorization.
    
    Returns:
        argparse.Namespace: Arguments for source/output directories, model name,
                          GPU memory settings, sequence length, and GPU devices.
    """
    parser = argparse.ArgumentParser(description="Tensorize a local GPTQ-8 Arctic model.")
    parser.add_argument("--source-dir", default=str(DEFAULT_GPTQ8_DIR))
    parser.add_argument("--served-model-name", default=DEFAULT_GPTQ8_REPO_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_TENSORIZED_DIR))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def copy_model_artifacts(source_dir: Path, target_dir: Path):
    """Copy non-tensor model artifacts from source to target directory.
    
    Selectively copies model files excluding tensorizer-specific and weight files
    (model-* prefixed, .safetensors, pytorch_model.bin) which will be regenerated.
    
    Args:
        source_dir (Path): Source model directory.
        target_dir (Path): Target directory for artifacts.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.name.startswith("model-"):
            continue
        if path.name == "model.tensors":
            continue
        if path.suffix == ".safetensors":
            continue
        if path.name.startswith("pytorch_model") and path.suffix == ".bin":
            continue
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)


def main():
    """Main entry point: tensorize GPTQ model for vLLM.
    
    Loads GPTQ model, creates vLLM engine configuration, tensorizes the model,
    and copies supporting artifacts to the output directory.
    """
    args = parse_args()
    # Set GPU device visibility if specified
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Import vLLM tensorizer components
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.tensorizer import (
        TensorizerConfig,
        tensorize_vllm_model,
    )

    # Validate source model exists
    source_dir = Path(args.source_dir)
    target_dir = Path(args.output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"GPTQ source model not found at {source_dir.resolve()}.")

    # Configure vLLM engine with GPTQ quantization
    engine_args = EngineArgs(
        model=str(source_dir),
        tokenizer=str(source_dir),
        served_model_name=args.served_model_name,
        quantization="gptq_marlin",
        dtype="float16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # Configure and run tensorization
    tensorizer_config = TensorizerConfig(tensorizer_dir=str(target_dir))
    tensorize_vllm_model(
        engine_args=engine_args,
        tensorizer_config=tensorizer_config,
        generate_keyfile=False,
    )
    # Copy tokenizer and config files from source
    copy_model_artifacts(source_dir, target_dir)

    print(f"Tensorized GPTQ model saved to {target_dir.resolve()}")


if __name__ == "__main__":
    main()
