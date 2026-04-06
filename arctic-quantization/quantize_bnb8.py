"""Quantize Arctic model using BitsAndBytes 8-bit quantization.

This script loads the Arctic model and quantizes it to 8-bit precision using the
BitsAndBytes library, reducing memory requirements while maintaining performance.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import BASE_MODEL_ID, DEFAULT_Q8_DIR


def parse_args():
    """Parse command-line arguments for BitsAndBytes quantization.
    
    Returns:
        argparse.Namespace: Arguments including model-id and output-dir.
    """
    parser = argparse.ArgumentParser(description="Save Arctic in bitsandbytes 8-bit format.")
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_Q8_DIR))
    return parser.parse_args()


def main():
    """Main entry point: quantize and save Arctic model in 8-bit format.
    
    Loads the model with BitsAndBytes 8-bit quantization config and saves
    both model and tokenizer to the specified output directory.
    """
    args = parse_args()
    # Configure BitsAndBytes for 8-bit quantization
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer and model with 8-bit quantization
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Save quantized model and tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print(f"Saved Arctic bitsandbytes Q8 model to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
