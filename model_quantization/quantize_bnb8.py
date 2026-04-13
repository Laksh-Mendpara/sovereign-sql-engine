"""Quantize Arctic model using BitsAndBytes 8-bit quantization.

This script loads the Arctic model and quantizes it to 8-bit precision using the
BitsAndBytes library, reducing memory requirements while maintaining performance.
"""
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import BASE_MODEL_ID, DEFAULT_Q8_DIR, setup_logging


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for BitsAndBytes quantization.
    
    Returns:
        argparse.Namespace: Arguments including model-id and output-dir.
    """
    parser = argparse.ArgumentParser(description="Save Arctic in bitsandbytes 8-bit format.")
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_Q8_DIR))
    parser.add_argument("--log-level", default="INFO", help="Logging level for debugging.")
    return parser.parse_args()


def main():
    """Main entry point: quantize and save Arctic model in 8-bit format.
    
    Loads the model with BitsAndBytes 8-bit quantization config and saves
    both model and tokenizer to the specified output directory.
    """
    args = parse_args()
    setup_logging(args.log_level)
    logger.info("Starting BitsAndBytes quantization for model %s", args.model_id)
    logger.debug("Output directory: %s", args.output_dir)

    # Configure BitsAndBytes for 8-bit quantization
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer and model with 8-bit quantization
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    logger.info("Loading model with 8-bit quantization")
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
    logger.info("Saving tokenizer and model to %s", output_dir.resolve())
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    logger.info("Saved Arctic bitsandbytes Q8 model to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
