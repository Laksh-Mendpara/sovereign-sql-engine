"""Quantize any Hugging Face causal-LM to 4-bit weights (W4A16) using llm-compressor.

Uses the GPTQ algorithm with group-size 128 to produce a compact model
that can be served efficiently with vLLM or loaded back with transformers.
"""
import argparse
import logging
import os
from pathlib import Path

from common import BASE_MODEL_ID, get_default_output_dir, setup_logging

# ── Compatibility shims for models with custom code ──────────────
# Phi-4 custom modeling code imports these from transformers.utils,
# but they may not exist in the pinned transformers version.
import typing
import transformers

if not hasattr(transformers.utils, "LossKwargs"):
    class _LossKwargs(typing.TypedDict, total=False):
        pass
    transformers.utils.LossKwargs = _LossKwargs

if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for W4A16 quantization."""
    parser = argparse.ArgumentParser(description="Quantize model to 4-bit (W4A16) using llm-compressor.")
    parser.add_argument("--model-id", default=BASE_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (defaults to <model-name>-awq4 in models dir)")
    parser.add_argument("--dataset", default="ultrachat",
                        help="Calibration dataset: 'ultrachat', 'wikitext', or 'pileval'")
    parser.add_argument("--num-samples", type=int, default=512,
                        help="Number of calibration samples")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length for calibration")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size")
    parser.add_argument("--cuda-visible-devices",
                        help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level for debugging.")
    return parser.parse_args()


def _prepare_dataset(dataset_name: str, tokenizer, num_samples: int, max_seq_length: int):
    """Load and tokenize a calibration dataset.

    Returns a HuggingFace Dataset with 'input_ids' and 'attention_mask' columns.
    """
    from datasets import load_dataset

    if dataset_name == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split=f"train_sft[:{num_samples}]")
        ds = ds.shuffle(seed=42)

        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            }
        ds = ds.map(preprocess)

    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 50)
        ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))


    elif dataset_name == "pileval":
        ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))


    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         "Choose 'ultrachat', 'wikitext', or 'pileval'.")

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


def main():
    """Main entry point: quantize and save model in W4A16 format."""
    args = parse_args()
    setup_logging(args.log_level)

    if args.output_dir is None:
        args.output_dir = str(get_default_output_dir(args.model_id, "awq4"))

    logger.info("Starting W4A16 quantization for model %s", args.model_id)

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        logger.info("Using CUDA_VISIBLE_DEVICES=%s", args.cuda_visible_devices)

    # ── 1. Load model & tokenizer ─────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    logger.info("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        trust_remote_code=True,
    )

    # ── 2. Prepare calibration data ───────────────────────────────
    logger.info("Preparing calibration dataset: %s (%d samples)", args.dataset, args.num_samples)
    ds = _prepare_dataset(args.dataset, tokenizer, args.num_samples, args.max_seq_length)

    # ── 3. Configure & run quantization ───────────────────────────
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )

    logger.info("Running oneshot quantization (this may take a while)...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
    )

    # ── 4. Save compressed model ──────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving tokenizer and model to %s", output_dir.resolve())
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    logger.info("✅ Saved W4A16 model to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
