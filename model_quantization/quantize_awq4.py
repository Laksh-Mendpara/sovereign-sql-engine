import argparse
import logging
import os
from pathlib import Path

import transformers
from transformers import AutoTokenizer

if not hasattr(transformers.activations, 'PytorchGELUTanh'):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

if not hasattr(transformers.utils, 'LossKwargs'):
    import typing
    class LossKwargs(typing.TypedDict, total=False):
        pass
    transformers.utils.LossKwargs = LossKwargs

from awq import AutoAWQForCausalLM

from common import BASE_MODEL_ID, get_default_output_dir, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for AWQ quantization."""
    parser = argparse.ArgumentParser(description="Quantize model to AWQ 4-bit.")
    parser.add_argument("--model-id", default=BASE_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to model-id-awq4 in models dir)")
    parser.add_argument("--dataset", default="pileval", help="Calibration dataset ('pileval', 'wikitext', etc.)")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--version", default="GEMM", help="AWQ version (GEMM or GEMV)")
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--log-level", default="INFO", help="Logging level for debugging.")
    return parser.parse_args()


def main():
    """Main entry point: quantize and save model in AWQ 4-bit format."""
    args = parse_args()
    setup_logging(args.log_level)
    
    if args.output_dir is None:
        args.output_dir = str(get_default_output_dir(args.model_id, "awq4"))
        
    logger.info("Starting AWQ quantization for model %s", args.model_id)
    logger.debug(
        "Quantization config: dataset=%s bits=%s group_size=%s version=%s output_dir=%s",
        args.dataset,
        args.bits,
        args.group_size,
        args.version,
        args.output_dir,
    )

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        logger.info("Using CUDA_VISIBLE_DEVICES=%s", args.cuda_visible_devices)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": args.version
    }

    logger.info("Loading model for AWQ quantization")
    model = AutoAWQForCausalLM.from_pretrained(args.model_id, **{"low_cpu_mem_usage": True})
    
    logger.info("Quantizing model using dataset: %s", args.dataset)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=args.dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving tokenizer and model to %s", output_dir.resolve())
    tokenizer.save_pretrained(output_dir)
    model.save_quantized(output_dir)

    logger.info("Saved AWQ-%s model to %s", args.bits, output_dir.resolve())


if __name__ == "__main__":
    main()
