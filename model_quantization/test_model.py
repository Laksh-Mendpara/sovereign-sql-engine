"""Test an Arctic Text2SQL model with a standard evaluation prompt set.

This script loads a quantized or tensorized Arctic model and evaluates it on
20 diverse SQL generation tasks, extracting and displaying the generated SQL queries.
"""
import argparse
import os
import re
from pathlib import Path

from common import DEFAULT_GPTQ8_DIR, DEFAULT_GPTQ8_REPO_ID, load_hf_token
from prompts import QUESTIONS, SYSTEM_PROMPT, build_user_prompt


def parse_args():
    """Parse command-line arguments for model testing.
    
    Returns:
        argparse.Namespace: Arguments for model path, quantization type, tensor settings,
                          generation parameters, and optional question subset.
    """
    parser = argparse.ArgumentParser(description="Test an Arctic model with the standard Text2SQL prompt set.")
    parser.add_argument("--model", default=DEFAULT_GPTQ8_REPO_ID, help="Local path or Hugging Face repo.")
    parser.add_argument("--tokenizer", help="Tokenizer path or repo. Defaults to --model.")
    parser.add_argument("--from-hub", action="store_true", help="Download the model repo to a local cache first.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_GPTQ8_DIR.parent / "arctic-test-cache"))
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--quantization", default="gptq_marlin")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--question-index", type=int, help="1-based question index to run instead of all questions.")
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def resolve_model_source(args, snapshot_download):
    """Resolve model and tokenizer sources, downloading from Hub if specified.
    
    Args:
        args: Parsed command-line arguments.
        snapshot_download: Hugging Face snapshot_download function.
        
    Returns:
        tuple: (model_source: str, tokenizer_source: str) - paths or repo IDs.
    """
    model_source = args.model
    tokenizer_source = args.tokenizer or args.model
    if not args.from_hub:
        return model_source, tokenizer_source

    # Download model from Hub to local cache
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = load_hf_token()
    model_source = snapshot_download(
        repo_id=args.model,
        local_dir=str(cache_dir),
        token=token,
    )
    tokenizer_source = model_source if args.tokenizer is None else args.tokenizer
    return model_source, tokenizer_source


def extract_sql(text: str) -> str:
    """Extract SQL code block from model output.
    
    Searches for ```sql...``` code block and extracts the SQL content,
    or returns original text if no SQL block found.
    
    Args:
        text (str): Model output text.
        
    Returns:
        str: Extracted SQL code block or original text.
    """
    match = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return f"```sql\n{match.group(1).strip()}\n```"
    return text


def main():
    """Main entry point: load model and run SQL generation evaluation.
    
    Loads the specified Arctic model, formats test questions with system prompt,
    generates SQL queries using vLLM, and displays results with extracted SQL.
    """
    args = parse_args()
    # Set GPU device visibility if specified
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Import inference libraries
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Resolve model path, downloading from Hub if needed
    model_source, tokenizer_source = resolve_model_source(args, snapshot_download)
    # Load tokenizer with Mistral regex fix
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        fix_mistral_regex=True,
        trust_remote_code=True,
    )

    # Select all or specific test question(s)
    selected_questions = QUESTIONS
    if args.question_index:
        if args.question_index < 1 or args.question_index > len(QUESTIONS):
            raise ValueError(f"--question-index must be between 1 and {len(QUESTIONS)}.")
        selected_questions = [QUESTIONS[args.question_index - 1]]

    # Format selected questions with system prompt and schema
    prompts = []
    for question in selected_questions:
        prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question)},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    # Configure sampling with greedy decoding (temperature=0)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens, n=1)
    # Initialize vLLM for optimized inference
    llm = LLM(
        model=model_source,
        tokenizer=tokenizer_source,
        load_format=args.load_format,
        quantization=args.quantization,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        hf_token=load_hf_token(),
        trust_remote_code=True,
    )

    # Generate SQL queries for all prompts
    outputs = llm.generate(prompts, sampling_params)
    # Display results with extracted SQL code
    for index, output in enumerate(outputs, start=1):
        print(f"Question {index}:")
        print(selected_questions[index - 1])
        print(extract_sql(output.outputs[0].text.strip()))
        print()


if __name__ == "__main__":
    main()
