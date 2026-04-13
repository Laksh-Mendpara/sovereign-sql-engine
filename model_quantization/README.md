# NL2Data

Natural language to SQL conversion.

## arctic-quantization

Utilities for quantizing, tensorizing, publishing, and testing Arctic Text2SQL models.

### Setup

**1. Install dependencies (Python 3.10 required)**

```bash
cd arctic-quantization
pip install -e .
```

**2. Configure your Hugging Face token**

A valid Hugging Face token with **write** access is required to push models to the Hub.

```bash
cp arctic-quantization/.env.example arctic-quantization/.env
```

Open `arctic-quantization/.env` and replace `your_huggingface_token_here` with your actual token.  
You can generate a token at <https://huggingface.co/settings/tokens>.

The token is read automatically by all scripts via `common.load_hf_token()`, which checks the `HF_TOKEN` environment variable first and falls back to the `.env` file.

> **Note:** The `.env` file is listed in `.gitignore` and will never be committed.

### Scripts

| Script | Description |
|---|---|
| `quantize_bnb8.py` | BitsAndBytes 8-bit quantization |
| `quantize_gptq8.py` | GPTQ 8-bit quantization |
| `tensorize_gptq8.py` | Convert GPTQ-8 model to vLLM tensorized format |
| `test_model.py` | Run the standard prompt set against a local or Hub model |
| `push_to_hub.py` | Upload a local model folder to the Hugging Face Hub |

All quantization and upload scripts accept `--log-level` for more verbose debugging output, for example `--log-level DEBUG`.

### Pushing a model to the Hub

```bash
cd arctic-quantization
python push_to_hub.py --source gptq8
```

Use `--source tensorized` for the tensorized variant, or `--source custom --local-dir <path> --repo-id <org/repo>` for an arbitrary folder.
