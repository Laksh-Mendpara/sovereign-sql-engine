FROM nvidia/cuda:12.9.1-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.9/compat/

# Install vLLM with FlashInfer - use CUDA 12.8 PyTorch wheels (compatible with vLLM 0.15.1)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm[flashinfer]==0.16.0" --extra-index-url https://download.pytorch.org/whl/cu129



# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

ARG VLLM_NIGHTLY="false"

ENV MODEL_NAME="ByteMaster01/arctic-text2sql-r1-7b-gptq8" \
    MODEL_REVISION="" \
    TOKENIZER_NAME="" \
    TOKENIZER_REVISION="" \
    BASE_PATH="/" \
    QUANTIZATION="gptq" \
    MAX_MODEL_LEN=4096 \
    ENFORCE_EAGER="true" \
    ENABLE_PREFIX_CACHING="true" \
    ENABLE_LMCACHE="true" \
    UPSTASH_REDIS_REST_URL="https://square-grubworm-103218.upstash.io" \
    UPSTASH_REDIS_REST_TOKEN="<your_upstash_token>" \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    HF_DATASETS_CACHE=/runpod-volume/huggingface-cache/datasets \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
    pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly && \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    pip install git+https://github.com/huggingface/transformers.git; \
fi

COPY src /src

# Start the handler
CMD ["python3", "/src/handler.py"]
