FROM lmcache/vllm-openai:v0.4.3-lightweight

# Install additional Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

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

COPY src /src

# Reset the base image's ENTRYPOINT (vllm CLI) so our handler runs directly
ENTRYPOINT []
CMD ["python3", "/src/handler.py"]
