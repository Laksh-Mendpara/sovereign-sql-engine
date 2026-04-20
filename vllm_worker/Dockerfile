FROM nvidia/cuda:12.9.1-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.9/compat/

# Install vLLM with FlashInfer - use CUDA 12.8 PyTorch wheels (compatible with vLLM 0.15.1)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm[flashinfer]==0.16.0" --extra-index-url https://download.pytorch.org/whl/cu129



# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

ARG VLLM_NIGHTLY="false"
ARG MAX_MODEL_LEN=""
ARG ENFORCE_EAGER=""

<<<<<<< Updated upstream
ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    MAX_MODEL_LEN=$MAX_MODEL_LEN \
    ENFORCE_EAGER=$ENFORCE_EAGER \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
=======
ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
>>>>>>> Stashed changes
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
