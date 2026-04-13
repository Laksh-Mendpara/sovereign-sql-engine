#!/bin/bash
# Description: Build the vllm_worker docker image baking the tensorizer model

docker build -t worker-vllm-baked \
  --build-arg MODEL_NAME="ByteMaster01/arcitic-8q-tensorizer" \
  --build-arg BASE_PATH="/models" \
  --build-arg MAX_MODEL_LEN="4096" \
  --build-arg ENFORCE_EAGER="true" \
  .
