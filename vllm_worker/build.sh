#!/bin/bash
# Build the worker image without baking any model or runtime settings in.

docker build -t worker-vllm .
