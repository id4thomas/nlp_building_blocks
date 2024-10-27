#!/bin/bash

VLLM_VERSION="v0.6.3.post1"
MODEL="Qwen/Qwen2-VL-7B-Instruct"

docker run --runtime nvidia --gpus all \
        --name vllm_serving \
        -v ./cache:/root/.cache/huggingface \
        -p 8010:8000 \
        --ipc=host \
        vllm/vllm-openai:${VLLM_VERSION} \
        --model ${MODEL}
