#!/bin/bash
VLLM_VERSION="v0.8.4"

MODEL_NAME="emotion-predictor"
BASE_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

RUN_NAME="250421-01-qwen2_5-3b-mini-try2"
ADAPTER_DIR="./weights/${RUN_NAME}/best"

docker container rm -f vllm_serving
docker run --runtime nvidia --gpus all \
        --name vllm_serving \
        -v ${ADAPTER_DIR}:/vllm-workspace/adapter \
        -v ./cache:/root/.cache/huggingface \
        -p 8010:8000 \
        --ipc=host \
        vllm/vllm-openai:${VLLM_VERSION} \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        --lora-modules ${MODEL_NAME}=/vllm-workspace/adapter \
        --enable-lora \
        --max-lora-rank 16 \
        --served-model-name ${BASE_MODEL_NAME} \
        --gpu-memory-utilization=0.5