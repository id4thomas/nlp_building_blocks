#!/bin/bash
# needs vllm>=0.7.0

source ../.env
MODEL_NAME="Qwen2-VL-2B"
echo "MODEL: ${model_dir}/${MODEL_NAME}"

vllm serve "${model_dir}/${MODEL_NAME}" \
        --served-model-name="${MODEL_NAME}" \
        --task=embed \
        --limit-mm-per-prompt image=4 \
        --chat-template="./chat_templates/qwen2-vl.jinja" \
        --chat-template-content-format=openai

# --gpu-memory-utilization=0.3 \