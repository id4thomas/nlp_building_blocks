#!/bin/bash
# VLLM_VERSION="v0.19.0"
VLLM_VERSION="gemma4-cu130"

IMAGE="vllm/vllm-openai:${VLLM_VERSION}"

CONFIG="configs/${1:-example}.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENIZER_MOUNT="/mnt/tokenizer"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

HOST_TOKENIZER_DIR="$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['tokenizer'])" "${SCRIPT_DIR}/${CONFIG}")"

if [ ! -d "${HOST_TOKENIZER_DIR}" ]; then
    echo "tokenizer dir not found: ${HOST_TOKENIZER_DIR}" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
CONFIG_STEM="$(basename "${CONFIG}" .yaml)"
LOG_FILE="${RESULTS_DIR}/${CONFIG_STEM}_${STAMP}.log"

docker run --rm -i \
    --network host \
    --ipc host \
    --entrypoint python3 \
    -v "${SCRIPT_DIR}":/workspace \
    -v "${HOST_TOKENIZER_DIR}":"${TOKENIZER_MOUNT}":ro \
    -w /workspace \
    -e "VLLM_BENCH_TOKENIZER_DIR=${TOKENIZER_MOUNT}" \
    -e "VLLM_BENCH_RESULTS_DIR=/workspace/results" \
    -e HF_HUB_OFFLINE=1 \
    "${IMAGE}" \
    bench_serve.py "${CONFIG}" 2>&1 | tee "${LOG_FILE}"

echo "[bench] console log: ${LOG_FILE}"
