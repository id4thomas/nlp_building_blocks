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
CONTAINER_NAME="vllm-bench-${CONFIG_STEM}-${STAMP}"
IN_CONTAINER_RESULTS="/tmp/bench-results"

cleanup() {
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

set -o pipefail
docker run -i \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --entrypoint bash \
    -v "${SCRIPT_DIR}":/workspace:ro \
    -v "${HOST_TOKENIZER_DIR}":"${TOKENIZER_MOUNT}":ro \
    -w /workspace \
    -e "VLLM_BENCH_TOKENIZER_DIR=${TOKENIZER_MOUNT}" \
    -e "VLLM_BENCH_RESULTS_DIR=${IN_CONTAINER_RESULTS}" \
    -e HF_HUB_OFFLINE=1 \
    "${IMAGE}" \
    -c "mkdir -p ${IN_CONTAINER_RESULTS} && exec python3 bench_serve.py ${CONFIG}" 2>&1 | tee "${LOG_FILE}"
rc=${PIPESTATUS[0]}

docker cp "${CONTAINER_NAME}:${IN_CONTAINER_RESULTS}/." "${RESULTS_DIR}/" 2>/dev/null || \
    echo "[bench] warning: failed to copy results from container" >&2

echo "[bench] console log: ${LOG_FILE}"
exit "${rc}"
