#!/bin/bash

source .env
echo "${WANDB_ENTITY} - ${WANDB_PROJECT}"

CONFIG_NAME="250421-01-qwen2_5-3b-mini-try2"
CONFIG_DIR="${PWD}/configs/${CONFIG_NAME}.json"

export WANDB_ENTITY=${WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_API_KEY=${WANDB_API_KEY}

cd src
python train.py --config_dir ${CONFIG_DIR}