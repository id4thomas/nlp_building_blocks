#!/bin/bash

source .env
echo "${WANDB_ENTITY} - ${WANDB_PROJECT}"

CONFIG_NAME="250418-01-qwen2_5-3b-try1"
CONFIG_DIR="/Users/id4thomas/github/nlp_building_blocks/projects/2025_04_character/02-emotion-predictor/configs/${CONFIG_NAME}.json"

export WANDB_ENTITY=${WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_API_KEY=${WANDB_API_KEY}

cd src
python train.py --config_dir ${CONFIG_DIR}