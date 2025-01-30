#!/bin/bash

source .env
KTEM_PORT=7860

rm -rf local_storage/kotaemon
mkdir -p local_storage/kotaemon

docker run \
-e GRADIO_SERVER_NAME=0.0.0.0 \
-e GRADIO_SERVER_PORT=7860 \
-v ./ktm.env:/app/.env \
-v ./local_storage/kotaemon:/app/ktem_app_data \
-p ${KTEM_PORT}:7860 -it --rm \
ghcr.io/cinnamon/kotaemon:main-lite