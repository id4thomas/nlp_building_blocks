#!/bin/bash
source .env
echo "User: ${POSTGRES_USER}"
echo "DB Name: ${POSTGRES_DB}"
echo "ENV: ${APP_ENV}"

POSTGRES_VERSION="16"

docker run \
    --name pgvector-llamaindex-db \
    -e POSTGRES_USER=${POSTGRES_USER:-langchain} \
    -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-langchain} \
    -e POSTGRES_DB=${POSTGRES_DB:-muhayu} \
    -p ${POSTGRES_PORT:-6024}:5432 \
    -v ./local_storage:/var/lib/postgresql/data \
    pgvector/pgvector:pg${POSTGRES_VERSION}