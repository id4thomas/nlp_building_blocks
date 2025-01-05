#!/bin/bash
source .env
echo "User: ${POSTGRES_USER}"
echo "DB Name: ${POSTGRES_DB}"
echo "ENV: ${APP_ENV}"

POSTGRES_VERSION="16"

docker run \
  --name postgres-init \
  -e POSTGRES_USER=${POSTGRES_USER:-llamaindex} \
  -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-llamaindex} \
  -e POSTGRES_DB=${POSTGRES_DB:-pgvector_llamaindex} \
  -v ./local_storage:/var/lib/postgresql/data \
  -v ./db-initialization:/docker-entrypoint-initdb.d \
  -p ${POSTGRES_PORT:-6024}:5432 \
  pgvector/pgvector:pg${POSTGRES_VERSION}

# postgres:$POSTGRES_VERSION

docker container rm -f postgres-init