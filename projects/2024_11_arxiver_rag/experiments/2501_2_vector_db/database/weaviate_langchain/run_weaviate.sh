#!/bin/bash
source .env
echo "PORT: ${WEAVIATE_PORT}"
echo "GRPC PORT: ${WEAVIATE_GRPC_PORT}"

WEAVIATE_VERSION="1.28.2"

# https://weaviate.io/developers/weaviate/installation/docker-compose#environment-variables
docker run \
    -e QUERY_DEFAULTS_LIMIT=25 \
    -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED='true' \
    -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
    -v ./local_storage:/var/lib/weaviate \
    -p ${WEAVIATE_PORT}:8080 \
    -p ${WEAVIATE_GRPC_PORT}:50051 \
    cr.weaviate.io/semitechnologies/weaviate:${WEAVIATE_VERSION}