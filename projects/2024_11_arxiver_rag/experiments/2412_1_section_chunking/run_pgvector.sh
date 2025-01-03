#!/bin/bash

PGVECTOR_PORT=6024

docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p $PGVECTOR_PORT:5432 \
    -v ./pgvector_data:/var/lib/postgresql/data \
    -d pgvector/pgvector:pg16