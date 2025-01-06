#!/bin/bash
MILVUS_PORT=19530
MILVUS_WEBUI_PORT=9091

docker run -d \
        --name milvus-standalone \
        --security-opt seccomp:unconfined \
        -e ETCD_USE_EMBED=true \
        -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
        -e COMMON_STORAGETYPE=local \
        -v ./local_storage:/var/lib/milvus \
        -v ./embedEtcd.yaml:/milvus/configs/embedEtcd.yaml \
        -v ./user.yaml:/milvus/configs/user.yaml \
        -p ${MILVUS_PORT}:19530 \
        -p ${MILVUS_WEBUI_PORT}:9091 \
        -p 2379:2379 \
        --health-cmd="curl -f http://localhost:9091/healthz" \
        --health-interval=30s \
        --health-start-period=90s \
        --health-timeout=20s \
        --health-retries=3 \
        milvusdb/milvus:v2.5.2 \
        milvus run standalone  1