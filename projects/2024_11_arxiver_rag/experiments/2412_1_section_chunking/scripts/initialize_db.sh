#!/bin/bash
# Get Base Dir
CWD=`pwd`
BASE_DIR=$(dirname ${CWD})
echo $BASE_DIR

source $BASE_DIR/.env
echo "User: ${POSTGRES_USER}"
echo "DB Name: ${POSTGRES_DB}"
echo "ENV: ${APP_ENV}"

POSTGRES_VERSION="16"

POSTGRES_PORT=8010

docker run \
  --name postgres-init \
  -e POSTGRES_USER=${POSTGRES_USER:-langchain} \
  -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-langchain} \
  -e POSTGRES_DB=${POSTGRES_DB:-muhayu} \
  -v ${LOCAL_STORAGE_DIR}:/var/lib/postgresql/data \
  -v ${BASE_DIR}/scripts/db-initialization:/docker-entrypoint-initdb.d \
  -p ${POSTGRES_PORT}:5432 \
  postgres:$POSTGRES_VERSION

docker container rm -f postgres-init