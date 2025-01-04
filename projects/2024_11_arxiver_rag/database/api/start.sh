#!/bin/bash
CWD=`pwd`
BASE_DIR=$(dirname ${CWD})
echo $BASE_DIR

source $BASE_DIR/.env
echo "ENV: ${APP_ENV}"
echo "PORT: ${API_PORT}"


docker container rm -f arxiver-db-api
docker run -it \
	--name arxiver-db-api \
	-v $BASE_DIR/.env:/workdir/src/.env \
	-p ${API_PORT}:8000 \
	arxiver-db-api