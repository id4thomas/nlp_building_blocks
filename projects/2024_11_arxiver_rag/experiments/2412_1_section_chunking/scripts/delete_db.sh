#!/bin/bash
# Get Base Dir
CWD=`pwd`
BASE_DIR=$(dirname ${CWD})
echo $BASE_DIR

source $BASE_DIR/.env
echo "User: ${POSTGRES_USER}"
echo "DB Name: ${POSTGRES_DB}"
echo "ENV: ${APP_ENV}"

rm -rf ${LOCAL_STORAGE_DIR}
mkdir ${LOCAL_STORAGE_DIR}