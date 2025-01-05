#!/bin/bash
# Get Base Dir
source .env
echo "User: ${POSTGRES_USER}"
echo "DB Name: ${POSTGRES_DB}"
echo "ENV: ${APP_ENV}"

rm -rf local_storage
mkdir local_storage