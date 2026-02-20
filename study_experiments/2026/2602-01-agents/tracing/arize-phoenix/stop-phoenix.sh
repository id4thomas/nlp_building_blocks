#!/bin/bash

docker compose -f docker-compose.yml down

# REMOVE DATA (ONLY FOR TESTING)
docker volume rm arize-phoenix_phoenix_data