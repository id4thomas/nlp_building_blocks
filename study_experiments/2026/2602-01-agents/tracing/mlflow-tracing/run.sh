#!/bin/bash

PORT=9041

# WARNING: allow all hosts only for dev purposes
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --host 0.0.0.0 \
    --port ${PORT} \
    --allowed-hosts "*" \
    --cors-allowed-origins "*"