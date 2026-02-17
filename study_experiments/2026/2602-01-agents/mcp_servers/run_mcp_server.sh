#!/bin/bash

export HOST=0.0.0.0
export CLIENT_PORT=9074
export SERVER_PORT=9077

# DNS Rebinding Protection
export ALLOWED_ORIGINS=http://YOURTESTIPHERE${CLIENT_PORT}

npx @modelcontextprotocol/inspector node build/index.js