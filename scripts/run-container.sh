#!/bin/sh
# Run the built Compress-Mart container on host networking (the sandbox's
# known-good network mode), on port 8020 so it does not clash with Jarvis
# on 8000. Data lives in ./container-data on the host.
set -e
cd "$(dirname "$0")/.."
mkdir -p container-data
podman-sandbox run --network=host --rm \
  -v "$PWD/container-data:/data" \
  -e COMPRESS_MART_DATA=/data \
  compress-mart:phase5 \
  uvicorn compress_mart.app:app --host 127.0.0.1 --port 8020
