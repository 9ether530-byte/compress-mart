#!/bin/sh
# Run the torch-variant Compress-Mart container on host networking
# (the sandbox's known-good network mode), on port 8020 so it does
# not clash with Jarvis on 8000. Data lives in ./container-data-torch
# on the host. Build it first with scripts/build-container-torch.sh.
set -e
cd "$(dirname "$0")/.."
mkdir -p container-data-torch
podman-sandbox run --network=host --rm \
  -v "$PWD/container-data-torch:/data" \
  -e COMPRESS_MART_DATA=/data \
  compress-mart:phase6-torch \
  uvicorn compress_mart.app:app --host 127.0.0.1 --port 8020
