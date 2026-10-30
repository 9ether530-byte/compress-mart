#!/bin/sh
# Run the Phase 1 API locally. The signing key is created on first run in
# $COMPRESS_MART_DATA/keys (default: ./data/keys) and is never printed.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export COMPRESS_MART_DATA="${COMPRESS_MART_DATA:-$PWD/data}"
exec uvicorn compress_mart.app:app --host 127.0.0.1 --port 8020
