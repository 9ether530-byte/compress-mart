#!/bin/sh
# Build the Compress-Mart container image in this sandbox, following the
# podman-sandbox factory pattern: proxy env vars and the egress CA bundle
# are passed as build args / a volume only when present on the host.
# No credential values are written into this script or the Dockerfile.
set -e
cd "$(dirname "$0")/.."
BUILD_ARGS=""
for var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
  if [ -n "$(eval echo \$$var)" ]; then BUILD_ARGS="$BUILD_ARGS --build-arg $var"; fi
done
VOLUME_ARGS=""
CA_FILE="/run/hatch/egress-tls/ca-bundle.pem"
if [ -f "$CA_FILE" ]; then
  VOLUME_ARGS="-v $CA_FILE:$CA_FILE:ro"
  BUILD_ARGS="$BUILD_ARGS --build-arg PIP_CERT=$CA_FILE --build-arg SSL_CERT_FILE=$CA_FILE --build-arg REQUESTS_CA_BUNDLE=$CA_FILE"
fi
# shellcheck disable=SC2086
podman-sandbox build --network=host $BUILD_ARGS $VOLUME_ARGS \
  -t compress-mart:phase5 -f infra/docker/Dockerfile .
echo "Built compress-mart:phase5"
