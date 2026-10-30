#!/bin/sh
# Build the torch variant of the Compress-Mart image (compress-mart:phase6-torch).
# Same sandbox egress handling as build-container.sh, plus INSTALL_TORCH=1.
# No credential values are written into this script or the Dockerfile.
set -e
cd "$(dirname "$0")/.."
BUILD_ARGS="--build-arg INSTALL_TORCH=1"
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
  -t compress-mart:phase6-torch -f infra/docker/Dockerfile .
echo "Built compress-mart:phase6-torch"
