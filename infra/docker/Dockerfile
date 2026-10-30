FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# Sandbox egress: proxy + CA bundle are passed as build args by
# scripts/build-container.sh when present; on a plain box they are empty
# and pip works normally. They are never baked into the repo.
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG PIP_CERT
ARG SSL_CERT_FILE
ARG REQUESTS_CA_BUNDLE
RUN pip install --no-cache-dir -r requirements.txt
# Optional torch layer: only when INSTALL_TORCH=1 (scripts/build-container-torch.sh).
# Kept out of the slim image by default so it stays ~259 MB instead of ~1.1 GB.
ARG INSTALL_TORCH=0
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0+cpu"; \
    fi
COPY src ./src
COPY schemas ./schemas
COPY targets ./targets
COPY policies ./policies
COPY storefront ./storefront
ENV PYTHONPATH=/app/src COMPRESS_MART_DATA=/data
EXPOSE 8000
CMD ["uvicorn", "compress_mart.app:app", "--host", "0.0.0.0", "--port", "8000"]
