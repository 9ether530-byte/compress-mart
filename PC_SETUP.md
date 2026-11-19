# Compress-Mart — PC / laptop setup

This repo is the complete Compress-Mart local core through Phase 9
(audit fixes, 2026-11-10), ready to clone onto a real PC or laptop. The sandbox work is done;
the PC is where the remaining "not proven" items get proven, because
it has real hardware, a normal network, and a GPU once the graphics
card is replaced.

## Quickstart (any PC, CPU only)

```sh
git clone https://github.com/9ether530-byte/compress-mart.git
cd compress-mart
python3 -m venv .venv && . .venv/bin/activate  # or: python -m venv .venv
pip install -r requirements.txt
./scripts/run.sh
# API on http://127.0.0.1:8020, docs at /docs
python3 -m pytest -q
```

Optional, for real `.bin` quantization:

```sh
pip install -r requirements-torch.txt
# CPU torch only; on a GPU PC, install the CUDA wheel from pytorch.org instead.
```

Optional external services (all local, all optional):

```sh
docker compose -f infra/docker/docker-compose.yml up -d
# API 8020, Meilisearch 7700, Postgres 5432
# Then, with the API running:
curl -X POST http://127.0.0.1:8020/postgres/sync
curl -X POST http://127.0.0.1:8020/search/index
curl http://127.0.0.1:8020/external/status
```

## What is already proven here

- FastAPI service, 10 targets + aliases, router policy, certification
  state machine with LICENSE_UNVERIFIED publish gate, Ed25519
  provenance with fail-closed pull, benchmark cards, license receipts,
  `cm` CLI, storefront, deploy snippets, benchmark trend.
- REAL quantization: F32/F16/BF16 safetensors INT8, .npz INT8,
  and .bin INT8 with torch (host and in the torch container).
- Persistent router policy, benchmark → router feedback loop,
  reverify quarantine, Meilisearch search (with local fallback),
  measured host dequant time, Postgres replica sync.
- Tests: 28 passed + 1 live test skipped by default in Phase 8;
  Phase 9 adds regression tests for the audit fixes (see
  `tests/test_phase9.py`). Run the live whisper test on the PC with:
  `COMPRESS_MART_LIVE=1 python3 -m pytest -q tests/test_phase3.py::test_live_whisper_tiny_safetensors`

## What the PC still needs to prove (in order)

1. **Real hardware latency** — run the API on the PC, then on a Pi
   or phone if available, and replace the emulated p95 numbers with
   measured ones. The benchmark schema carries
   `measured_host_dequant_ms` as an optional field (Phase 9 added it
   to `schemas/benchmark_card.json`); target latency is the next field.
2. **MinIO live** — the S3 client is built (stdlib SigV4) but the
   MinIO image pull was registry-denied in the sandbox. On the PC,
   `docker pull docker.io/minio/minio:latest` should work; then
   `POST /storage/sync` against it and check `/external/status`.
3. **Kubernetes cluster run** — manifests are YAML-validated only.
   On the PC, `kind` or a local cluster can actually apply
   `infra/k8s/deployment.yaml` (API + Meilisearch + Postgres services;
   build the API image first with `scripts/build-container-phase8.sh`).
4. **Kubeflow execution** — `pipelines/kfp/compression-pipeline.yaml`
   is declarative; running it needs a Kubeflow cluster.
5. **Billing** — not built. No Stripe or other keys are in this repo
   and none should ever be pasted into chat; add a payment provider
   as a separate, deliberate step with keys in environment variables
   only.

## Security notes

- The signing key is generated on first run in `$COMPRESS_MART_DATA/keys`
  and is never printed or served. Do not commit `data/`.
- `docker-compose.yml`, `postgres.py` and `storage.py` take all
  passwords from the environment and refuse to run without them —
  there are no default passwords in this repo. Set
  `COMPRESS_MART_PG_PASSWORD` and `COMPRESS_MART_S3_SECRET_KEY` in
  `infra/docker/.env` (copy `.env.example`) for Compose, or in the
  `compress-mart-secrets` Secret for Kubernetes.
- `COMPRESS_MART_API_TOKEN`, when set, gates all writes and pull
  with header `X-API-Token`. Set it for any shared deployment.
