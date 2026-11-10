# Compress-Mart — Phase 9

A Model Compression Marketplace, built as a verified local core first.
Separate from VISION. No real hardware, no billing in this phase —
see the limits below, stated in the same breath as what works.

**Moving to a PC or laptop?** See [PC_SETUP.md](PC_SETUP.md) — quickstart,
what is already proven, and the five items the PC still needs to prove,
in order.

## What works (verified by tests + live smoke)

## Phase 9 additions (2026-11-10) — audit fixes

A full audit (2026-11-10) found the Phase 4 loop broken for real
artifacts and several honesty gaps. All are fixed here, with
regression tests in `tests/test_phase9.py`:

- Router feedback loop, real: feedback is matched on the tool that
  actually runs (`SafetensorsInt8/16`, `TorchInt8/16`, `NumPyInt8/16`),
  not only on the router's named tool, and a refinement to 16 bit now
  produces a real 16-bit artifact (INT16 safetensors / .pt / .npz),
  larger and more accurate than the 8-bit one. An explicitly
  requested bit width is never overridden by the loop.
- REAL bundles label themselves: the tar README and benchmark card
  say REAL with measured quality (target latency still emulated), and
  `benchmark.json` inside the tar now carries
  `measured_host_dequant_ms` / `measured_host_dequant_elements`
  (the tar is rewritten after measuring). The benchmark schema
  carries those fields as optional properties.
- Unsupported safetensors dtypes (e.g. F64) now fail loudly with
  400 instead of silently compressing as SIMULATED, and a header
  larger than 1 MB is read in full for detection. Data offsets are
  validated against shape and file size; truncated files are
  rejected, not quantized with wrong stats.
- `GET /benchmarks/trend?target=<unknown>` is 404, not 500.
  `POST /router/feedback/apply-auto` reports real-quantizer
  aggregates as skipped with the real reason, and refuses ambiguous
  policy matches instead of bumping the first rule that fits.
- Uploads are capped at 200 MB (413 above, same cap as HF downloads).
- `DEPRECATED` is reachable: `POST /models/{id}/deprecate` retires a
  CERTIFIED/PUBLISHED artifact (and the model, when its last artifact
  is retired). `POST /agent/query` is now token-gated like every
  other write. The signing key is written 0600-then-renamed, and a
  corrupt key is quarantined and replaced instead of crashing startup.
- K8s now reflects Phase 8/9: API image `compress-mart:phase8`
  (build it with `scripts/build-container-phase8.sh`), Postgres
  Deployment/Service with `COMPRESS_MART_PG_DSN`, S3 env vars, and
  Meilisearch `v1.17.1`. Compose now defines MinIO for real (plus
  S3 env on the API) and pins Meilisearch `v1.17.1`. The KFP DAG no
  longer skips the whole chain when `distill` is gated off, and its
  templates use the built `compress-mart:phase8` image.
- Docs corrected: the Run-it curls carry
  `-H 'Content-Type: application/json'` (they 422'd without it),
  the auth list is complete, and the Compose description is current.

## Phase 8 additions (2026-10-29)

- Live Postgres replica: a real PostgreSQL 16.15 (pgvector:pg16
  image) ran in Podman on host port 5432. `POST /postgres/sync`
  creates cm_models/cm_artifacts tables and replaces replica rows
  with the current SQLite catalog, then verifies by counting back.
  Live smoke: 1 model, 1 artifact synced and verified in Postgres
  (art-cc4945bda5ad, raspberry-pi-5, TinyPort, CERTIFIED).
  SQLite stays authoritative; Postgres is a replica, synced on
  demand, not a live primary — that limit is stated in the module.
- S3-compatible object storage client: stdlib-only SigV4 signing
  (no boto3) for make-bucket, PUT, GET, plus
  `POST /storage/sync` for CERTIFIED/PUBLISHED artifact tars.
  Proven in tests (signature structure, 502 when down) but NOT
  live-proven against MinIO here: the MinIO image pull is denied
  by the registry in this sandbox (docker.io and quay.io both
  refused, after several tags), so MinIO was stopped, not worked
  around with a fake server. Point COMPRESS_MART_S3_ENDPOINT at a
  real MinIO/S3 endpoint and the client works unchanged.
- Unified external status: `GET /external/status` reports
  Meilisearch, Postgres replica, and S3 storage availability plus
  the primary (SQLite). Live smoke showed exactly the truth:
  postgres true, search false (Meilisearch was stopped after
  Phase 7), storage false (no MinIO). Also as `cm external`.

## Phase 7 additions (2026-10-29)

- Live Meilisearch discovery: `POST /search/index` pushes the
  SQLite catalog into a real Meilisearch (1.17.1, run in Podman on
  host port 7700, `MEILI_NO_ANALYTICS=true`), and `GET /search?q=`
  returns ranked Meilisearch hits with full catalog entries in
  ranking order. If Meilisearch is down or unset, search falls back
  to local free-text and says so — never a 500 for a missing index.
  `GET /health` reports search availability. Compose now defines
  the meilisearch service for real; the K8s manifest adds a
  meilisearch Deployment + Service. Live smoke: index 1 artifact,
  search "whisper" returned 1 hit, 1 artifact, engine meilisearch.
- Measured host benchmark: REAL artifacts now carry
  `measured_host_dequant_ms` and `measured_host_dequant_elements`
  in the benchmark — the time to actually dequantize the payload
  on this host. Live smoke: an INT8 .npz artifact measured 1.87 ms
  for 256 elements. This is host time, NOT target (Pi/phone/laptop)
  latency, and it is labelled as such; it does not change the
  emulated target-latency gate. SIMULATED artifacts carry no
  measured field, correctly.
- K8s manifests refreshed: API image updated to phase7, search
  service added, and both manifests YAML-validated (4 K8s docs:
  2 Deployments, 2 Services). Not cluster-proven — no kubectl/kind
  is available in this sandbox, so the manifests are validated
  only, not applied, and the manifest says so. (Phase 9: image
  bumped to phase8 with Postgres wiring — see above.)

## Phase 6 additions (2026-10-29)

- Torch container variant: `Dockerfile` gained an `INSTALL_TORCH=1`
  build arg, and `scripts/build-container-torch.sh` builds
  `compress-mart:phase6-torch` (1.12 GB, vs 259 MB for the slim
  image — the torch trade-off, stated plainly). Inside that
  container, a real .bin upload compresses to REAL TorchInt8 with
  the same numbers as on the host (tiny-gpt2: 33 tensors, 31
  quantized, 2 copied, 2,514,146 → 2,311,519 bytes, ratio 0.9194,
  mean per-tensor rel. MSE 0.000025, CERTIFIED). The slim image is
  unchanged and still falls back to SIMULATED for .bin.
- Deploy snippets: `GET /models/{id}/deploy-snippet` (auto runtime
  by format, or `?runtime=` ollama/onnxruntime/transformers/torch/
  tflite/numpy/generic) generates a pull + load snippet from the
  artifact itself; SIMULATED artifacts carry a plain-text WARNING
  that they are evaluation only. Wired into the storefront and
  `cm deploy`. A brace-substitution bug in the torch/transformers
  templates (literal `{...}` code broke `.format()`) was found by
  the container smoke and fixed before reporting, with regression
  tests for every runtime.
- Benchmark trend: `GET /benchmarks/trend` (optional
  `?model_id=&target=`, target aliases resolved) returns the
  benchmark history oldest-first, the time series the router
  feedback aggregates are built from. Wired into `cm trend`.

## Phase 5 additions (2026-10-29)

- Persistent router policy: `POST /router/policy/apply` (rule +
  bit width) and `POST /router/feedback/apply-auto` write an override
  to the data dir, and a fresh app on the same data dir loads it —
  refinements now survive restarts instead of applying per-run only.
  Auto-apply only bumps unambiguous laptop rules (8 → 16); other
  categories are reported as skipped, not guessed. Real-quantizer
  aggregates (`SafetensorsInt8/16`, `TorchInt8/16`, `NumPyInt8/16`)
  have no policy rule and are reported as skipped with that reason
  (Phase 9 fix — they used to be silently dropped).
- F16/BF16 safetensors: the parser/quantizer now reads F16 and BF16
  tensors (converted to float32 for quantization, output still INT8),
  verified with synthetic mixed-dtype files.
- Container deployment, proven in this sandbox: the image builds
  with Podman (259 MB, via the podman-sandbox factory pattern for
  proxy + egress CA, plus the cgroupfs build flag) and runs on host
  networking on 8020, compressing a demo model to CERTIFIED.
  Scripts: `scripts/build-container.sh`, `scripts/run-container.sh`.
- Optional torch .bin quantization: torch 2.8.0+cpu is an optional
  dependency (`requirements-torch.txt`), not in the main image.
  With torch present, a real .bin state dict quantizes: float
  tensors → per-tensor INT8 with scales, non-float tensors (e.g.
  uint8 attention masks) copied unchanged. Verified live with
  sshleifer/tiny-gpt2 `pytorch_model.bin`: 33 tensors, 31 quantized,
  2 copied, 2,514,146 → 2,311,519 bytes (ratio 0.9194, because the
  2 MB of copied masks dominate; the float weights themselves shrink
  ~4x), mean per-tensor dequant relative MSE 0.000025, CERTIFIED,
  pulled and `cm verify` all OK. The mean (not sum-ratio) metric
  matters here: a single -10000 masked_bias tensor would otherwise
  dominate a sum ratio and hide real weight error. A 0-d tensor bug
  found by this live run (numpy scalar vs array) was fixed in
  `quantization.py` before reporting.

## Phase 4 additions (2026-10-29) — the synergetic loop

- Benchmark → router: every compress writes a router_feedback row.
  `GET /router/feedback` aggregates pass rate, quality delta, size
  ratio and latency per tool/target/bit-width, with a recommendation.
  When a tool/target has 3+ samples and under half passing, the next
  compress for that tool/target is automatically refined to the next
  recommended bit width (reported as `router_refined`). Phase 9
  fixed this for real artifacts: the match is on the tool that
  actually runs, and a refinement to 16 bit is a real 16-bit
  quantization, not a relabelled 8-bit payload. An explicitly
  requested bit width is never overridden.
- Provenance → catalog: `POST /maintenance/reverify` re-verifies
  every CERTIFIED/PUBLISHED artifact against the trusted key and
  quarantines any that now fail, so tampering after certification
  cannot leave a bad artifact listed. Verified in tests by tampering
  a stored tar, then reverify → quarantined → pull 403.
- Catalog → storefront: `GET /facets` (counts by target, task, tool,
  bit width, state, license) and `GET /models?q=` free-text search,
  both wired into the storefront and into `cm feedback` / `cm facets`.
- Auth: when `COMPRESS_MART_API_TOKEN` is set, ingest, compress,
  publish, deprecate, maintenance, router, search/index,
  postgres/sync, storage/sync, agent/query and pull require
  `X-API-Token` (401 otherwise); reads stay open, and `GET /health`
  reports `auth_enabled`. Without the env var, behaviour is unchanged.

## Phase 3 additions (2026-10-29)

- Real per-tensor INT8 quantization of F32 safetensors models, with
  no torch/transformers: a stdlib + numpy safetensors parser/writer,
  chunked so a 151 MB model is never held whole in memory (the
  quantized output is accumulated as it is produced; one float32
  chunk at a time is in memory). Scales live
  in the output header's __metadata__, so the artifact dequantizes
  from its own bytes. Verified live with openai/whisper-tiny
  `model.safetensors`: 151,061,672 bytes, apache-2.0 VERIFIED from its
  card, 167 tensors, 37,760,640 elements → 37,794,266 bytes (ratio
  0.2502), measured dequant relative MSE 0.00008, CERTIFIED, pulled
  (25 MB tar.gz) and `cm verify` all OK, now reporting
  `Compression mode REAL (SafetensorsInt8, safetensors)`.
- HF download limit raised from 20 MB to 200 MB so the whisper file
  fits; a failed download still returns 502, never a demo fallback.
- `cm verify` now prints the manifest's compression mode and tool,
  so REAL vs SIMULATED is visible at verification time, not just in
  the API response.

## Phase 2 additions (2026-10-29)

- Real Hugging Face ingest, on request: `POST /ingest/hub` with
  `download_file` fetches the model card (license, pipeline tag) and
  downloads that file from huggingface.co. Verified live with
  sshleifer/tiny-gpt2 `pytorch_model.bin`: 2,514,146 bytes,
  `artifact_source=huggingface-download`, task llm, LICENSE_UNVERIFIED
  (its card carries no license). A failed download returns 502, never a
  silent fallback to demo bytes. Without `download_file`, ingest keeps
  the Phase 1 demo behaviour.
- Real INT8 quantization for .npz weights: `POST /ingest/upload-weights`
  accepts a float32 `weights` array, and compress routes it to a numpy
  INT8 adapter (`compression_mode=REAL`, measured dequant relative MSE
  in the benchmark, payload is a real int8 .npz with a scale). HF
  .bin/.safetensors weights still cannot be parsed here (no
  torch/transformers) and stay SIMULATED, labelled as such — a real
  tiny-gpt2 download compressed through the pipeline is therefore
  still a simulated artifact, and the API says so.

## Phase 1 core

- Ingest from a model id (`POST /ingest/hub`) or a file upload
  (`POST /ingest/upload`). Phase 1 ingest does **not** download from
  Hugging Face; it records a deterministic local-demo original
  (`artifact_source=local-demo-seed`) so the pipeline is testable
  without network or a 150 MB download.
- Target matrix: all 10 prompt targets, plus aliases `rpi5`, `phone`,
  `laptop`. Every profile is marked `emulated=true`.
- Tool router (`policies/router_policy.yaml`): LLM+laptop → AngelSlim,
  vision+phone → AIMET, IoT → TinyPort, LLM+GPU → TurboQuant,
  general → ShrinkAI. The named upstream tools are **not installed**;
  each rule runs a deterministic local adapter labelled
  `compression_mode=SIMULATED` in every manifest.
- Benchmark engine: deterministic, emulated latency (p50/p95/p99),
  quality delta, peak memory and size ratio, plus a Markdown
  benchmark card with reproduction commands.
- Certification state machine:
  `INGESTED → COMPRESSED → BENCHMARKED → CERTIFIED → PUBLISHED → DEPRECATED`,
  plus `QUARANTINED`. Illegal transitions are rejected by the store.
- Provenance: Ed25519-signed bundles (SHA-256, license, upstream URL,
  compression manifest). Pull verifies against the trusted server key;
  a bad signature returns 403 and quarantines the artifact.
- License gate: `LICENSE_UNVERIFIED` artifacts cannot reach PUBLISHED
  without `approve_unverified=true`, which records the human legal
  review the prompt requires.
- Catalog: `GET /models` (filter by target, task, bit width, license,
  size, state), `GET /models/{id}`, `GET /certifications`,
  `GET /models/{id}/pull` (tar.gz bundle).
- Agent API: `POST /agent/query` by category, task, size and memory,
  returning only CERTIFIED/PUBLISHED artifacts. Full OpenAPI at `/docs`.
- Storefront at `/` and CLI `cm` (`search`, `pull`, `verify`, `targets`).
- Docker Compose + Kubernetes manifests and GitHub Actions workflows
  (schema validation gate, nightly pipeline test) are in the repo; the
  Compose file defines the API, Meilisearch and Postgres services
  (Phase 8; in Phase 1 it defined only the API service). Note: the two
  workflow files are in the local repo but are not on GitHub `main`
  yet — the GitHub App lacks the Workflows permission to push them,
  so add them from a PC clone or the web UI.

## What is NOT built or proven in Phase 9

- Live MinIO/S3 object storage (client built and tested, but the
  image pull is registry-denied in this sandbox; see above).
- Real Raspberry Pi / phone / laptop latency measurements (target
  latency is still emulated; only host dequant time is measured).
- Kubernetes cluster deployment (manifests YAML-validated only,
  not applied — no kubectl/kind here), Kubeflow execution, or billing.
  Postgres is a replica, not the primary store.

## Run it

```bash
pip install -r requirements.txt
./scripts/run.sh
# in another shell, from the repo root:
PYTHONPATH=src python3 -m compress_mart.cli --base-url http://127.0.0.1:8020 targets
curl -X POST localhost:8020/ingest/hub -H 'Content-Type: application/json' -d '{"model_id":"openai/whisper-tiny"}'
curl -X POST localhost:8020/compress -H 'Content-Type: application/json' -d '{"model_id":"openai/whisper-tiny","target":"rpi5","policy":"auto"}'
curl "localhost:8020/models?target=rpi5&task=asr"
```

Note the port collapse: the prompt's 8001/8002/8003 services are one
local service on 8020 in Phase 1, with all of the prompt's routes kept.

## Tests

```bash
python3 -m pytest -q
python3 scripts/validate_schemas.py
```
