"""Compress-Mart Phase 9 API — one local service, module boundaries intact.

Phase 9 is the 2026-11-10 audit-fix release: the router feedback loop
now actually refines REAL artifacts (real 16-bit quantization, not a
relabelled 8-bit payload), REAL bundles label themselves honestly,
unsupported safetensors dtypes fail loudly, and the measured host
timing is inside the delivered tar. See README Phase 9.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import deploy as deploy_mod, hf, host_bench as host_bench_mod, pipeline, postgres as postgres_mod, quantization, router as router_mod, safetensors as st_mod, search as search_mod, storage as storage_mod, targets as targets_mod, torch_bin as torch_bin_mod
from .provenance import load_or_create_keypair, sha256_bytes, build_provenance
from .store import Store, demo_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]

# Upload cap, same as the HF download cap (hf.py): a multi-GB upload
# must not be a trivial memory-exhaustion DoS. Exceeding it is 413.
MAX_UPLOAD_BYTES = 200_000_000


async def _read_upload_capped(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1_048_576)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_task(model_id: str) -> str:
    low = model_id.lower()
    if "whisper" in low:
        return "asr"
    if any(k in low for k in ("vit", "resnet", "clip", "vision")):
        return "vision"
    return "llm"


class IngestHubRequest(BaseModel):
    model_id: str
    task: str | None = None
    model_type: str | None = None
    license: str | None = None
    download_file: str | None = None


class CompressRequest(BaseModel):
    model_id: str
    target: str
    policy: str = "auto"
    bit_width: int | None = None


class PublishRequest(BaseModel):
    approve_unverified: bool = False


class PolicyApplyRequest(BaseModel):
    rule: str
    bit_width: int


class AgentQuery(BaseModel):
    category: str | None = None
    task: str | None = None
    max_size: int | None = None
    min_memory_mb: int | None = None


def catalog_entry(a: dict) -> dict:
    return {
        "id": a["id"], "model_id": a["upstream_id"], "task": a["task"],
        "target": a["target"], "tool": a["tool"], "bit_width": a["bit_width"],
        "format": a["format"], "license": a["license"],
        "license_status": a["license_status"], "size_bytes": a["size_bytes"],
        "state": a["state"], "benchmark": a["benchmark"],
    }


def create_app(data_dir: Path | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get(
        "COMPRESS_MART_DATA", REPO_ROOT / "data"))
    store = Store(data_dir)
    private, public_pem = load_or_create_keypair(data_dir / "keys")
    matrix = targets_mod.load_targets()
    # Effective policy: a data-dir override (written by /router/policy/apply)
    # wins over the repo policy, so refinements survive restarts.
    _data_policy = data_dir / "router_policy.yaml"
    policy = router_mod.load_policy(_data_policy if _data_policy.exists() else None)
    policy_source = "data_override" if _data_policy.exists() else "repo"

    app = FastAPI(title="Compress-Mart (Phase 9)",
                  description="Model Compression Marketplace — Phase 9: "
                  "audit fixes for the router feedback loop, REAL bundle "
                  "labelling, and measured host timing in the tar. "
                  "See README.md for limits.",
                  version="0.9.0")
    api_token = os.environ.get("COMPRESS_MART_API_TOKEN")

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        if api_token:
            path = request.url.path
            protected = (
                (request.method == "POST" and (
                    path.startswith("/ingest") or path == "/compress"
                    or path.endswith("/publish") or path.endswith("/deprecate")
                    or path.startswith("/maintenance")
                    or path.startswith("/router") or path == "/search/index"
                    or path == "/postgres/sync" or path == "/storage/sync"
                    or path == "/agent/query"))
                or (request.method == "GET" and path.endswith("/pull"))
            )
            if protected and request.headers.get("X-API-Token") != api_token:
                return JSONResponse({"detail": "missing or invalid X-API-Token"}, status_code=401)
        return await call_next(request)
    app.state.store = store
    app.state.public_pem = public_pem
    app.state.matrix = matrix

    def _save_model(*, upstream_id: str, task: str, model_type: str,
                    license_name, original: bytes, source: str) -> dict:
        model_id = "model-" + uuid.uuid4().hex[:12]
        orig_path = data_dir / "originals" / f"{model_id}.bin"
        orig_path.write_bytes(original)
        row = {
            "id": model_id, "upstream_id": upstream_id, "task": task,
            "model_type": model_type, "license": license_name,
            "license_status": "VERIFIED" if license_name else "LICENSE_UNVERIFIED",
            "sha256": sha256_bytes(original),
            "upstream_url": f"https://huggingface.co/{upstream_id}" if "/" in upstream_id else upstream_id,
            "size_bytes": len(original), "artifact_source": source,
            "state": "INGESTED", "created_at": _now(),
        }
        return store.insert_model(row)

    @app.get("/health")
    def health():
        return {"ok": True, "phase": "9-audit-fixes",
                "compression": "REAL for F32/F16/BF16 safetensors (8/16-bit), .npz weights (8/16-bit), and .bin with torch (8/16-bit); SIMULATED otherwise",
                "benchmarks": "EMULATED target latency, MEASURED dequant error + host dequant time for INT8/INT16",
                "search": search_mod.health(),
                "auth_enabled": bool(api_token),
                "targets": len(matrix["targets"])}

    @app.get("/public-key")
    def public_key():
        return {"public_key": public_pem, "algorithm": "Ed25519"}

    @app.get("/targets")
    def list_targets():
        return [targets_mod.hcp(t, matrix) for t in matrix["targets"]]

    @app.get("/targets/{target_id}/capabilities")
    def target_capabilities(target_id: str):
        try:
            return targets_mod.hcp(target_id, matrix)
        except KeyError:
            raise HTTPException(404, f"unknown target: {target_id}")

    @app.post("/ingest/hub", status_code=201)
    def ingest_hub(req: IngestHubRequest):
        if req.download_file:
            # Real path: card + file must both come from Hugging Face.
            # Any failure is a 502, never a silent fallback to demo bytes.
            try:
                card = hf.fetch_card(req.model_id)
                data = hf.download_file(req.model_id, req.download_file)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(502, f"Hugging Face ingest failed: {exc}")
            task = req.task or card["task"] or _infer_task(req.model_id)
            model_type = req.model_type or ("vision" if task == "vision" else "llm")
            license_name = req.license or card["license"]
            return _save_model(upstream_id=req.model_id, task=task, model_type=model_type,
                               license_name=license_name, original=data,
                               source="huggingface-download")
        # Demo path (unchanged from Phase 1): no HF download is attempted.
        task = req.task or _infer_task(req.model_id)
        model_type = req.model_type or ("vision" if task == "vision" else "llm")
        model = _save_model(upstream_id=req.model_id, task=task, model_type=model_type,
                            license_name=req.license, original=demo_bytes(req.model_id),
                            source="local-demo-seed")
        return model

    @app.post("/ingest/upload", status_code=201)
    async def ingest_upload(file: UploadFile = File(...), task: str = "llm",
                            model_type: str = "llm", license: str | None = None):
        data = await _read_upload_capped(file)
        model = _save_model(upstream_id=file.filename or "local-upload", task=task,
                            model_type=model_type, license_name=license,
                            original=data, source="local-upload")
        return model

    @app.post("/ingest/upload-weights", status_code=201)
    async def ingest_upload_weights(file: UploadFile = File(...), task: str = "llm",
                                    license: str | None = None):
        """Upload a real float32 weight tensor as .npz (exactly one
        `weights` array). This is the only ingest path whose compression
        is REAL in Phase 2."""
        data = await _read_upload_capped(file)
        try:
            quantization.load_npz_weights(data)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"invalid .npz weights file: {exc}")
        return _save_model(upstream_id=file.filename or "local-weights", task=task,
                           model_type="weights", license_name=license,
                           original=data, source="local-upload")

    @app.get("/ingest/{model_id}/provenance")
    def ingest_provenance(model_id: str):
        model = store.get_model(model_id)
        if not model:
            raise HTTPException(404, "model not found")
        arts = [a for a in store.list_artifacts() if a["model_id"] == model["id"]]
        return {"model": model, "provenance_bundles": [a["provenance"] for a in arts]}

    @app.post("/compress", status_code=201)
    def compress(req: CompressRequest):
        model = store.get_model(req.model_id)
        if not model:
            raise HTTPException(404, "model not found")
        try:
            target = targets_mod.resolve_target(req.target, matrix)
        except KeyError:
            raise HTTPException(404, f"unknown target: {req.target}")
        hcp = targets_mod.hcp(target, matrix)
        try:
            decision = router_mod.route(model_type=model["model_type"], category=hcp["category"],
                                        requested_bit_width=req.bit_width,
                                        recommended_bit_widths=hcp["recommended_bit_widths"],
                                        policy=policy)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        # What will actually run for this file? The feedback loop must
        # match on the tool that runs, not only on the router's named
        # tool: REAL paths run SafetensorsInt8 / TorchInt8 / NumPyInt8,
        # while the router decision names AngelSlim / TinyPort / ...
        # Detect the file type first (reading only the header probe),
        # then match feedback on the effective tool.
        orig_path = data_dir / "originals" / f"{model['id']}.bin"
        real_source = model["artifact_source"] in ("huggingface-download", "local-upload")
        probe = st_mod.read_probe(orig_path) if real_source else b""
        st_detect = st_mod.detect_safetensors(probe) if real_source else "not-safetensors"
        if st_detect == "unsupported":
            raise HTTPException(
                400, "safetensors file has an unsupported dtype "
                     "(only F32/F16/BF16 quantize here); refusing to "
                     "compress it as SIMULATED")
        is_st = st_detect == "ok"
        is_torch = (not is_st and real_source
                    and orig_path.stat().st_size < 100_000_000
                    and torch_bin_mod.is_torch_bin(orig_path))
        if is_st:
            effective_tool = f"SafetensorsInt{decision['bit_width']}"
        elif is_torch:
            effective_tool = f"TorchInt{decision['bit_width']}"
        elif model["model_type"] == "weights":
            effective_tool = f"NumPyInt{decision['bit_width']}"
        else:
            effective_tool = decision["tool"]
        # Synergetic loop, step 1: benchmark feedback refines the router.
        # If this effective tool/target/bit-width has 3+ samples and
        # under half passed, bump to the next recommended bit width for
        # this run — but never override a bit width the caller asked
        # for explicitly, and never claim a refinement that did not
        # happen (a REAL INT quantizer only supports 8 and 16 bit).
        router_refined = False
        if req.bit_width is None:
            for agg in store.feedback_aggregates():
                if (agg["tool"] == effective_tool and agg["target"] == target
                        and agg["bit_width"] == decision["bit_width"]
                        and agg["recommendation"] == "increase bit_width"):
                    higher = [b for b in sorted(hcp["recommended_bit_widths"])
                              if b > decision["bit_width"] and b in (8, 16)]
                    if higher:
                        decision = {**decision, "bit_width": higher[0]}
                        router_refined = True
                    break
        if effective_tool in ("SafetensorsInt8", "TorchInt8") and decision["bit_width"] not in (8, 16):
            raise HTTPException(
                400, f"real {effective_tool} quantization supports 8 or 16 bit, "
                     f"not {decision['bit_width']}")
        if is_st:
            # REAL path: quantize the actual HF model tensors, per tensor.
            # The 151 MB original is never loaded whole into memory here.
            artifact_id = "art-" + uuid.uuid4().hex[:12]
            quantized_path = data_dir / "artifacts" / f"{artifact_id}.safetensors"
            try:
                stats = st_mod.quantize_safetensors_file(
                    orig_path, quantized_path, bit_width=decision["bit_width"])
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"cannot quantize safetensors: {exc}")
            compressed = quantized_path.read_bytes()
            from .provenance import sha256_file
            _orig_sha = sha256_file(orig_path)
            import hashlib as _hashlib
            _repro = _hashlib.sha256(
                (_orig_sha + f":SafetensorsInt{decision['bit_width']}:"
                 f"{decision['bit_width']}:" + target).encode()).hexdigest()
            manifest = pipeline.build_manifest(
                artifact_id=artifact_id, model_id=model["upstream_id"], target=target,
                tool="SafetensorsInt8" if decision["bit_width"] == 8 else "SafetensorsInt16",
                adapter="safetensors_int8",
                bit_width=decision["bit_width"], fmt="safetensors",
                sparsity=decision["sparsity"], original=b"", compressed=compressed,
                calibration_hash=sha256_bytes(b"compress-mart-phase3-safetensors-int8-v1"),
                compression_mode="REAL",
                original_size_bytes=int(orig_path.stat().st_size),
                reproducibility_hash=_repro)
            benchmark = pipeline.build_benchmark(artifact_id=artifact_id, target=target,
                                                 category=hcp["category"], task=model["task"],
                                                 hcp=hcp, manifest=manifest)
            benchmark["quality_metric"] = "dequant_relative_mse"
            benchmark["quality_value"] = round(stats["relative_mse"], 6)
            benchmark["quality_delta"] = round(-stats["relative_mse"], 6)
            benchmark["tensor_count"] = stats["tensor_count"]
            benchmark["total_elements"] = stats["total_elements"]
            benchmark["passed"] = (
                benchmark["quality_delta"] >= hcp["min_quality_delta"]
                and benchmark["latency_ms"]["p95"] <= hcp["latency_ceiling_ms_p95"]
            )
        elif is_torch:
            # REAL path, torch optional: float tensors -> INT8/INT16, other
            # tensors copied unchanged.
            artifact_id = "art-" + uuid.uuid4().hex[:12]
            quantized_path = data_dir / "artifacts" / f"{artifact_id}.pt"
            try:
                stats = torch_bin_mod.quantize_torch_file(
                    orig_path, quantized_path, bit_width=decision["bit_width"])
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"cannot quantize torch file: {exc}")
            compressed = quantized_path.read_bytes()
            from .provenance import sha256_file as _sha_file
            import hashlib as _hashlib2
            _repro2 = _hashlib2.sha256(
                (_sha_file(orig_path) + f":TorchInt{decision['bit_width']}:"
                 f"{decision['bit_width']}:" + target).encode()).hexdigest()
            manifest = pipeline.build_manifest(
                artifact_id=artifact_id, model_id=model["upstream_id"], target=target,
                tool=f"TorchInt{decision['bit_width']}", adapter="torch_int8",
                bit_width=decision["bit_width"], fmt="pt",
                sparsity=decision["sparsity"], original=b"", compressed=compressed,
                calibration_hash=sha256_bytes(b"compress-mart-phase5-torch-int8-v1"),
                compression_mode="REAL",
                original_size_bytes=int(orig_path.stat().st_size),
                reproducibility_hash=_repro2)
            benchmark = pipeline.build_benchmark(artifact_id=artifact_id, target=target,
                                                 category=hcp["category"], task=model["task"],
                                                 hcp=hcp, manifest=manifest)
            benchmark["quality_metric"] = "dequant_relative_mse"
            benchmark["quality_value"] = round(stats["relative_mse"], 6)
            benchmark["quality_delta"] = round(-stats["relative_mse"], 6)
            benchmark["tensor_count"] = stats["tensor_count"]
            benchmark["quantized_tensor_count"] = stats["quantized_tensor_count"]
            benchmark["copied_tensor_count"] = stats["copied_tensor_count"]
            benchmark["passed"] = (
                benchmark["quality_delta"] >= hcp["min_quality_delta"]
                and benchmark["latency_ms"]["p95"] <= hcp["latency_ceiling_ms_p95"]
            )
        elif model["model_type"] == "weights":
            # REAL path: quantize the actual tensor and measure the error.
            original = orig_path.read_bytes()
            try:
                weights = quantization.load_npz_weights(original)
                q = quantization.quantize_int(weights, bit_width=decision["bit_width"])
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"cannot quantize weights: {exc}")
            compressed = quantization.pack_npz(
                q["quantized"], q["scale"], bit_width=decision["bit_width"])
            artifact_id = "art-" + uuid.uuid4().hex[:12]
            manifest = pipeline.build_manifest(
                artifact_id=artifact_id, model_id=model["upstream_id"], target=target,
                tool=f"NumPyInt{decision['bit_width']}", adapter=decision["adapter"],
                bit_width=decision["bit_width"], fmt=decision["format"],
                sparsity=decision["sparsity"], original=original, compressed=compressed,
                calibration_hash=sha256_bytes(b"compress-mart-phase2-numpy-int8-v1"),
                compression_mode="REAL")
            benchmark = pipeline.build_benchmark(artifact_id=artifact_id, target=target,
                                                 category=hcp["category"], task=model["task"],
                                                 hcp=hcp, manifest=manifest)
            # Override the emulated quality fields with the measured error.
            benchmark["quality_metric"] = "dequant_relative_mse"
            benchmark["quality_value"] = round(q["relative_mse"], 6)
            benchmark["quality_delta"] = round(-q["relative_mse"], 6)
            benchmark["dequant_mse"] = q["mse"]
            benchmark["dequant_max_abs_error"] = q["max_abs_error"]
            benchmark["int8_scale"] = q["scale"]
            benchmark["passed"] = (
                benchmark["quality_delta"] >= hcp["min_quality_delta"]
                and benchmark["latency_ms"]["p95"] <= hcp["latency_ceiling_ms_p95"]
            )
        else:
            original = orig_path.read_bytes()
            compressed = pipeline.compress_payload(original, decision["bit_width"], decision["sparsity"])
            artifact_id = "art-" + uuid.uuid4().hex[:12]
            manifest = pipeline.build_manifest(
                artifact_id=artifact_id, model_id=model["upstream_id"], target=target,
                tool=decision["tool"], adapter=decision["adapter"],
                bit_width=decision["bit_width"], fmt=decision["format"],
                sparsity=decision["sparsity"], original=original, compressed=compressed,
                calibration_hash=sha256_bytes(b"compress-mart-phase1-calibration-v1"))
            benchmark = pipeline.build_benchmark(artifact_id=artifact_id, target=target,
                                                 category=hcp["category"], task=model["task"],
                                                 hcp=hcp, manifest=manifest)
        provenance = build_provenance(
            model_id=model["upstream_id"], artifact_id=artifact_id,
            sha256=sha256_bytes(compressed), license_name=model["license"],
            license_status=model["license_status"], upstream_url=model["upstream_url"],
            compression_manifest=manifest, private=private, public_pem=public_pem)
        receipt = pipeline.license_receipt(artifact_id=artifact_id, model_id=model["upstream_id"],
                                           license_name=model["license"],
                                           license_status=model["license_status"])
        tar_path = data_dir / "artifacts" / f"{artifact_id}.tar.gz"
        pipeline.write_artifact_tar(tar_path=tar_path, payload=compressed, manifest=manifest,
                                    benchmark=benchmark,
                                    benchmark_md=pipeline.benchmark_card_markdown(benchmark, hcp),
                                    provenance=provenance, receipt=receipt)
        # Phase 7: first measured (not emulated) timing — host dequant
        # time for REAL artifacts only. It is informational: it does not
        # change the emulated target-latency gate above. The tar is
        # rewritten after measuring, so the delivered bundle's
        # benchmark.json and card actually carry the measured fields.
        if manifest.get("compression_mode") == "REAL":
            measured = host_bench_mod.measure_artifact_tar(tar_path, manifest["format"])
            if measured:
                benchmark.update(measured)
                pipeline.write_artifact_tar(
                    tar_path=tar_path, payload=compressed, manifest=manifest,
                    benchmark=benchmark,
                    benchmark_md=pipeline.benchmark_card_markdown(benchmark, hcp),
                    provenance=provenance, receipt=receipt)
        row = {"id": artifact_id, "model_id": model["id"], "target": target,
               "tool": manifest["tool"], "bit_width": manifest["bit_width"],
               "format": manifest["format"], "size_bytes": len(compressed),
               "sha256": sha256_bytes(compressed), "state": "INGESTED",
               "benchmark_json": json.dumps(benchmark), "manifest_json": json.dumps(manifest),
               "provenance_json": json.dumps(provenance), "file_path": str(tar_path),
               "created_at": _now()}
        try:
            store.insert_artifact(row)
            # Synergetic loop, step 2: this benchmark becomes router feedback.
            store.insert_feedback({
                "artifact_id": artifact_id, "tool": manifest["tool"], "target": target,
                "category": hcp["category"], "bit_width": manifest["bit_width"],
                "quality_delta": benchmark["quality_delta"],
                "size_ratio": benchmark["size_ratio"],
                "latency_p95": benchmark["latency_ms"]["p95"],
                "passed": benchmark["passed"], "created_at": _now()})
            store.update_artifact_state(artifact_id, "COMPRESSED")
            store.update_artifact_state(artifact_id, "BENCHMARKED")
            if benchmark["passed"]:
                store.update_artifact_state(artifact_id, "CERTIFIED")
        except Exception:
            # Roll back: no orphan INGESTED row and no orphan tar file
            # for a compress that failed after the insert.
            store.delete_artifact(artifact_id)
            try:
                tar_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        art = store.get_artifact(artifact_id)
        return {"artifact": catalog_entry({**art, "upstream_id": model["upstream_id"],
                                           "task": model["task"], "license": model["license"],
                                           "license_status": model["license_status"]}),
                "manifest": manifest, "benchmark": benchmark, "provenance": provenance,
                "rule": decision["rule"], "router_refined": router_refined}

    @app.get("/models")
    def list_models(target: str | None = None, task: str | None = None,
                    bit_width: int | None = None, license: str | None = None,
                    max_size: int | None = None, state: str | None = None,
                    q: str | None = None):
        # Accept the prompt's aliases (rpi5, phone, laptop) for target.
        resolved = None
        if target:
            try:
                resolved = targets_mod.resolve_target(target, matrix)
            except KeyError:
                raise HTTPException(404, f"unknown target: {target}")
        arts = store.list_artifacts(target=resolved, task=task, bit_width=bit_width,
                                    max_size=max_size, state=state)
        if license:
            arts = [a for a in arts if a["license"] == license]
        if q:
            needle = q.lower()
            arts = [a for a in arts if needle in " ".join(
                str(a[k]) for k in ("upstream_id", "tool", "target", "task")).lower()]
        return [catalog_entry(a) for a in arts]

    @app.get("/facets")
    def facets():
        return store.facets()

    @app.get("/router/feedback")
    def router_feedback():
        return store.feedback_aggregates()

    @app.get("/router/policy")
    def router_policy():
        return {"rules": policy, "feedback": store.feedback_aggregates(),
                "source": "data_override" if _data_policy.exists() else "repo"}

    @app.post("/router/policy/apply")
    def router_policy_apply(req: "PolicyApplyRequest"):
        try:
            result = router_mod.apply_bit_width(policy, req.rule, req.bit_width)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        router_mod.save_policy(policy, _data_policy)
        return {**result, "source": "data_override", "persisted": True}

    @app.post("/router/feedback/apply-auto")
    def router_feedback_apply_auto():
        """Persist the unambiguous feedback recommendations: laptop rules
        bump 8 -> 16 (the only next step in laptop recommended widths).
        Aggregates for the real quantizers (SafetensorsInt8/16,
        TorchInt8/16, NumPyInt8/16) match no router policy rule — they
        are reported as skipped with the real reason, not silently
        dropped and not guessed. A rule match is also ambiguous when
        two rules share tool+category with different model_types."""
        REAL_TOOLS = {"SafetensorsInt8", "SafetensorsInt16", "TorchInt8",
                      "TorchInt16", "NumPyInt8", "NumPyInt16"}
        applied, skipped = [], []
        for agg in store.feedback_aggregates():
            if agg["recommendation"] != "increase bit_width":
                continue
            if agg["tool"] in REAL_TOOLS:
                skipped.append({**agg, "reason": (
                    "real-quantizer aggregate (no router policy rule to bump; "
                    "the router loop refines the next compress directly)")})
                continue
            if agg["category"] != "laptop" or agg["bit_width"] != 8:
                skipped.append({**agg, "reason": "no unambiguous next bit width"})
                continue
            matches = [rule for rule in policy
                       if rule["tool"] == agg["tool"]
                       and rule["when"]["category"] == "laptop"
                       and rule["bit_width"] == 8]
            if len(matches) != 1:
                skipped.append({**agg, "reason": (
                    "ambiguous policy match: "
                    f"{len(matches)} rules match tool+category")})
                continue
            rule = matches[0]
            rule["bit_width"] = 16
            applied.append({"rule": rule["name"], "tool": rule["tool"],
                            "old_bit_width": 8, "new_bit_width": 16})
        if applied:
            router_mod.save_policy(policy, _data_policy)
        return {"applied": applied, "skipped": skipped, "persisted": bool(applied)}

    @app.post("/maintenance/reverify")
    def maintenance_reverify():
        """Re-verify every CERTIFIED/PUBLISHED artifact against the trusted
        key. Failures are quarantined, so a tampered artifact cannot stay
        in the catalog even if it was certified before the tampering."""
        checked = ok = 0
        quarantined = []
        for art in store.list_artifacts():
            if art["state"] not in ("CERTIFIED", "PUBLISHED"):
                continue
            checked += 1
            result = pipeline.verify_artifact_tar(Path(art["file_path"]),
                                                  trusted_public_pem=public_pem)
            if result["ok"]:
                ok += 1
            else:
                store.update_artifact_state(art["id"], "QUARANTINED", force=True)
                quarantined.append(art["id"])
        return {"checked": checked, "ok": ok, "quarantined": quarantined}

    @app.get("/models/{artifact_id}")
    def get_model_artifact(artifact_id: str):
        art = store.get_artifact(artifact_id)
        if not art:
            raise HTTPException(404, "artifact not found")
        model = store.get_model(art["model_id"])
        entry = catalog_entry({**art, "upstream_id": model["upstream_id"] if model else art["model_id"],
                               "task": model["task"] if model else None,
                               "license": model["license"] if model else None,
                               "license_status": model["license_status"] if model else None})
        entry["manifest"] = art["manifest"]
        entry["provenance"] = art["provenance"]
        entry["benchmark_card_url"] = f"/models/{artifact_id}/benchmark-card"
        return entry

    @app.get("/models/{artifact_id}/benchmark-card")
    def benchmark_card(artifact_id: str):
        art = store.get_artifact(artifact_id)
        if not art:
            raise HTTPException(404, "artifact not found")
        hcp = targets_mod.hcp(art["target"], matrix)
        return {"benchmark_card": pipeline.benchmark_card_markdown(art["benchmark"], hcp)}

    @app.get("/models/{artifact_id}/deploy-snippet")
    def deploy_snippet(artifact_id: str, runtime: str | None = None):
        art = store.get_artifact(artifact_id)
        if not art:
            raise HTTPException(404, "artifact not found")
        try:
            return deploy_mod.snippet(art, runtime)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/benchmarks/trend")
    def benchmark_trend(model_id: str | None = None, target: str | None = None):
        if target:
            try:
                target = targets_mod.resolve_target(target, matrix)
            except KeyError:
                raise HTTPException(404, f"unknown target: {target}")
        return store.trend(model_id=model_id, target=target)

    @app.post("/search/index")
    def search_index():
        """Push the current catalog into Meilisearch. 502, not 500,
        when Meilisearch is unreachable — the catalog is still fine."""
        try:
            return search_mod.index_artifacts(store.list_artifacts())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"meilisearch unreachable: {exc}")

    @app.get("/search")
    def search(q: str, limit: int = 20):
        """Ranked search via Meilisearch, falling back to the local
        free-text catalog search when it is down or unset."""
        try:
            result = search_mod.search(q, limit=limit)
            ids = {h["id"] for h in result["hits"]}
            arts = [a for a in store.list_artifacts() if a["id"] in ids]
            # Preserve Meilisearch ranking order.
            order = {h["id"]: i for i, h in enumerate(result["hits"])}
            arts.sort(key=lambda a: order.get(a["id"], 999))
            result["artifacts"] = [catalog_entry(a) for a in arts]
            return result
        except Exception:  # noqa: BLE001
            local = [catalog_entry(a) for a in store.list_artifacts()
                     if q.lower() in (a.get("upstream_id") or "").lower()
                     or q.lower() in (a.get("tool") or "").lower()]
            return {"hits": [], "artifacts": local[:limit],
                    "engine": "local-fallback",
                    "note": "meilisearch unavailable, local free-text search used"}

    @app.post("/postgres/sync")
    def postgres_sync():
        """Sync the SQLite catalog to the Postgres replica."""
        try:
            return postgres_mod.sync(store)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"postgres unreachable: {exc}")

    @app.post("/storage/sync")
    def storage_sync():
        """Upload CERTIFIED/PUBLISHED artifact tars to S3 storage."""
        try:
            result = storage_mod.sync_artifacts(store)
        except FileNotFoundError as exc:
            # A tar missing from disk is a server data problem, not a
            # storage outage: 500, not 502.
            raise HTTPException(500, f"artifact tar missing: {exc}")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"storage unreachable: {exc}")
        if not result["ok"]:
            raise HTTPException(502, f"storage sync failed: {result}")
        return result

    @app.get("/external/status")
    def external_status():
        return {
            "search_meilisearch": search_mod.health(),
            "postgres_replica": postgres_mod.status(),
            "storage_s3": storage_mod.status(),
            "primary": "sqlite (authoritative)",
        }

    @app.post("/models/{artifact_id}/publish")
    def publish(artifact_id: str, req: PublishRequest):
        art = store.get_artifact(artifact_id)
        if not art:
            raise HTTPException(404, "artifact not found")
        if art["state"] != "CERTIFIED":
            raise HTTPException(400, f"artifact is {art['state']}, not CERTIFIED")
        model = store.get_model(art["model_id"])
        if model and model["license_status"] == "LICENSE_UNVERIFIED" and not req.approve_unverified:
            raise HTTPException(403, "LICENSE_UNVERIFIED: human legal review required "
                                     "(pass approve_unverified=true to record that review)")
        store.update_artifact_state(artifact_id, "PUBLISHED")
        return {"artifact_id": artifact_id, "state": "PUBLISHED"}

    @app.post("/models/{artifact_id}/deprecate")
    def deprecate(artifact_id: str):
        """DEPRECATED is a real, reachable state now: CERTIFIED or
        PUBLISHED artifacts can be retired. If every artifact of the
        model is deprecated, the model itself is marked DEPRECATED."""
        art = store.get_artifact(artifact_id)
        if not art:
            raise HTTPException(404, "artifact not found")
        if art["state"] not in ("CERTIFIED", "PUBLISHED"):
            raise HTTPException(400, f"artifact is {art['state']}, not CERTIFIED or PUBLISHED")
        store.update_artifact_state(artifact_id, "DEPRECATED")
        remaining = [a for a in store.list_artifacts()
                     if a["model_id"] == art["model_id"]
                     and a["state"] not in ("DEPRECATED", "QUARANTINED")]
        if not remaining:
            store.update_model_state(art["model_id"], "DEPRECATED")
        return {"artifact_id": artifact_id, "state": "DEPRECATED"}

    @app.get("/certifications")
    def certifications(model_id: str | None = None):
        return store.certifications(model_id)

    @app.get("/models/{artifact_id}/pull")
    def pull(artifact_id: str):
        art = store.get_artifact(artifact_id)
        if not art:
            raise HTTPException(404, "artifact not found")
        if art["state"] not in ("CERTIFIED", "PUBLISHED"):
            raise HTTPException(403, f"artifact is {art['state']}, not certified for pull")
        result = pipeline.verify_artifact_tar(Path(art["file_path"]), trusted_public_pem=public_pem)
        if not result["ok"]:
            store.update_artifact_state(artifact_id, "QUARANTINED", force=True)
            raise HTTPException(403, f"provenance verification failed, artifact quarantined: {result['reason']}")
        return FileResponse(art["file_path"], media_type="application/gzip",
                            filename=f"{artifact_id}.tar.gz")

    @app.post("/agent/query")
    def agent_query(req: AgentQuery):
        arts = store.list_artifacts(state=None)
        out = []
        for a in arts:
            if a["state"] not in ("CERTIFIED", "PUBLISHED"):
                continue
            if req.task and a["task"] != req.task:
                continue
            if req.max_size and a["size_bytes"] > req.max_size:
                continue
            hcp = targets_mod.hcp(a["target"], matrix)
            if req.category and hcp["category"] != req.category:
                continue
            if req.min_memory_mb and hcp["memory_mb"] < req.min_memory_mb:
                continue
            out.append(catalog_entry(a))
        return out

    @app.get("/")
    def storefront():
        return FileResponse(REPO_ROOT / "storefront" / "index.html")

    @app.get("/app.js")
    def storefront_js():
        return FileResponse(REPO_ROOT / "storefront" / "app.js",
                            media_type="application/javascript")

    return app


app = create_app()
