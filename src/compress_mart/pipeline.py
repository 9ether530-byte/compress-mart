"""Compression adapters (SIMULATED in Phase 1) + deterministic benchmark.

Honesty contract: every manifest says compression_mode=SIMULATED and
every benchmark card says emulated=true. The payload is a deterministic
prefix of the original bytes at the size ratio implied by the bit width —
it is NOT a real quantized model and must never be presented as one.
A real adapter replaces `compress_payload` for one tool, not the pipeline.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path

from .provenance import sha256_bytes, verify_bundle

RATIO = {4: 0.15, 8: 0.30, 16: 0.55, 32: 1.0}
BASE_LATENCY_MS = {"laptop": 40.0, "phone": 30.0, "iot": 180.0, "gpu-server": 15.0}
QUALITY_METRIC = {
    "llm": ("perplexity", 12.4),
    "asr": ("wer", 0.082),
    "vision": ("top1_accuracy", 0.76),
    "text-classification": ("accuracy", 0.88),
    "embedding": ("recall@10", 0.81),
}


def compress_payload(original: bytes, bit_width: int, sparsity: float) -> bytes:
    ratio = RATIO.get(bit_width, 0.30)
    size = max(1, int(len(original) * ratio * (1 - sparsity * 0.5)))
    return original[:size]


def reproducibility_hash(original: bytes, tool: str, bit_width: int, target: str) -> str:
    return sha256_bytes(original + f"{tool}:{bit_width}:{target}".encode())


def build_manifest(*, artifact_id: str, model_id: str, target: str, tool: str,
                   adapter: str, bit_width: int, fmt: str, sparsity: float,
                   original: bytes, compressed: bytes, calibration_hash: str,
                   compression_mode: str = "SIMULATED",
                   original_size_bytes: int | None = None,
                   reproducibility_hash: str | None = None) -> dict:
    return {
        "artifact_id": artifact_id,
        "model_id": model_id,
        "target": target,
        "tool": tool,
        "tool_adapter": adapter,
        "bit_width": bit_width,
        "format": fmt,
        "sparsity": sparsity,
        "original_size_bytes": original_size_bytes if original_size_bytes is not None else len(original),
        "compressed_size_bytes": len(compressed),
        "calibration_dataset_hash": calibration_hash,
        "reproducibility_hash": reproducibility_hash or reproducibility_hash_from(original, tool, bit_width, target),
        "compression_mode": compression_mode,
    }


def reproducibility_hash_from(original: bytes, tool: str, bit_width: int, target: str) -> str:
    return reproducibility_hash(original, tool, bit_width, target)


def build_benchmark(*, artifact_id: str, target: str, category: str, task: str,
                    hcp: dict, manifest: dict) -> dict:
    bit_width = manifest["bit_width"]
    delta = -0.005 - (32 - bit_width) * 0.001
    size_ratio = manifest["compressed_size_bytes"] / max(1, manifest["original_size_bytes"])
    p50 = round(BASE_LATENCY_MS.get(category, 40.0) * (0.5 + size_ratio), 2)
    metric, baseline = QUALITY_METRIC.get(task, ("quality_score", 0.80))
    quality_value = round(baseline * (1 + delta), 4) if metric != "wer" and metric != "perplexity" else round(baseline * (1 - delta), 4)
    passed = delta >= hcp["min_quality_delta"] and (p50 * 1.4) <= hcp["latency_ceiling_ms_p95"]
    return {
        "artifact_id": artifact_id,
        "target": target,
        "quality_metric": metric,
        "quality_value": quality_value,
        "quality_delta": round(delta, 4),
        "latency_ms": {"p50": p50, "p95": round(p50 * 1.4, 2), "p99": round(p50 * 1.8, 2)},
        "peak_memory_mb": round(50 + manifest["compressed_size_bytes"] / 1024 / 1024, 2),
        "original_size_bytes": manifest["original_size_bytes"],
        "compressed_size_bytes": manifest["compressed_size_bytes"],
        "size_ratio": round(size_ratio, 4),
        "passed": passed,
        "emulated": True,
        "reproduction_commands": [
            f"cm pull {artifact_id}",
            f"cm verify {artifact_id}.tar.gz",
            "# Benchmarks are emulated in Phase 1 (see benchmark card).",
            "# A physical-device run replaces the emulated values, not the schema.",
        ],
    }


def benchmark_card_markdown(benchmark: dict, hcp: dict) -> str:
    lat = benchmark["latency_ms"]
    return (
        f"# Benchmark card — {benchmark['artifact_id']}\n\n"
        f"**Target:** {benchmark['target']} (emulated profile: {hcp['cpu']})\n\n"
        f"**Emulated:** yes — these numbers are deterministic estimates from the\n"
        f"target profile, NOT measurements on physical hardware.\n\n"
        f"- Quality: {benchmark['quality_metric']} = {benchmark['quality_value']} "
        f"(delta {benchmark['quality_delta']}, floor {hcp['min_quality_delta']})\n"
        f"- Latency: p50 {lat['p50']} ms, p95 {lat['p95']} ms, p99 {lat['p99']} ms "
        f"(ceiling p95 {hcp['latency_ceiling_ms_p95']} ms)\n"
        f"- Peak memory: {benchmark['peak_memory_mb']} MB\n"
        f"- Size: {benchmark['compressed_size_bytes']} / {benchmark['original_size_bytes']} "
        f"bytes (ratio {benchmark['size_ratio']})\n"
        f"- Passed gate: {benchmark['passed']}\n\n"
        f"## Reproduce\n\n"
        + "\n".join(f"`{c}`" for c in benchmark["reproduction_commands"])
        + "\n"
    )


def license_receipt(*, artifact_id: str, model_id: str, license_name, license_status: str) -> dict:
    return {
        "artifact_id": artifact_id,
        "model_id": model_id,
        "license": license_name,
        "license_status": license_status,
        "terms": (
            "Use under the upstream license. LICENSE_UNVERIFIED artifacts "
            "require human legal review before PUBLISHED."
        ),
    }


def write_artifact_tar(*, tar_path: Path, payload: bytes, manifest: dict,
                       benchmark: dict, benchmark_md: str, provenance: dict,
                       receipt: dict) -> None:
    files = {
        "model.bin": payload,
        "compression_manifest.json": json.dumps(manifest, indent=2).encode(),
        "benchmark_card.md": benchmark_md.encode(),
        "benchmark.json": json.dumps(benchmark, indent=2).encode(),
        "provenance_bundle.json": json.dumps(provenance, indent=2).encode(),
        "license_receipt.json": json.dumps(receipt, indent=2).encode(),
        "README.txt": (
            "Compress-Mart Phase 1 artifact.\n"
            "Compression is SIMULATED and benchmarks are EMULATED.\n"
            "Do not deploy this payload as a real quantized model.\n"
        ).encode(),
    }
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def verify_artifact_tar(tar_path: Path, trusted_public_pem: str | None = None) -> dict:
    """Shared by the server pull path and `cm verify`. Fail-closed."""
    checks = {"sha256_ok": False, "provenance_signature_ok": False,
              "benchmark_card_present": False, "license_receipt_present": False}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmp, filter="data")
            tmp = Path(tmp)
            required = ["model.bin", "provenance_bundle.json"]
            if not all((tmp / n).exists() for n in required):
                return {"ok": False, "checks": checks, "reason": "missing model.bin or provenance_bundle.json"}
            provenance = json.loads((tmp / "provenance_bundle.json").read_text())
            payload = (tmp / "model.bin").read_bytes()
            checks["sha256_ok"] = sha256_bytes(payload) == provenance["sha256"]
            ok, reason = verify_bundle(provenance, trusted_public_pem)
            checks["provenance_signature_ok"] = ok
            checks["benchmark_card_present"] = (tmp / "benchmark_card.md").exists()
            checks["license_receipt_present"] = (tmp / "license_receipt.json").exists()
            if not all(checks.values()):
                return {"ok": False, "checks": checks, "reason": reason if not ok else "a required check failed"}
            manifest = json.loads((tmp / "compression_manifest.json").read_text())
            return {"ok": True, "checks": checks, "reason": "ok",
                    "artifact_id": provenance["artifact_id"], "model_id": provenance["model_id"],
                    "manifest": manifest, "provenance": provenance}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "checks": checks, "reason": f"unreadable artifact: {exc}"}
