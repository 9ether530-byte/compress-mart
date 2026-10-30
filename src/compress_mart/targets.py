"""Target matrix loading and Hardware Capability Profiles (emulated)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = REPO_ROOT / "targets" / "targets.yaml"


def load_targets(path: Path | None = None) -> dict:
    path = path or DEFAULT_TARGETS
    with open(path) as f:
        data = yaml.safe_load(f)
    targets = {t["id"]: t for t in data["targets"]}
    return {"targets": targets, "aliases": data.get("aliases", {})}


def resolve_target(name: str, matrix: dict) -> str:
    if name in matrix["targets"]:
        return name
    if name in matrix["aliases"]:
        return matrix["aliases"][name]
    raise KeyError(f"unknown target: {name}")


def hcp(target_id: str, matrix: dict) -> dict:
    resolved = resolve_target(target_id, matrix)
    t = matrix["targets"][resolved]
    return {
        "target_id": resolved,
        "category": t["category"],
        "cpu": t["cpu"],
        "isa_flags": t["isa_flags"],
        "npu_available": t["npu_available"],
        "memory_mb": t["memory_mb"],
        "memory_bandwidth_gbs": t["memory_bandwidth_gbs"],
        "recommended_bit_widths": t["recommended_bit_widths"],
        "latency_ceiling_ms_p95": t["latency_ceiling_ms_p95"],
        "min_quality_delta": t["min_quality_delta"],
        "emulated": True,
    }
