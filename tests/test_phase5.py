import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart.app import create_app  # noqa: E402
from compress_mart.safetensors import (  # noqa: E402
    load_safetensors_int8, quantize_safetensors_file)


def make_safetensors_typed(tensors: dict[str, tuple[str, np.ndarray]]) -> bytes:
    header, data = {}, bytearray()
    for name, (dtype, arr) in tensors.items():
        a = np.ascontiguousarray(arr)
        header[name] = {"dtype": dtype, "shape": list(a.shape),
                        "data_offsets": [len(data), len(data) + a.nbytes]}
        data.extend(a.tobytes())
    hb = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    return struct.pack("<Q", len(hb)) + hb + bytes(data)


def bf16_bytes(values: np.ndarray) -> np.ndarray:
    """Encode float32 values as BF16 (uint16) by truncating mantissa."""
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c, app, tmp_path


def test_f16_and_bf16_safetensors_quantize(tmp_path):
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.5, size=(32, 32)).astype(np.float32)
    src = tmp_path / "mixed.safetensors"
    src.write_bytes(make_safetensors_typed({
        "f16": ("F16", base.astype(np.float16)),
        "bf16": ("BF16", bf16_bytes(base)),
    }))
    dst = tmp_path / "q.safetensors"
    stats = quantize_safetensors_file(src, dst)
    assert stats["tensor_count"] == 2
    assert stats["relative_mse"] < 0.01
    tensors, scales = load_safetensors_int8(dst.read_bytes())
    assert tensors["f16"].dtype == np.int8 and tensors["bf16"].dtype == np.int8
    for name in ("f16", "bf16"):
        deq = tensors[name].astype(np.float32) * scales[name]
        assert np.mean((base - deq) ** 2) < 0.01 * np.mean(base ** 2)


def test_policy_apply_persists_and_reloads(client):
    c, app, tmp_path = client
    r = c.post("/router/policy/apply", json={"rule": "llm_laptop", "bit_width": 16})
    assert r.status_code == 200, r.text
    assert r.json()["new_bit_width"] == 16 and r.json()["persisted"] is True
    assert (tmp_path / "data" / "router_policy.yaml").exists()
    r = c.get("/router/policy")
    assert r.json()["source"] == "data_override"
    assert [x for x in r.json()["rules"] if x["name"] == "llm_laptop"][0]["bit_width"] == 16
    # A fresh app on the same data dir must pick up the override.
    app2 = create_app(data_dir=tmp_path / "data")
    with TestClient(app2) as c2:
        r = c2.get("/router/policy")
        assert r.json()["source"] == "data_override"
        assert [x for x in r.json()["rules"] if x["name"] == "llm_laptop"][0]["bit_width"] == 16
    r = c.post("/router/policy/apply", json={"rule": "nope", "bit_width": 8})
    assert r.status_code == 400


def test_feedback_apply_auto_persists_laptop_rule(client):
    c, app, tmp_path = client
    store = app.state.store
    for i in range(3):
        store.insert_feedback({
            "artifact_id": f"seed-{i}", "tool": "AngelSlim",
            "target": "intel-i7-13th", "category": "laptop", "bit_width": 8,
            "quality_delta": -0.09, "size_ratio": 0.3, "latency_p95": 999.0,
            "passed": False, "created_at": "2026-10-29T00:00:00+00:00"})
    r = c.post("/router/feedback/apply-auto")
    assert r.status_code == 200, r.text
    assert r.json()["persisted"] is True
    assert r.json()["applied"][0]["rule"] == "llm_laptop"
    assert r.json()["applied"][0]["new_bit_width"] == 16
    r = c.get("/router/policy")
    assert [x for x in r.json()["rules"] if x["name"] == "llm_laptop"][0]["bit_width"] == 16
