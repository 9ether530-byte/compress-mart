import io
import json
import os
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart.app import create_app  # noqa: E402
from compress_mart.pipeline import verify_artifact_tar  # noqa: E402
from compress_mart.safetensors import (  # noqa: E402
    is_safetensors, load_safetensors_int8, quantize_safetensors_file)


def make_safetensors(tensors: dict[str, np.ndarray]) -> bytes:
    header = {}
    data = bytearray()
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [len(data), len(data) + a.nbytes]}
        data.extend(a.tobytes())
    header_bytes = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    return struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(data)


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c


def test_safetensors_quantize_units(tmp_path):
    rng = np.random.default_rng(3)
    original = {"a": rng.normal(0, 0.5, size=(16, 16)).astype(np.float32),
                "b": rng.normal(0, 2.0, size=(8,)).astype(np.float32)}
    src = tmp_path / "orig.safetensors"
    src.write_bytes(make_safetensors(original))
    assert is_safetensors(src.read_bytes())
    dst = tmp_path / "q.safetensors"
    stats = quantize_safetensors_file(src, dst)
    assert stats["tensor_count"] == 2
    assert stats["compressed_size_bytes"] < stats["original_size_bytes"]
    assert stats["relative_mse"] < 0.01
    tensors, scales = load_safetensors_int8(dst.read_bytes())
    assert tensors["a"].dtype == np.int8
    deq = tensors["a"].astype(np.float32) * scales["a"]
    assert np.mean((original["a"] - deq) ** 2) < 0.01 * np.mean(original["a"] ** 2)


def test_safetensors_upload_compress_verify(client, tmp_path):
    rng = np.random.default_rng(11)
    w = rng.normal(0, 1.0, size=(64, 64)).astype(np.float32)
    payload = make_safetensors({"layer.weight": w})
    r = client.post("/ingest/upload", files={"file": ("model.safetensors", payload)},
                    params={"task": "llm", "model_type": "llm"})
    assert r.status_code == 201, r.text
    model = r.json()
    r = client.post("/compress", json={"model_id": model["upstream_id"], "target": "rpi5"})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["manifest"]["compression_mode"] == "REAL"
    assert out["manifest"]["tool"] == "SafetensorsInt8"
    assert out["manifest"]["format"] == "safetensors"
    assert out["benchmark"]["tensor_count"] == 1
    assert out["benchmark"]["quality_value"] < 0.01
    assert out["artifact"]["state"] == "CERTIFIED"
    art_id = out["artifact"]["id"]
    r = client.get(f"/models/{art_id}/pull")
    assert r.status_code == 200
    tar_path = tmp_path / "st.tar.gz"
    tar_path.write_bytes(r.content)
    assert verify_artifact_tar(tar_path)["ok"] is True
    with tarfile.open(tar_path, "r:gz") as tar:
        model_bin = tar.extractfile("model.bin").read()
    tensors, _ = load_safetensors_int8(model_bin)
    assert tensors["layer.weight"].dtype == np.int8


@pytest.mark.skipif(os.environ.get("COMPRESS_MART_LIVE") != "1",
                    reason="live HF test only with COMPRESS_MART_LIVE=1")
def test_live_whisper_tiny_safetensors(client):
    """Downloads the real 151 MB whisper-tiny safetensors and quantizes it.
    Deliberately not in the default suite: it is slow and network-bound."""
    r = client.post("/ingest/hub", json={
        "model_id": "openai/whisper-tiny", "download_file": "model.safetensors"})
    assert r.status_code == 201, r.text
    model = r.json()
    assert model["artifact_source"] == "huggingface-download"
    assert model["size_bytes"] > 100_000_000
    assert model["license"] == "apache-2.0"
    r = client.post("/compress", json={"model_id": model["upstream_id"], "target": "rpi5"})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["manifest"]["compression_mode"] == "REAL"
    assert out["manifest"]["format"] == "safetensors"
    assert out["benchmark"]["tensor_count"] == 167
    assert out["benchmark"]["quality_value"] < 0.01
    assert out["artifact"]["size_bytes"] < out["benchmark"]["original_size_bytes"] * 0.4
