import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart.app import create_app  # noqa: E402
from compress_mart.pipeline import verify_artifact_tar  # noqa: E402
from compress_mart.quantization import load_npz_weights, quantize_int8  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c


def test_real_int8_quantization_units():
    rng = np.random.default_rng(42)
    w = rng.normal(0, 0.5, size=(64, 64)).astype(np.float32)
    q = quantize_int8(w)
    assert q["quantized"].dtype == np.int8
    assert q["relative_mse"] < 0.01
    assert q["quantized"].shape == w.shape


def test_weights_upload_compress_pull_verify(client, tmp_path):
    rng = np.random.default_rng(7)
    w = rng.normal(0, 1.0, size=(200, 200)).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, weights=w)
    r = client.post("/ingest/upload-weights", files={"file": ("weights.npz", buf.getvalue())})
    assert r.status_code == 201, r.text
    model = r.json()
    assert model["model_type"] == "weights"
    r = client.post("/compress", json={"model_id": model["upstream_id"], "target": "rpi5"})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["manifest"]["compression_mode"] == "REAL"
    assert out["manifest"]["tool"] == "NumPyInt8"
    assert out["benchmark"]["quality_metric"] == "dequant_relative_mse"
    assert out["benchmark"]["quality_value"] < 0.01
    assert out["benchmark"]["dequant_mse"] >= 0
    assert out["benchmark"]["passed"] is True
    assert out["artifact"]["state"] == "CERTIFIED"
    assert out["artifact"]["size_bytes"] < out["benchmark"]["original_size_bytes"]
    # The pulled payload really is an int8 npz, not a truncated prefix.
    art_id = out["artifact"]["id"]
    r = client.get(f"/models/{art_id}/pull")
    assert r.status_code == 200
    tar_path = tmp_path / "w.tar.gz"
    tar_path.write_bytes(r.content)
    assert verify_artifact_tar(tar_path)["ok"] is True
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tar:
        payload = tar.extractfile("model.bin").read()
    with np.load(io.BytesIO(payload)) as z:
        assert z["weights_int8"].dtype == np.int8
        assert float(z["scale"]) > 0


def test_real_hf_download_small_model(client):
    """Live test: skipped (not failed) if HF is unreachable from this box."""
    try:
        r = client.post("/ingest/hub", json={
            "model_id": "sshleifer/tiny-gpt2", "download_file": "pytorch_model.bin"})
    except Exception:
        pytest.skip("HF unreachable")
    if r.status_code == 502:
        pytest.skip(f"HF ingest failed in this environment: {r.text}")
    assert r.status_code == 201, r.text
    model = r.json()
    assert model["artifact_source"] == "huggingface-download"
    assert model["size_bytes"] > 1_000_000
    assert model["task"] == "llm"
    # tiny-gpt2 has no license in its card, so it must stay unverified.
    assert model["license_status"] == "LICENSE_UNVERIFIED"
