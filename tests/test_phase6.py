import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c


def _compress(client, model, target):
    client.post("/ingest/hub", json={"model_id": model})
    r = client.post("/compress", json={"model_id": model, "target": target})
    assert r.status_code == 201, r.text
    return r.json()


def test_deploy_snippet_auto_and_explicit(client):
    out = _compress(client, "openai/whisper-tiny", "rpi5")
    art = out["artifact"]["id"]
    # Demo path is onnx -> onnxruntime auto, and warns SIMULATED.
    r = client.get(f"/models/{art}/deploy-snippet")
    assert r.status_code == 200
    d = r.json()
    assert d["runtime"] == "onnxruntime"
    assert d["compression_mode"] == "SIMULATED"
    assert "WARNING" in d["snippet"]
    assert f"cm pull {art}" in d["snippet"]
    r = client.get(f"/models/{art}/deploy-snippet?runtime=ollama")
    assert r.json()["runtime"] == "ollama"
    assert "ollama run" in r.json()["snippet"]
    # Templates with literal braces in their code (torch, transformers)
    # must not 500 on .format-style substitution.
    for rt in ("torch", "transformers", "tflite", "numpy", "generic"):
        r = client.get(f"/models/{art}/deploy-snippet?runtime={rt}")
        assert r.status_code == 200, (rt, r.text)
        assert f"cm pull {art}" in r.json()["snippet"]
    r = client.get("/models/nope/deploy-snippet")
    assert r.status_code == 404
    r = client.get(f"/models/{art}/deploy-snippet?runtime=bogus")
    assert r.status_code == 400


def test_benchmark_trend(client):
    a1 = _compress(client, "openai/whisper-tiny", "rpi5")
    a2 = _compress(client, "openai/whisper-tiny", "rpi5")
    r = client.get("/benchmarks/trend?model_id=openai/whisper-tiny&target=rpi5")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert {row["artifact_id"] for row in rows} == {a1["artifact"]["id"], a2["artifact"]["id"]}
    assert all(row["size_ratio"] is not None for row in rows)
    # No filter returns the same rows here (only this model exists).
    assert len(client.get("/benchmarks/trend").json()) == 2
