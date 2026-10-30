import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point search at a dead port so tests exercise the local fallback,
    # not a real Meilisearch (live Meilisearch is proven in smoke, not CI).
    monkeypatch.setenv("COMPRESS_MART_SEARCH_URL", "http://127.0.0.1:7799")
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c


def _compress(client, model, target):
    client.post("/ingest/hub", json={"model_id": model})
    r = client.post("/compress", json={"model_id": model, "target": target})
    assert r.status_code == 201, r.text
    return r.json()


def test_search_fallback_and_index_502(client):
    _compress(client, "openai/whisper-tiny", "rpi5")
    r = client.get("/search?q=whisper")
    assert r.status_code == 200
    d = r.json()
    assert d["engine"] == "local-fallback"
    assert len(d["artifacts"]) == 1
    r = client.post("/search/index")
    assert r.status_code == 502


def test_measured_host_dequant_npz(client, tmp_path):
    import numpy as np
    buf = tmp_path / "w.npz"
    np.savez(buf, weights=np.random.randn(16, 16).astype(np.float32))
    with open(buf, "rb") as f:
        r = client.post("/ingest/upload-weights", files={"file": ("w.npz", f.read())},
                        params={"task": "llm"})
    assert r.status_code == 201, r.text
    model = r.json()
    r = client.post("/compress", json={"model_id": model["upstream_id"], "target": "rpi5"})
    assert r.status_code == 201, r.text
    b = r.json()["benchmark"]
    assert b["measured_host"] is True
    assert b["measured_host_dequant_elements"] == 256
    assert b["measured_host_dequant_ms"] >= 0


def test_measured_host_absent_for_simulated(client):
    out = _compress(client, "openai/whisper-tiny", "rpi5")
    b = out["benchmark"]
    assert "measured_host_dequant_ms" not in b
