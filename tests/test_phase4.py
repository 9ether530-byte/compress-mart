import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c, app


def _compress(client, model_id, target="rpi5"):
    r = client.post("/compress", json={"model_id": model_id, "target": target})
    assert r.status_code == 201, r.text
    return r.json()


def test_feedback_loop_and_facets_and_search(client):
    c, app = client
    r = c.post("/ingest/hub", json={"model_id": "openai/whisper-tiny"})
    assert r.status_code == 201
    out = _compress(c, "openai/whisper-tiny", "rpi5")
    assert out["router_refined"] is False
    # Benchmark -> feedback.
    r = c.get("/router/feedback")
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["tool"] == "TinyPort" and rows[0]["n"] == 1
    assert rows[0]["recommendation"] == "keep"
    # Facets + free-text search feed the storefront.
    r = c.get("/facets")
    assert r.json()["total"] == 1
    assert r.json()["tool"]["TinyPort"] == 1
    r = c.get("/models", params={"q": "whisper"})
    assert len(r.json()) == 1
    r = c.get("/models", params={"q": "nonexistent-model"})
    assert r.json() == []


def test_router_auto_refinement_from_failing_feedback(client):
    c, app = client
    store = app.state.store
    # Seed 3 failing samples for AngelSlim on intel-i7-13th at 8 bit.
    for i in range(3):
        store.insert_feedback({
            "artifact_id": f"seed-{i}", "tool": "AngelSlim",
            "target": "intel-i7-13th", "category": "laptop", "bit_width": 8,
            "quality_delta": -0.09, "size_ratio": 0.3, "latency_p95": 999.0,
            "passed": False, "created_at": "2026-10-29T00:00:00+00:00"})
    r = c.get("/router/feedback")
    agg = [a for a in r.json() if a["tool"] == "AngelSlim"][0]
    assert agg["recommendation"] == "increase bit_width"
    # A real llm+laptop compress must now be refined to 16 bit.
    r = c.post("/ingest/hub", json={"model_id": "demo-llm", "task": "llm", "model_type": "llm"})
    assert r.status_code == 201
    out = _compress(c, "demo-llm", "intel-i7-13th")
    assert out["router_refined"] is True
    assert out["manifest"]["bit_width"] == 16
    assert out["manifest"]["tool"] == "AngelSlim"


def test_reverify_quarantines_tampered_artifact(client, tmp_path):
    c, app = client
    c.post("/ingest/hub", json={"model_id": "openai/whisper-tiny"})
    out = _compress(c, "openai/whisper-tiny", "rpi5")
    art_id = out["artifact"]["id"]
    # Tamper the stored tar on disk, then run maintenance.
    art = app.state.store.get_artifact(art_id)
    tar_path = Path(art["file_path"])
    with tarfile.open(tar_path, "r:gz") as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    prov = json.loads(members["provenance_bundle.json"])
    prov["signature"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    members["provenance_bundle.json"] = json.dumps(prov).encode()
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    r = c.post("/maintenance/reverify")
    assert r.status_code == 200
    assert r.json()["quarantined"] == [art_id]
    r = c.get(f"/models/{art_id}/pull")
    assert r.status_code == 403


def test_token_auth_gates_writes_and_pull(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPRESS_MART_API_TOKEN", "test-token-123")
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        r = c.post("/ingest/hub", json={"model_id": "x"})
        assert r.status_code == 401
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["auth_enabled"] is True
        r = c.post("/ingest/hub", json={"model_id": "openai/whisper-tiny"},
                   headers={"X-API-Token": "test-token-123"})
        assert r.status_code == 201, r.text
        r = c.post("/compress", json={"model_id": "openai/whisper-tiny", "target": "rpi5"},
                   headers={"X-API-Token": "test-token-123"})
        assert r.status_code == 201, r.text
        art_id = r.json()["artifact"]["id"]
        r = c.get(f"/models/{art_id}/pull")
        assert r.status_code == 401
        r = c.get(f"/models/{art_id}/pull", headers={"X-API-Token": "test-token-123"})
        assert r.status_code == 200
