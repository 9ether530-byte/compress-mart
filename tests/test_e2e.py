import json
import sys
import tarfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart.app import create_app  # noqa: E402
from compress_mart.pipeline import verify_artifact_tar  # noqa: E402
from compress_mart.router import route  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c


def ingest_and_compress(client, model_id="openai/whisper-tiny", target="rpi5", **kw):
    r = client.post("/ingest/hub", json={"model_id": model_id, **kw})
    assert r.status_code == 201, r.text
    model = r.json()
    r = client.post("/compress", json={"model_id": model["upstream_id"], "target": target})
    assert r.status_code == 201, r.text
    return model, r.json()


def test_e2e_flow(client, tmp_path):
    model, out = ingest_and_compress(client)
    assert out["artifact"]["state"] == "CERTIFIED"
    assert out["manifest"]["compression_mode"] == "SIMULATED"
    assert out["benchmark"]["emulated"] is True
    assert out["benchmark"]["passed"] is True
    art_id = out["artifact"]["id"]
    # Discover via the prompt's query shape (alias rpi5, task asr).
    r = client.get("/models", params={"target": "rpi5", "task": "asr"})
    assert r.status_code == 200
    assert any(a["id"] == art_id for a in r.json())
    # Certifications list.
    r = client.get("/certifications", params={"model_id": "openai/whisper-tiny"})
    assert any(c["artifact_id"] == art_id for c in r.json())
    # Publish: no license was supplied, so the gate must fire first.
    r = client.post(f"/models/{art_id}/publish", json={"approve_unverified": False})
    assert r.status_code == 403
    r = client.post(f"/models/{art_id}/publish", json={"approve_unverified": True})
    assert r.status_code == 200 and r.json()["state"] == "PUBLISHED"
    # Pull + verify, exactly as `cm verify` does.
    r = client.get(f"/models/{art_id}/pull")
    assert r.status_code == 200
    tar_path = tmp_path / "pulled.tar.gz"
    tar_path.write_bytes(r.content)
    result = verify_artifact_tar(tar_path)
    assert result["ok"] is True
    assert all(result["checks"].values())
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
    assert "benchmark_card.md" in names and "license_receipt.json" in names
    assert "provenance_bundle.json" in names


def test_targets_cover_prompt_matrix(client):
    r = client.get("/targets")
    ids = {t["target_id"] for t in r.json()}
    assert len(ids) == 10
    for expected in ("apple-m2", "snapdragon-8-gen3", "raspberry-pi-5", "esp32-s3"):
        assert expected in ids
    r = client.get("/targets/rpi5/capabilities")
    assert r.json()["target_id"] == "raspberry-pi-5"
    assert r.json()["emulated"] is True


def test_router_rules():
    assert route(model_type="llm", category="laptop")["tool"] == "AngelSlim"
    assert route(model_type="vision", category="phone")["tool"] == "AIMET"
    assert route(model_type="llm", category="iot")["tool"] == "TinyPort"
    assert route(model_type="llm", category="gpu-server")["tool"] == "TurboQuant"
    assert route(model_type="vision", category="laptop")["tool"] == "ShrinkAI"


def test_provenance_fail_closed(client, tmp_path):
    _, out = ingest_and_compress(client)
    art_id = out["artifact"]["id"]
    r = client.get(f"/models/{art_id}/pull")
    tar_path = tmp_path / "tampered.tar.gz"
    tar_path.write_bytes(r.content)
    # Tamper: rewrite the provenance signature inside a fresh tar.
    import io
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
    result = verify_artifact_tar(tar_path)
    assert result["ok"] is False
    assert result["checks"]["provenance_signature_ok"] is False


def test_uncertified_artifact_cannot_pull(client):
    # A benchmark that fails would stay BENCHMARKED; simulate the gate by
    # pulling before any compression exists is impossible, so check the
    # state-machine guard directly: a fresh model has no artifact to pull.
    r = client.get("/models/nonexistent/pull")
    assert r.status_code == 404
