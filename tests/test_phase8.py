import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart.app import create_app  # noqa: E402
from compress_mart.storage import signature_headers  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPRESS_MART_SEARCH_URL", "http://127.0.0.1:7799")
    monkeypatch.setenv("COMPRESS_MART_PG_DSN", "host=127.0.0.1 port=5599 user=x password=x dbname=x")
    monkeypatch.setenv("COMPRESS_MART_S3_ENDPOINT", "http://127.0.0.1:9999")
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c


def test_external_status_reports_unavailable(client):
    r = client.get("/external/status")
    assert r.status_code == 200
    d = r.json()
    assert d["primary"] == "sqlite (authoritative)"
    assert d["search_meilisearch"]["available"] is False
    assert d["postgres_replica"]["available"] is False
    assert d["storage_s3"]["available"] is False


def test_postgres_sync_502_when_down(client):
    r = client.post("/postgres/sync")
    assert r.status_code == 502


def test_storage_sync_502_when_down(client):
    client.post("/ingest/hub", json={"model_id": "openai/whisper-tiny"})
    client.post("/compress", json={"model_id": "openai/whisper-tiny", "target": "rpi5"})
    r = client.post("/storage/sync")
    assert r.status_code == 502


def test_s3_signature_headers_structure(monkeypatch):
    monkeypatch.setenv("COMPRESS_MART_S3_ACCESS_KEY", "testkey")
    monkeypatch.setenv("COMPRESS_MART_S3_SECRET_KEY", "testsecret")
    h = signature_headers("PUT", "http://127.0.0.1:9000/compress-mart/art-1.tar.gz", b"hello")
    assert h["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=testkey/")
    assert "SignedHeaders=" in h["Authorization"]
    assert len(h["Authorization"].split("Signature=")[1]) == 64
