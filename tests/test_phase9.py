"""Phase 9 regression tests — one test per 2026-11-10 audit finding.

H1 router auto-bump for REAL paths (real 16-bit, not relabelled 8-bit);
H2 REAL bundles label themselves; M1 unsupported dtype fails loudly;
M2 trend 404; M3 apply-auto reports REAL aggregates; M4 measured
fields inside the tar; M5 offset validation; M6 upload cap;
L1 deprecate reachable; L4 agent/query gated; L6 hf edge cases;
L10 corrupt signing key quarantined, not a crash; L3 storage status
on a generic S3 endpoint; CLI search message.
"""
import io
import json
import struct
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compress_mart import hf as hf_mod  # noqa: E402
from compress_mart.app import create_app  # noqa: E402
from compress_mart.cli import cmd_search  # noqa: E402
from compress_mart.provenance import load_or_create_keypair  # noqa: E402
from compress_mart.safetensors import load_safetensors_int8  # noqa: E402
from compress_mart.storage import status as storage_status  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c, app


def _st_bytes(dtype, shape=(4, 4), values=None):
    arr = (np.arange(16, dtype=np.float64) if values is None else values).reshape(shape)
    if dtype == "F32":
        raw = arr.astype(np.float32).tobytes()
    elif dtype == "F64":
        raw = arr.astype(np.float64).tobytes()
    else:
        raw = arr.astype(np.float32).tobytes()
    hdr = {"w": {"dtype": dtype, "shape": list(shape), "data_offsets": [0, len(raw)]}}
    hb = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hb)) + hb + raw


def _upload_st(c, data, name="m.safetensors"):
    r = c.post("/ingest/upload", files={"file": (name, data)})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _seed_failing(store, tool, target, category="iot", bit_width=8):
    for i in range(3):
        store.insert_feedback({
            "artifact_id": f"seed-{i}", "tool": tool, "target": target,
            "category": category, "bit_width": bit_width,
            "quality_delta": -0.09, "size_ratio": 0.25, "latency_p95": 999.0,
            "passed": False, "created_at": "2026-10-29T00:00:00+00:00"})


def test_h1_real_safetensors_refines_to_real_16bit(client):
    c, app = client
    mid = _upload_st(c, _st_bytes("F32"))
    _seed_failing(app.state.store, "SafetensorsInt8", "jetson-orin-nano")
    r = c.post("/compress", json={"model_id": mid, "target": "jetson-orin-nano"})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["router_refined"] is True
    assert out["manifest"]["bit_width"] == 16
    assert out["manifest"]["tool"] == "SafetensorsInt16"
    assert out["manifest"]["compression_mode"] == "REAL"
    # The payload is really 16-bit, not a relabelled 8-bit file.
    art = app.state.store.get_artifact(out["artifact"]["id"])
    with tarfile.open(art["file_path"], "r:gz") as tar:
        payload = tar.extractfile("model.bin").read()
    tensors, scales = load_safetensors_int8(payload)
    assert tensors["w"].dtype == np.int16


def test_h1_explicit_bit_width_is_not_overridden(client):
    c, app = client
    mid = _upload_st(c, _st_bytes("F32"))
    _seed_failing(app.state.store, "SafetensorsInt8", "jetson-orin-nano")
    r = c.post("/compress", json={"model_id": mid, "target": "jetson-orin-nano",
                                  "bit_width": 8})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["router_refined"] is False
    assert out["manifest"]["bit_width"] == 8


def test_h1_real_npz_refines_to_real_16bit(client):
    c, app = client
    z = io.BytesIO()
    np.savez(z, weights=np.arange(256, dtype=np.float32).reshape(16, 16))
    r = c.post("/ingest/upload-weights", files={"file": ("w.npz", z.getvalue())})
    mid = r.json()["id"]
    _seed_failing(app.state.store, "NumPyInt8", "jetson-orin-nano")
    r = c.post("/compress", json={"model_id": mid, "target": "jetson-orin-nano"})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["router_refined"] is True
    assert out["manifest"]["bit_width"] == 16
    assert out["manifest"]["tool"] == "NumPyInt16"
    art = app.state.store.get_artifact(out["artifact"]["id"])
    with tarfile.open(art["file_path"], "r:gz") as tar:
        payload = tar.extractfile("model.bin").read()
    with np.load(io.BytesIO(payload)) as data:
        assert "weights_int16" in data.files


def test_h2_real_tar_labels_itself_real(client):
    c, app = client
    z = io.BytesIO()
    np.savez(z, weights=np.arange(256, dtype=np.float32).reshape(16, 16))
    r = c.post("/ingest/upload-weights", files={"file": ("w.npz", z.getvalue())})
    mid = r.json()["id"]
    out = c.post("/compress", json={"model_id": mid, "target": "rpi5"}).json()
    assert out["benchmark"]["emulated"] is False
    art = app.state.store.get_artifact(out["artifact"]["id"])
    with tarfile.open(art["file_path"], "r:gz") as tar:
        readme = tar.extractfile("README.txt").read().decode()
        assert "SIMULATED" not in readme and "Compression is REAL" in readme
        card = tar.extractfile("benchmark_card.md").read().decode()
        assert "Emulated:** no for quality" in card
        bench = json.loads(tar.extractfile("benchmark.json").read())
    assert bench["emulated"] is False
    assert bench["measured_host_dequant_ms"] >= 0
    assert bench["measured_host_dequant_elements"] == 256


def test_m1_f64_safetensors_fails_loudly(client):
    c, app = client
    mid = _upload_st(c, _st_bytes("F64"))
    r = c.post("/compress", json={"model_id": mid, "target": "rpi5"})
    assert r.status_code == 400
    assert "unsupported dtype" in r.text


def test_m2_trend_unknown_target_is_404(client):
    c, app = client
    r = c.get("/benchmarks/trend", params={"target": "bogus-target"})
    assert r.status_code == 404


def test_m3_apply_auto_reports_real_aggregates_as_skipped(client):
    c, app = client
    _seed_failing(app.state.store, "SafetensorsInt8", "jetson-orin-nano")
    r = c.post("/router/feedback/apply-auto")
    assert r.status_code == 200, r.text
    assert r.json()["applied"] == []
    assert len(r.json()["skipped"]) == 1
    assert "real-quantizer" in r.json()["skipped"][0]["reason"]
    assert r.json()["persisted"] is False


def test_m5_truncated_safetensors_offsets_rejected(client):
    c, app = client
    # Header claims 16 bytes of tensor data but only 8 are present.
    arr = np.arange(4, dtype=np.float32)
    hdr = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    hb = json.dumps(hdr).encode()
    data = struct.pack("<Q", len(hb)) + hb + arr.tobytes()[:8]
    mid = _upload_st(c, data)
    r = c.post("/compress", json={"model_id": mid, "target": "rpi5"})
    assert r.status_code == 400


def test_m6_upload_cap_is_413(client, monkeypatch):
    import compress_mart.app as app_mod
    monkeypatch.setattr(app_mod, "MAX_UPLOAD_BYTES", 100)
    c, app = client
    r = c.post("/ingest/upload", files={"file": ("big.bin", b"x" * 200)})
    assert r.status_code == 413


def test_l1_deprecate_reaches_deprecated(client):
    c, app = client
    c.post("/ingest/hub", json={"model_id": "openai/whisper-tiny"})
    out = c.post("/compress", json={"model_id": "openai/whisper-tiny",
                                    "target": "rpi5"}).json()
    art_id = out["artifact"]["id"]
    assert out["artifact"]["state"] == "CERTIFIED"
    r = c.post(f"/models/{art_id}/deprecate")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "DEPRECATED"
    r = c.get(f"/models/{art_id}")
    assert r.json()["state"] == "DEPRECATED"
    # Deprecating again is a 400, not a silent no-op.
    r = c.post(f"/models/{art_id}/deprecate")
    assert r.status_code == 400


def test_l4_agent_query_is_token_gated(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPRESS_MART_API_TOKEN", "test-token-123")
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        r = c.post("/agent/query", json={})
        assert r.status_code == 401
        r = c.post("/agent/query", json={}, headers={"X-API-Token": "test-token-123"})
        assert r.status_code == 200


def test_l6_hf_card_edge_cases(monkeypatch):
    class FakeResp:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode()
        def read(self):
            return self._payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    card = {"cardData": {"license": ["apache-2.0", "mit"]}, "tags": None,
            "pipeline_tag": "text-generation",
            "siblings": [{"size": 1}, {"rfilename": "model.safetensors"}]}
    monkeypatch.setattr(hf_mod.urllib.request, "urlopen",
                        lambda req, timeout=15: FakeResp(card))
    out = hf_mod.fetch_card("some/model")
    assert out["license"] == "apache-2.0"
    assert out["task"] == "llm"
    assert out["siblings"] == ["model.safetensors"]


def test_l10_corrupt_signing_key_is_quarantined(tmp_path):
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    (key_dir / "ed25519_private.pem").write_text("not a pem key")
    private, public_pem = load_or_create_keypair(key_dir)
    assert public_pem.startswith("-----BEGIN PUBLIC KEY-----")
    assert (key_dir / "ed25519_private.pem").exists()
    assert list(key_dir.glob("ed25519_private.corrupt-*.pem"))


def test_l3_storage_status_generic_s3(monkeypatch):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404)
            self.end_headers()
        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("COMPRESS_MART_S3_ENDPOINT",
                       f"http://127.0.0.1:{srv.server_address[1]}")
    assert storage_status()["available"] is True
    srv.shutdown()


def test_cli_search_message_for_empty_results(monkeypatch, capsys):
    import compress_mart.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_get", lambda base, path: [])
    args = type("A", (), {"target": None, "task": None, "max_size": None,
                         "state": None, "q": None, "base_url": "http://x"})()
    assert cmd_search(args) == 0
    assert "No artifacts match." in capsys.readouterr().out
