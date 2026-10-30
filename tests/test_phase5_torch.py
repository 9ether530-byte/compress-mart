import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

from compress_mart.app import create_app  # noqa: E402
from compress_mart.torch_bin import quantize_torch_file  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as c:
        yield c


def test_torch_bin_quantize_units(tmp_path):
    sd = {
        "w": torch.randn(32, 32),
        "b": torch.randn(8),
        "mask": torch.ones(4, 4, dtype=torch.uint8),
    }
    src = tmp_path / "model.bin"
    torch.save(sd, str(src))
    dst = tmp_path / "q.pt"
    stats = quantize_torch_file(src, dst)
    assert stats["quantized_tensor_count"] == 2
    assert stats["copied_tensor_count"] == 1
    assert stats["relative_mse"] < 0.01
    bundle = torch.load(str(dst), map_location="cpu", weights_only=True)
    assert bundle["quantized"]["w"].dtype == torch.int8
    assert torch.equal(bundle["copied"]["mask"], sd["mask"])
    deq = bundle["quantized"]["w"].float() * bundle["scales"]["w"]
    assert float(((sd["w"] - deq) ** 2).mean()) < 0.01 * float((sd["w"] ** 2).mean())


def test_torch_bin_upload_compress(client):
    sd = {"w": torch.randn(16, 16), "b": torch.randn(4)}
    buf_path = "/tmp/cm-test-torch.bin"
    torch.save(sd, buf_path)
    data = Path(buf_path).read_bytes()
    r = client.post("/ingest/upload", files={"file": ("model.bin", data)},
                    params={"task": "llm", "model_type": "llm"})
    assert r.status_code == 201, r.text
    model = r.json()
    r = client.post("/compress", json={"model_id": model["upstream_id"], "target": "rpi5"})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["manifest"]["compression_mode"] == "REAL"
    assert out["manifest"]["tool"] == "TorchInt8"
    assert out["manifest"]["format"] == "pt"
    assert out["benchmark"]["quality_value"] < 0.01
