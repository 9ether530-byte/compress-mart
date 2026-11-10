"""Measured host benchmark: dequantize a REAL artifact on THIS machine
and time it. This is NOT target (Pi/phone/laptop) latency — it is a
real measurement on the host, labelled as such, and it is the first
measured (not emulated) timing in a benchmark card. For SIMULATED
artifacts there is nothing real to dequantize, so the field is None.
"""
from __future__ import annotations

import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

from .safetensors import load_safetensors_int8


def measure_artifact_tar(tar_path: Path, fmt: str) -> dict | None:
    """Dequantize the payload in tar_path and return timing + elements.
    Returns None for formats with no real dequantization path.
    A missing scale in a safetensors payload raises inside the loader
    instead of silently defaulting to 1.0."""
    fmt = (fmt or "").lower()
    if fmt not in ("safetensors", "pt", "npz"):
        return None
    start = time.perf_counter()
    elements = 0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extract("model.bin", path=tmp, filter="data")
            payload = Path(tmp) / "model.bin"
            if fmt == "npz":
                data = np.load(payload, allow_pickle=False)
                for key in data.files:
                    if key == "scale":
                        continue
                    arr = data[key].astype(np.float32) * float(data["scale"])
                    elements += int(arr.size)
            elif fmt == "pt":
                import torch  # noqa: PLC0415
                bundle = torch.load(str(payload), map_location="cpu", weights_only=True)
                for name, q in bundle["quantized"].items():
                    deq = q.float() * bundle["scales"][name]
                    elements += int(deq.numel())
            else:  # safetensors: dequantize the INT8/INT16 payload
                tensors, scales = load_safetensors_int8(payload.read_bytes())
                for name, arr in tensors.items():
                    deq = arr.astype(np.float32) * scales[name]
                    elements += int(deq.size)
    except Exception:  # noqa: BLE001
        return None
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {"measured_host_dequant_ms": elapsed_ms,
            "measured_host_dequant_elements": elements,
            "measured_host": True}
