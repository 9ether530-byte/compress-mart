"""Safetensors parsing + real INT8 quantization — numpy + stdlib only.

Format (safetensors spec): 8-byte little-endian header length, JSON
header, then raw tensor bytes at the header's data_offsets. Phase 3
supported F32 only; Phase 5 adds F16 and BF16 reading (converted to
float32 for quantization; output is still per-tensor INT8). Any other
dtype fails loudly instead of being silently approximated.

Quantization is per-tensor symmetric INT8, chunked so the largest
tensor (whisper-tiny's 51,865 x 384 embedding, ~80 MB float32) is never
held as float32 + int8 + dequantized float32 all at once. Scales are
stored in the output header's __metadata__ so dequantization is exact
from the artifact alone.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

CHUNK = 1_000_000


def _to_float32(dtype: str, raw: bytes) -> np.ndarray:
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (bits << 16).view(np.float32)
    raise ValueError(f"unsupported dtype: {dtype}")


def parse_header(data: bytes) -> tuple[dict, int]:
    """Parse a safetensors header from the start of `data`.
    Returns (tensors dict, data_start offset). Raises ValueError if the
    bytes are not a safetensors file."""
    if len(data) < 8:
        raise ValueError("too short for a safetensors header")
    (header_len,) = struct.unpack("<Q", data[:8])
    if header_len > 10_000_000 or 8 + header_len > len(data):
        raise ValueError("invalid safetensors header length")
    try:
        header = json.loads(data[8:8 + header_len])
    except Exception as exc:
        raise ValueError(f"invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    for name, spec in tensors.items():
        if spec.get("dtype") not in ("F32", "F16", "BF16") or "shape" not in spec or "data_offsets" not in spec:
            raise ValueError(f"unsupported tensor {name}: {spec}")
    return tensors, 8 + header_len


def is_safetensors(data: bytes) -> bool:
    try:
        parse_header(data)
        return True
    except Exception:
        return False


def quantize_safetensors_file(src: Path, dst: Path) -> dict:
    """Quantize every tensor in `src` to INT8, writing `dst`.
    Streams one tensor at a time (never the whole file in memory).
    Returns aggregate stats (measured, not estimated)."""
    with open(src, "rb") as f:
        header_bytes = f.read(8)
        (header_len,) = struct.unpack("<Q", header_bytes)
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        for name, spec in tensors.items():
            if spec.get("dtype") not in ("F32", "F16", "BF16") or "shape" not in spec or "data_offsets" not in spec:
                raise ValueError(f"unsupported tensor {name}: {spec}")
        out_tensors: dict[str, dict] = {}
        out_data = bytearray()
        scales: dict[str, float] = {}
        total_sq_err = 0.0
        total_sq_orig = 0.0
        total_elems = 0
        for name, spec in tensors.items():
            start, end = spec["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            arr = _to_float32(spec["dtype"], raw)
            flat = arr.reshape(-1)
            max_abs = float(np.max(np.abs(flat))) if flat.size else 0.0
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            scales[name] = scale
            q_chunks = []
            for i in range(0, flat.size, CHUNK):
                chunk = flat[i:i + CHUNK]
                q = np.clip(np.round(chunk / scale), -127, 127).astype(np.int8)
                deq = q.astype(np.float32) * scale
                total_sq_err += float(np.sum((chunk - deq) ** 2))
                total_sq_orig += float(np.sum(chunk ** 2))
                q_chunks.append(q)
            q_all = np.concatenate(q_chunks) if q_chunks else np.zeros(0, dtype=np.int8)
            out_tensors[name] = {
                "dtype": "I8",
                "shape": spec["shape"],
                "data_offsets": [len(out_data), len(out_data) + len(q_all.tobytes())],
            }
            out_data.extend(q_all.tobytes())
            total_elems += int(flat.size)
    metadata = {
        "format": "pt",
        "compress_mart_compression": "int8-symmetric-per-tensor",
    }
    for name, scale in scales.items():
        metadata[f"compress_mart_scale:{name}"] = repr(scale)
    out_header = {"__metadata__": metadata, **out_tensors}
    out_header_bytes = json.dumps(out_header, separators=(",", ":")).encode()
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(out_header_bytes)))
        f.write(out_header_bytes)
        f.write(out_data)
    rel_mse = total_sq_err / total_sq_orig if total_sq_orig > 0 else 0.0
    return {
        "tensor_count": len(tensors),
        "total_elements": total_elems,
        "original_size_bytes": int(src.stat().st_size),
        "compressed_size_bytes": int(dst.stat().st_size),
        "relative_mse": rel_mse,
        "mean_sq_error": total_sq_err / total_elems if total_elems else 0.0,
    }


def load_safetensors_int8(data: bytes) -> tuple[dict, dict]:
    """Load an INT8 safetensors artifact: returns (tensors, scales).
    Tensors are int8 numpy arrays; dequantize with tensor * scale."""
    if len(data) < 8:
        raise ValueError("too short")
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + header_len])
    metadata = header.get("__metadata__", {})
    data_start = 8 + header_len
    tensors = {}
    scales = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec["dtype"] != "I8":
            raise ValueError(f"expected I8 tensor {name}, got {spec['dtype']}")
        start, end = spec["data_offsets"]
        arr = np.frombuffer(data[data_start + start:data_start + end], dtype=np.int8)
        tensors[name] = arr.reshape(spec["shape"])
        scales[name] = float(metadata[f"compress_mart_scale:{name}"])
    return tensors, scales
