"""Safetensors parsing + real INT8/INT16 quantization — numpy + stdlib only.

Format (safetensors spec): 8-byte little-endian header length, JSON
header, then raw tensor bytes at the header's data_offsets. Supported
dtypes are F32, F16 and BF16 (converted to float32 for quantization;
output is per-tensor INT8 or INT16). Any other dtype fails loudly
instead of being silently approximated: `parse_header` raises, and
`detect_safetensors` tells the caller a safetensors-shaped file with
an unsupported dtype was seen, so the API returns 400 rather than
falling through to a SIMULATED compress.

Quantization is per-tensor symmetric, chunked in 1M-element pieces.
One tensor at a time is read as float32; the quantized output is
appended to the output buffer as it is produced, so the only large
thing held in memory is the quantized output itself (~1/4 of the
original at 8 bit, ~1/2 at 16 bit), never the original file whole.
Scales are stored in the output header's __metadata__ so
dequantization is exact from the artifact alone.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

CHUNK = 1_000_000
SUPPORTED_DTYPES = ("F32", "F16", "BF16")
DTYPE_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I8": 1, "I16": 2,
                  "I32": 4, "I64": 8, "U8": 1, "U16": 2, "U32": 4, "U64": 8,
                  "BOOL": 1}
BITS = {8: ("I8", np.int8, 127), 16: ("I16", np.int16, 32767)}


def _to_float32(dtype: str, raw: bytes) -> np.ndarray:
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (bits << 16).view(np.float32)
    raise ValueError(f"unsupported dtype: {dtype}")


def _validate_offsets(tensors: dict) -> None:
    """Every tensor's data_offsets must describe exactly its shape's
    bytes, in-bounds and non-overlapping. A truncated or corrupt file
    is rejected here, never quantized with wrong stats."""
    spans = []
    for name, spec in tensors.items():
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if shape is None or offsets is None:
            raise ValueError(f"tensor {name} is missing shape or data_offsets")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            raise ValueError(f"tensor {name} has malformed data_offsets: {offsets!r}")
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError(f"tensor {name} has invalid data_offsets: {offsets!r}")
        itemsize = DTYPE_ITEMSIZE.get(spec.get("dtype"))
        if itemsize is None:
            raise ValueError(f"unsupported tensor {name}: {spec}")
        expected = math.prod(shape) * itemsize if shape else itemsize
        if end - start != expected:
            raise ValueError(
                f"tensor {name}: data_offsets span {end - start} bytes but "
                f"shape {shape} x {spec['dtype']} needs {expected}")
        spans.append((start, end, name))
    spans.sort()
    for (s1, e1, n1), (s2, e2, n2) in zip(spans, spans[1:]):
        if s2 < e1:
            raise ValueError(f"tensors {n1} and {n2} have overlapping data_offsets")


def parse_header(data: bytes) -> tuple[dict, int]:
    """Parse a safetensors header from the start of `data`.
    Returns (tensors dict, data_start offset). Raises ValueError if the
    bytes are not a safetensors file or any tensor is unsupported."""
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
        if spec.get("dtype") not in SUPPORTED_DTYPES or "shape" not in spec or "data_offsets" not in spec:
            raise ValueError(f"unsupported tensor {name}: {spec}")
    _validate_offsets(tensors)
    return tensors, 8 + header_len


def detect_safetensors(data: bytes) -> str:
    """Classify bytes as a safetensors file without quantizing.
    Returns "ok" (parseable, supported dtypes), "unsupported" (the
    header parses as safetensors but a tensor dtype is unsupported —
    fail loudly, do not fall through to SIMULATED), or "not-safetensors".
    """
    if len(data) < 8:
        return "not-safetensors"
    try:
        (header_len,) = struct.unpack("<Q", data[:8])
    except Exception:
        return "not-safetensors"
    if header_len > 10_000_000 or 8 + header_len > len(data):
        return "not-safetensors"
    try:
        header = json.loads(data[8:8 + header_len])
    except Exception:
        return "not-safetensors"
    if not isinstance(header, dict):
        return "not-safetensors"
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors or not all(
            isinstance(v, dict) and "data_offsets" in v and "shape" in v
            for v in tensors.values()):
        return "not-safetensors"
    if any(v.get("dtype") not in SUPPORTED_DTYPES for v in tensors.values()):
        return "unsupported"
    try:
        _validate_offsets(tensors)
    except ValueError:
        return "unsupported"
    return "ok"


def read_probe(path: Path) -> bytes:
    """Read just enough of a file to classify it as safetensors:
    the 8-byte length, then the whole header (capped at 10 MB by
    parse_header). A header larger than ~1 MB is handled, not
    misrouted to SIMULATED."""
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            return prefix
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > 10_000_000:
            return prefix
        return prefix + f.read(header_len)


def is_safetensors(data: bytes) -> bool:
    try:
        parse_header(data)
        return True
    except Exception:
        return False


def quantize_safetensors_file(src: Path, dst: Path, bit_width: int = 8) -> dict:
    """Quantize every tensor in `src` to INT8 or INT16, writing `dst`.
    Streams one tensor at a time in 1M-element chunks, appending the
    quantized bytes to the output as they are produced.
    Returns aggregate stats (measured, not estimated)."""
    if bit_width not in BITS:
        raise ValueError(f"bit_width must be 8 or 16, got {bit_width}")
    out_dtype, np_dtype, qmax = BITS[bit_width]
    file_size = int(src.stat().st_size)
    with open(src, "rb") as f:
        header_bytes = f.read(8)
        (header_len,) = struct.unpack("<Q", header_bytes)
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        for name, spec in tensors.items():
            if spec.get("dtype") not in SUPPORTED_DTYPES or "shape" not in spec or "data_offsets" not in spec:
                raise ValueError(f"unsupported tensor {name}: {spec}")
        _validate_offsets(tensors)
        for name, spec in tensors.items():
            start, end = spec["data_offsets"]
            if data_start + end > file_size:
                raise ValueError(f"tensor {name} extends past end of file (truncated)")
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
            if len(raw) != end - start:
                raise ValueError(f"tensor {name} is truncated in file")
            arr = _to_float32(spec["dtype"], raw)
            flat = arr.reshape(-1)
            max_abs = float(np.max(np.abs(flat))) if flat.size else 0.0
            scale = max_abs / qmax if max_abs > 0 else 1.0
            scales[name] = scale
            out_start = len(out_data)
            for i in range(0, flat.size, CHUNK):
                chunk = flat[i:i + CHUNK]
                q = np.clip(np.round(chunk / scale), -qmax, qmax).astype(np_dtype)
                deq = q.astype(np.float32) * scale
                total_sq_err += float(np.sum((chunk - deq) ** 2))
                total_sq_orig += float(np.sum(chunk ** 2))
                out_data.extend(q.tobytes())
            out_tensors[name] = {
                "dtype": out_dtype,
                "shape": spec["shape"],
                "data_offsets": [out_start, len(out_data)],
            }
            total_elems += int(flat.size)
    metadata = {
        "format": "pt",
        "compress_mart_compression": f"int{bit_width}-symmetric-per-tensor",
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
        "bit_width": bit_width,
        "original_size_bytes": int(src.stat().st_size),
        "compressed_size_bytes": int(dst.stat().st_size),
        "relative_mse": rel_mse,
        "mean_sq_error": total_sq_err / total_elems if total_elems else 0.0,
    }


def load_safetensors_int8(data: bytes) -> tuple[dict, dict]:
    """Load an INT8/INT16 safetensors artifact: returns (tensors, scales).
    Tensors are int8/int16 numpy arrays; dequantize with tensor * scale.
    A missing scale raises instead of silently defaulting to 1.0."""
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
        if spec["dtype"] not in ("I8", "I16"):
            raise ValueError(f"expected I8/I16 tensor {name}, got {spec['dtype']}")
        start, end = spec["data_offsets"]
        dtype = np.int8 if spec["dtype"] == "I8" else np.int16
        arr = np.frombuffer(data[data_start + start:data_start + end], dtype=dtype)
        tensors[name] = arr.reshape(spec["shape"])
        key = f"compress_mart_scale:{name}"
        if key not in metadata:
            raise ValueError(f"missing scale for tensor {name}")
        scales[name] = float(metadata[key])
    return tensors, scales
