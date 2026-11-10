"""Real symmetric quantization — numpy only, no torch needed.

This adapter is NOT simulated: it takes a real float32 weight tensor
(uploaded as an .npz with exactly one `weights` array), quantizes it
to int8 or int16 with a per-tensor scale, and measures the dequant
error. Bit width 16 is the real refinement step the router feedback
loop bumps to: a 16-bit artifact is genuinely larger and more accurate
than the 8-bit one, not a relabelled 8-bit payload.
"""
from __future__ import annotations

import io

import numpy as np

BITS = {8: (np.int8, 127), 16: (np.int16, 32767)}


def quantize_int(weights: np.ndarray, bit_width: int = 8) -> dict:
    if bit_width not in BITS:
        raise ValueError(f"bit_width must be 8 or 16, got {bit_width}")
    dtype, qmax = BITS[bit_width]
    w = np.asarray(weights, dtype=np.float32)
    max_abs = float(np.max(np.abs(w))) if w.size else 0.0
    scale = max_abs / qmax if max_abs > 0 else 1.0
    # np.array (not .astype on a possible scalar) so 0-d tensors stay
    # 0-d arrays, which torch.from_numpy accepts.
    q = np.array(np.clip(np.round(w / scale), -qmax, qmax), dtype=dtype)
    deq = q.astype(np.float32) * scale
    mse = float(np.mean((w - deq) ** 2)) if w.size else 0.0
    denom = float(np.mean(w ** 2)) if w.size else 0.0
    rel_mse = mse / denom if denom > 0 else 0.0
    return {
        "quantized": q,
        "scale": scale,
        "bit_width": bit_width,
        "dequantized": deq,
        "mse": mse,
        "relative_mse": rel_mse,
        "max_abs_error": float(np.max(np.abs(w - deq))) if w.size else 0.0,
    }


def quantize_int8(weights: np.ndarray) -> dict:
    return quantize_int(weights, bit_width=8)


def pack_npz(quantized: np.ndarray, scale: float, bit_width: int = 8) -> bytes:
    buf = io.BytesIO()
    key = "weights_int8" if bit_width == 8 else "weights_int16"
    np.savez_compressed(buf, **{key: quantized}, scale=np.float32(scale))
    return buf.getvalue()


def load_npz_weights(data: bytes) -> np.ndarray:
    with np.load(io.BytesIO(data)) as z:
        files = set(z.files)
        if files != {"weights"}:
            raise ValueError(
                f"npz must contain exactly one 'weights' array, got {sorted(files)}")
        return np.asarray(z["weights"], dtype=np.float32)
