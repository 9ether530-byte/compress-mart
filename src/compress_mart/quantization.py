"""Real INT8 symmetric quantization — numpy only, no torch needed.

This is the one Phase 2 adapter that is NOT simulated: it takes a real
float32 weight tensor (uploaded as an .npz with a single `weights` array),
quantizes it to int8 with a per-tensor scale, and measures the dequant
error. It does NOT apply to Hugging Face .bin/.safetensors files, whose
weights cannot be parsed without torch/transformers — those stay SIMULATED
and say so in their manifests.
"""
from __future__ import annotations

import io

import numpy as np


def quantize_int8(weights: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=np.float32)
    max_abs = float(np.max(np.abs(w))) if w.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    # np.array (not .astype on a possible scalar) so 0-d tensors stay
    # 0-d arrays, which torch.from_numpy accepts.
    q = np.array(np.clip(np.round(w / scale), -127, 127), dtype=np.int8)
    deq = q.astype(np.float32) * scale
    mse = float(np.mean((w - deq) ** 2)) if w.size else 0.0
    denom = float(np.mean(w ** 2)) if w.size else 0.0
    rel_mse = mse / denom if denom > 0 else 0.0
    return {
        "quantized": q,
        "scale": scale,
        "dequantized": deq,
        "mse": mse,
        "relative_mse": rel_mse,
        "max_abs_error": float(np.max(np.abs(w - deq))) if w.size else 0.0,
    }


def pack_npz(quantized: np.ndarray, scale: float) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, weights_int8=quantized, scale=np.float32(scale))
    return buf.getvalue()


def load_npz_weights(data: bytes) -> np.ndarray:
    with np.load(io.BytesIO(data)) as z:
        if "weights" not in z:
            raise ValueError("npz must contain a single 'weights' array")
        return np.asarray(z["weights"], dtype=np.float32)
