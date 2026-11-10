"""Optional PyTorch .bin quantization — torch is NOT a hard dependency.

If torch is installed (see requirements-torch.txt, CPU wheel), a .bin
state dict can be really quantized: float tensors become per-tensor
INT8 (or INT16, the router's refinement step) with scales, non-float tensors (e.g. attention masks, uint8) are
copied unchanged and reported as such. If torch is missing, callers
must fall back to the SIMULATED path — never claim a real quantization
that did not happen.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .quantization import quantize_int


def is_torch_bin(path: Path) -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        import torch
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        return isinstance(sd, dict) and len(sd) > 0
    except Exception:
        return False


def quantize_torch_file(src: Path, dst: Path, bit_width: int = 8) -> dict:
    """Quantize float tensors to INT8 or INT16 (16 is the router's
    refinement step); non-float tensors are copied unchanged."""
    if bit_width not in (8, 16):
        raise ValueError(f"bit_width must be 8 or 16, got {bit_width}")
    import torch

    sd = torch.load(str(src), map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError("torch file does not contain a state dict")
    quantized, scales, copied = {}, {}, {}
    per_tensor_rel = []
    total_elems = 0
    quantized_count = 0
    for name, tensor in sd.items():
        if torch.is_floating_point(tensor) and tensor.numel() > 0:
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            q = quantize_int(arr, bit_width=bit_width)
            quantized[name] = torch.from_numpy(
                np.asarray(q["quantized"]).reshape(tensor.shape))
            scales[name] = float(q["scale"])
            # Mean per-tensor relative MSE, NOT a sum ratio: a single
            # -10000 masked_bias tensor would otherwise dominate the
            # aggregate (its squared magnitude is 1e8) and hide the
            # error on every real weight tensor.
            orig_sq = float(np.sum(arr.astype(np.float64) ** 2))
            if orig_sq > 0:
                per_tensor_rel.append(q["relative_mse"])
            total_elems += int(arr.size)
            quantized_count += 1
        else:
            copied[name] = tensor
    bundle = {
        "quantized": quantized,
        "scales": scales,
        "copied": copied,
        "bit_width": bit_width,
        "compress_mart_compression": f"int{bit_width}-symmetric-per-tensor",
    }
    torch.save(bundle, str(dst))
    rel_mse = float(np.mean(per_tensor_rel)) if per_tensor_rel else 0.0
    return {
        "tensor_count": len(sd),
        "bit_width": bit_width,
        "quantized_tensor_count": quantized_count,
        "copied_tensor_count": len(copied),
        "total_elements": total_elems,
        "original_size_bytes": int(src.stat().st_size),
        "compressed_size_bytes": int(dst.stat().st_size),
        "relative_mse": rel_mse,
    }
