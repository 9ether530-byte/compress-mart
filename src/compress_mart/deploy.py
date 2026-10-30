"""Deploy snippets: catalog -> runtime, generated from the artifact itself.

The tar always contains model.bin (see pipeline.py), so every snippet
starts with cm pull + extraction and then a runtime-specific load.
A SIMULATED artifact's snippet says so in plain text and is for
evaluation only — it must never be presented as a real quantized model.
"""
from __future__ import annotations

AUTO = {
    "gguf": "ollama",
    "onnx": "onnxruntime",
    "safetensors": "transformers",
    "pt": "torch",
    "tflite": "tflite",
    "npz": "numpy",
}
RUNTIMES = {"ollama", "onnxruntime", "transformers", "torch", "tflite", "numpy", "generic"}

_TEMPLATES = {
    "ollama": """# Ollama (GGUF)
cm pull {aid}
tar -xzf {aid}.tar.gz
# If the payload is a real GGUF file, load it with Ollama:
ollama create {aid} --file - <<'EOF'
FROM ./model.bin
EOF
ollama run {aid}""",
    "onnxruntime": """# ONNX Runtime
cm pull {aid}
tar -xzf {aid}.tar.gz
python3 - <<'PY'
import onnxruntime as ort
sess = ort.InferenceSession("model.bin")
print([i.name for i in sess.get_inputs()])
PY""",
    "transformers": """# Transformers (safetensors / torch)
cm pull {aid}
tar -xzf {aid}.tar.gz
python3 - <<'PY'
# For a real safetensors payload, load with safetensors.torch.load_file
# or AutoModelForCausalLM.from_pretrained(".") after renaming to model.safetensors.
from safetensors.torch import load_file
state = load_file("model.bin")
print(f"{len(state)} tensors")
PY""",
    "torch": """# PyTorch (.pt bundle: quantized + scales + copied)
cm pull {aid}
tar -xzf {aid}.tar.gz
python3 - <<'PY'
import torch
bundle = torch.load("model.bin", map_location="cpu", weights_only=True)
deq = {k: v.float() * bundle["scales"][k] for k, v in bundle["quantized"].items()}
print(f"{len(deq)} dequantized tensors, {len(bundle['copied'])} copied")
PY""",
    "tflite": """# TensorFlow Lite
cm pull {aid}
tar -xzf {aid}.tar.gz
python3 - <<'PY'
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="model.bin")
interpreter.allocate_tensors()
print(interpreter.get_input_details())
PY""",
    "numpy": """# NumPy (.npz payload)
cm pull {aid}
tar -xzf {aid}.tar.gz
python3 - <<'PY'
import numpy as np
# For a real INT8 .npz payload: dequantize with the stored scale.
data = np.load("model.bin")
print(list(data.files))
PY""",
    "generic": """# Generic
cm pull {aid}
tar -xzf {aid}.tar.gz
# Load model.bin with the runtime for format {fmt} on target {target}.
""",
}


def snippet(art: dict, runtime: str | None = None) -> dict:
    fmt = (art.get("format") or "").lower()
    chosen = runtime or AUTO.get(fmt, "generic")
    if chosen not in RUNTIMES:
        raise ValueError(f"unknown runtime {chosen}")
    text = (_TEMPLATES[chosen]
            .replace("{aid}", art["id"])
            .replace("{fmt}", fmt)
            .replace("{target}", str(art.get("target") or "")))
    mode = (art.get("manifest") or {}).get("compression_mode", "SIMULATED")
    if mode == "SIMULATED":
        text = ("# WARNING: this artifact is SIMULATED (a deterministic byte prefix, "
                "not a real quantized model). Use this snippet for evaluation only.\n"
                + text)
    return {
        "artifact_id": art["id"],
        "model_id": art.get("upstream_id") or art.get("model_id"),
        "target": art.get("target"),
        "format": fmt,
        "compression_mode": mode,
        "runtime": chosen,
        "pull_command": f"cm pull {art['id']}",
        "snippet": text,
    }
