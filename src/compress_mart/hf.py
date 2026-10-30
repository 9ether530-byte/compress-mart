"""Hugging Face client — stdlib only, no huggingface_hub dependency.

Two separate facts, kept apart on purpose:
- the model *card* (license, pipeline tag) is metadata from the HF API;
- the model *file* is bytes downloaded from the resolve URL.
A download is only recorded as huggingface-download when the bytes
actually came from that URL. Nothing else in Phase 2 claims a download.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

API_BASE = "https://huggingface.co/api/models"
RESOLVE_BASE = "https://huggingface.co"

PIPELINE_TO_TASK = {
    "text-generation": "llm",
    "text2text-generation": "llm",
    "automatic-speech-recognition": "asr",
    "image-classification": "vision",
    "object-detection": "vision",
    "image-segmentation": "vision",
    "text-classification": "text-classification",
    "feature-extraction": "embedding",
}


def fetch_card(model_id: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(
        f"{API_BASE}/{model_id}", headers={"User-Agent": "compress-mart/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        card = json.loads(resp.read())
    license_name = None
    card_data = card.get("cardData") or {}
    if card_data.get("license"):
        license_name = card_data["license"]
    else:
        for tag in card.get("tags", []):
            if tag.startswith("license:"):
                license_name = tag.split(":", 1)[1]
                break
    pipeline_tag = card.get("pipeline_tag")
    return {
        "license": license_name,
        "pipeline_tag": pipeline_tag,
        "task": PIPELINE_TO_TASK.get(pipeline_tag) if pipeline_tag else None,
        "siblings": [s["rfilename"] for s in card.get("siblings", [])],
    }


def download_file(model_id: str, filename: str, max_bytes: int = 200_000_000,
                  timeout: int = 300) -> bytes:
    url = f"{RESOLVE_BASE}/{model_id}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "compress-mart/0.2"})
    chunks: list[bytes] = []
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"download exceeds {max_bytes} bytes limit")
                chunks.append(chunk)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Hugging Face download failed ({exc.code}) for {filename}") from exc
    return b"".join(chunks)
