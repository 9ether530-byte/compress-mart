"""Meilisearch integration (additive, not authoritative).

The SQLite catalog stays authoritative. Meilisearch is an index of
catalog entries for typo-tolerant, ranked search. If Meilisearch is
down or unset, search falls back to the local free-text /models?q=
path and says so in the response — never a 500 for a missing index.
Env: COMPRESS_MART_SEARCH_URL (default http://127.0.0.1:7700),
COMPRESS_MART_SEARCH_KEY (optional, for a secured Meilisearch).
Index name: compress-mart.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

INDEX = "compress-mart"


def search_url() -> str:
    return os.environ.get("COMPRESS_MART_SEARCH_URL", "http://127.0.0.1:7700").rstrip("/")


def _req(method: str, path: str, body: dict | list | None = None, timeout: int = 10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(search_url() + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    key = os.environ.get("COMPRESS_MART_SEARCH_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def document_for(art: dict) -> dict:
    return {
        "id": art["id"],
        "model_id": art.get("upstream_id") or art.get("model_id"),
        "target": art.get("target"),
        "task": art.get("task"),
        "tool": art.get("tool"),
        "format": art.get("format"),
        "bit_width": art.get("bit_width"),
        "state": art.get("state"),
        "license_status": art.get("license_status"),
    }


def index_artifacts(arts: list[dict]) -> dict:
    docs = [document_for(a) for a in arts]
    task = _req("PUT", f"/indexes/{INDEX}/documents?primaryKey=id", docs)
    return {"indexed": len(docs), "task_uid": task.get("taskUid")}


def search(query: str, limit: int = 20) -> dict:
    result = _req("POST", f"/indexes/{INDEX}/search",
                  {"q": query, "limit": limit})
    return {"hits": result.get("hits", []),
            "estimated_total_hits": result.get("estimatedTotalHits"),
            "processing_time_ms": result.get("processingTimeMs"),
            "engine": "meilisearch"}


def health() -> dict:
    try:
        h = _req("GET", "/health")
        return {"available": h.get("status") == "available", "url": search_url()}
    except Exception:  # noqa: BLE001
        return {"available": False, "url": search_url()}
