"""SQLite store + certification state machine.

States: INGESTED -> COMPRESSED -> BENCHMARKED -> CERTIFIED -> PUBLISHED
         -> DEPRECATED, plus QUARANTINED (fail-closed, from any state).
Transitions are validated here, not in the API layer, so the CLI and the
API cannot bypass the gate.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

ALLOWED = {
    "INGESTED": {"COMPRESSED"},
    "COMPRESSED": {"BENCHMARKED"},
    "BENCHMARKED": {"CERTIFIED"},
    "CERTIFIED": {"PUBLISHED", "DEPRECATED"},
    "PUBLISHED": {"DEPRECATED"},
    "DEPRECATED": set(),
    "QUARANTINED": set(),
}


def demo_bytes(model_id: str, size: int = 262144) -> bytes:
    """Deterministic local-demo original, seeded by model id. This is NOT
    a download from Hugging Face; ingest records it as local-demo-seed."""
    seed = hashlib.sha256(model_id.encode()).digest()
    out = bytearray()
    while len(out) < size:
        out.extend(seed)
        seed = hashlib.sha256(seed).digest()
    return bytes(out[:size])


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "originals").mkdir(exist_ok=True)
        (self.data_dir / "artifacts").mkdir(exist_ok=True)
        self.db_path = self.data_dir / "compress_mart.db"
        self._init()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY, upstream_id TEXT, task TEXT, model_type TEXT,
                    license TEXT, license_status TEXT, sha256 TEXT, upstream_url TEXT,
                    size_bytes INTEGER, artifact_source TEXT, state TEXT, created_at TEXT);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, model_id TEXT, target TEXT, tool TEXT,
                    bit_width INTEGER, format TEXT, size_bytes INTEGER, sha256 TEXT,
                    state TEXT, benchmark_json TEXT, manifest_json TEXT,
                    provenance_json TEXT, file_path TEXT, created_at TEXT);
                CREATE TABLE IF NOT EXISTS router_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, artifact_id TEXT,
                    tool TEXT, target TEXT, category TEXT, bit_width INTEGER,
                    quality_delta REAL, size_ratio REAL, latency_p95 REAL,
                    passed INTEGER, created_at TEXT);
            """)

    # -- models --
    def insert_model(self, row: dict) -> dict:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (row["id"], row["upstream_id"], row["task"], row["model_type"],
                 row["license"], row["license_status"], row["sha256"],
                 row["upstream_url"], row["size_bytes"], row["artifact_source"],
                 row["state"], row["created_at"]))
        return row

    def get_model(self, model_id: str) -> dict | None:
        with self._conn() as conn:
            r = conn.execute(
                "SELECT * FROM models WHERE id=? OR upstream_id=?", (model_id, model_id)
            ).fetchone()
        return dict(r) if r else None

    def update_model_state(self, model_id: str, state: str):
        with self._conn() as conn:
            conn.execute("UPDATE models SET state=? WHERE id=?", (state, model_id))

    # -- artifacts --
    def insert_artifact(self, row: dict) -> dict:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row["id"], row["model_id"], row["target"], row["tool"],
                 row["bit_width"], row["format"], row["size_bytes"], row["sha256"],
                 row["state"], row["benchmark_json"], row["manifest_json"],
                 row["provenance_json"], row["file_path"], row["created_at"]))
        return row

    def get_artifact(self, artifact_id: str) -> dict | None:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        for k in ("benchmark_json", "manifest_json", "provenance_json"):
            d[k.replace("_json", "")] = json.loads(d[k]) if d[k] else None
        return d

    def update_artifact_state(self, artifact_id: str, new_state: str, *, force: bool = False):
        art = self.get_artifact(artifact_id)
        if not art:
            raise KeyError(artifact_id)
        if not force and new_state not in ALLOWED.get(art["state"], set()):
            raise ValueError(f"illegal transition {art['state']} -> {new_state}")
        with self._conn() as conn:
            conn.execute("UPDATE artifacts SET state=? WHERE id=?", (new_state, artifact_id))

    def delete_artifact(self, artifact_id: str):
        """Remove an artifact row (used to roll back a compress that
        failed after insert, so no orphan INGESTED row is left)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM artifacts WHERE id=?", (artifact_id,))

    def list_artifacts(self, *, target=None, task=None, bit_width=None,
                       license_status=None, max_size=None, state=None) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT a.*, m.task AS task, m.license AS license,
                       m.license_status AS license_status, m.upstream_id AS upstream_id
                FROM artifacts a LEFT JOIN models m ON m.id = a.model_id
            """).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["benchmark"] = json.loads(d["benchmark_json"]) if d["benchmark_json"] else None
            if target and d["target"] != target:
                continue
            if task and d["task"] != task:
                continue
            if bit_width and d["bit_width"] != int(bit_width):
                continue
            if license_status and d["license_status"] != license_status:
                continue
            if max_size and d["size_bytes"] > int(max_size):
                continue
            if state and d["state"] != state:
                continue
            out.append(d)
        return out

    def certifications(self, model_id: str | None = None) -> list[dict]:
        arts = self.list_artifacts()
        if model_id:
            arts = [a for a in arts if a["upstream_id"] == model_id or a["model_id"] == model_id]
        return [{"artifact_id": a["id"], "model_id": a["upstream_id"], "target": a["target"],
                 "state": a["state"], "benchmark": a["benchmark"]} for a in arts]

    # -- router feedback: benchmark -> router, closing the loop --
    def insert_feedback(self, row: dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO router_feedback "
                "(artifact_id, tool, target, category, bit_width, quality_delta, "
                "size_ratio, latency_p95, passed, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (row["artifact_id"], row["tool"], row["target"], row["category"],
                 row["bit_width"], row["quality_delta"], row["size_ratio"],
                 row["latency_p95"], 1 if row["passed"] else 0, row["created_at"]))

    def feedback_aggregates(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT tool, target, category, bit_width, COUNT(*) AS n, "
                "AVG(passed) AS pass_rate, AVG(quality_delta) AS avg_quality_delta, "
                "AVG(size_ratio) AS avg_size_ratio, AVG(latency_p95) AS avg_latency_p95 "
                "FROM router_feedback GROUP BY tool, target, category, bit_width"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["pass_rate"] = round(d["pass_rate"], 4)
            d["avg_quality_delta"] = round(d["avg_quality_delta"], 4)
            d["avg_size_ratio"] = round(d["avg_size_ratio"], 4)
            d["avg_latency_p95"] = round(d["avg_latency_p95"], 2)
            # Recommendation rule, applied by the router on the next compress:
            # 3+ samples and under half passing means the bit width is too
            # aggressive for this tool/target — bump it if the target allows.
            d["recommendation"] = (
                "increase bit_width" if d["n"] >= 3 and d["pass_rate"] < 0.5 else "keep"
            )
            out.append(d)
        return out

    def facets(self) -> dict:
        from collections import Counter
        arts = self.list_artifacts()
        def counts(key):
            return dict(Counter(str(a[key]) for a in arts))
        return {
            "target": counts("target"), "task": counts("task"),
            "tool": counts("tool"), "bit_width": counts("bit_width"),
            "state": counts("state"), "license_status": counts("license_status"),
            "total": len(arts),
        }

    def trend(self, model_id: str | None = None, target: str | None = None) -> list[dict]:
        """Benchmark history for a model/target, oldest first: the time
        series the router feedback aggregates are built from."""
        arts = self.list_artifacts()
        out = []
        for a in arts:
            if model_id and a["upstream_id"] != model_id and a["model_id"] != model_id:
                continue
            if target and a["target"] != target:
                continue
            b = a["benchmark"] or {}
            out.append({
                "artifact_id": a["id"], "model_id": a["upstream_id"],
                "target": a["target"], "tool": a["tool"],
                "bit_width": a["bit_width"], "format": a["format"],
                "state": a["state"], "created_at": a["created_at"],
                "size_ratio": b.get("size_ratio"),
                "quality_delta": b.get("quality_delta"),
                "latency_p95_ms": (b.get("latency_ms") or {}).get("p95"),
                "passed": b.get("passed"),
            })
        out.sort(key=lambda d: (d["created_at"] or "", d["artifact_id"]))
        return out
