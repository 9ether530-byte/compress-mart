"""Postgres replica: a reporting copy of the SQLite catalog.

SQLite stays authoritative (the certification state machine lives
there). Postgres is a replica, synced on demand via POST
/postgres/sync, for reporting and for the day the primary moves.
It is NOT a live primary, and sync is not automatic — that is the
honest limit of this phase.

Env: COMPRESS_MART_PG_DSN (default host=127.0.0.1 port=5432
user=compress_mart password=compress_mart_dev dbname=compress_mart).
psycopg is optional: importorskip in tests, and the endpoint
returns 502 when it is missing or Postgres is unreachable.
"""
from __future__ import annotations

import json
import os

SCHEMA = """
CREATE TABLE IF NOT EXISTS cm_models (
    id TEXT PRIMARY KEY, upstream_id TEXT, task TEXT, license TEXT,
    license_status TEXT, size_bytes BIGINT, artifact_source TEXT);
CREATE TABLE IF NOT EXISTS cm_artifacts (
    id TEXT PRIMARY KEY, model_id TEXT, target TEXT, tool TEXT,
    bit_width INT, format TEXT, size_bytes BIGINT, state TEXT,
    benchmark_json JSONB, created_at TEXT);
"""


def dsn() -> str:
    base = os.environ.get(
        "COMPRESS_MART_PG_DSN",
        "host=127.0.0.1 port=5432 user=compress_mart "
        "password=compress_mart_dev dbname=compress_mart")
    # Health/status probes must not hang the API when Postgres is down.
    return base if "connect_timeout" in base else base + " connect_timeout=2"


def connect():
    import psycopg  # noqa: PLC0415
    return psycopg.connect(dsn())


def sync(store) -> dict:
    """Create tables if needed, replace replica rows with the
    current catalog, and return counts + a verification count."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute("DELETE FROM cm_artifacts")
            cur.execute("DELETE FROM cm_models")
            models = {}
            arts = store.list_artifacts()
            for a in arts:
                # An artifact whose model row is gone (orphan) is still
                # synced as an artifact; it is NOT re-invented as a
                # model by inserting the artifact id as a model id.
                model = store.get_model(a["model_id"])
                if model is not None:
                    models[a["model_id"]] = model
            for m in models.values():
                cur.execute(
                    "INSERT INTO cm_models VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (m["id"], m["upstream_id"], m["task"], m["license"],
                     m["license_status"], m.get("size_bytes"),
                     m.get("artifact_source")))
            for a in arts:
                cur.execute(
                    "INSERT INTO cm_artifacts VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)",
                    (a["id"], a["model_id"], a["target"], a["tool"],
                     a["bit_width"], a["format"], a["size_bytes"], a["state"],
                     json.dumps(a["benchmark"] or {}), a["created_at"]))
            cur.execute("SELECT COUNT(*) FROM cm_artifacts")
            verified = cur.fetchone()[0]
        conn.commit()
    return {"models_synced": len(models), "artifacts_synced": len(arts),
            "artifacts_verified_in_postgres": verified}


def status() -> dict:
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM cm_artifacts")
                n = cur.fetchone()[0]
        return {"available": True, "artifacts_in_replica": n}
    except Exception:  # noqa: BLE001
        return {"available": False, "artifacts_in_replica": None}
