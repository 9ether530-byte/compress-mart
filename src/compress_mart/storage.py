"""S3-compatible object storage client, stdlib only (no boto3).

Implements the subset Compress-Mart needs: make bucket, PUT an
artifact tar, GET it back, list objects. Signing is AWS Signature
Version 4, the same scheme MinIO speaks. This module is proven in
tests against a local HTTP stub and by signature structure — it is
NOT live-proven against MinIO in this sandbox, because the MinIO
image pull is denied by the registry here (see README Phase 8).
Point COMPRESS_MART_S3_ENDPOINT at a real MinIO/S3 endpoint and it
works unchanged; the endpoint returns 502 until then.

Env: COMPRESS_MART_S3_ENDPOINT (default http://127.0.0.1:9000),
COMPRESS_MART_S3_ACCESS_KEY, COMPRESS_MART_S3_SECRET_KEY,
COMPRESS_MART_S3_BUCKET (default compress-mart), region us-east-1.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def endpoint() -> str:
    return os.environ.get("COMPRESS_MART_S3_ENDPOINT", "http://127.0.0.1:9000").rstrip("/")


def bucket() -> str:
    return os.environ.get("COMPRESS_MART_S3_BUCKET", "compress-mart")


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def signature_headers(method: str, url: str, body: bytes, content_type: str = "application/octet-stream") -> dict:
    """Return the SigV4 headers for a request. Pure function, unit-testable."""
    access = os.environ.get("COMPRESS_MART_S3_ACCESS_KEY", "compress_mart")
    secret = os.environ.get("COMPRESS_MART_S3_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "COMPRESS_MART_S3_SECRET_KEY is not set; the S3 secret "
            "must come from the environment (no default password).")
    region = "us-east-1"
    now = datetime.now(timezone.utc)
    amzdate = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    host = url.split("://", 1)[1].split("/", 1)[0]
    path = "/" + url.split("://", 1)[1].split("/", 1)[1] if "/" in url.split("://", 1)[1] else "/"
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {"host": host, "x-amz-content-sha256": payload_hash,
               "x-amz-date": amzdate, "content-type": content_type}
    signed = ";".join(sorted(headers))
    canonical = "\n".join([method, path, "", "\n".join(f"{k}:{headers[k]}" for k in sorted(headers)), "", signed, payload_hash])
    scope = f"{datestamp}/{region}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k_date = _sign(("AWS4" + secret).encode(), datestamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, "s3")
    k_signing = _sign(k_service, "aws4_request")
    sig = hmac.new(k_signing, to_sign.encode(), hashlib.sha256).hexdigest()
    return {
        "Authorization": f"AWS4-HMAC-SHA256 Credential={access}/{scope}, SignedHeaders={signed}, Signature={sig}",
        "x-amz-date": amzdate, "x-amz-content-sha256": payload_hash,
        "Content-Type": content_type,
    }


def _request(method: str, key: str, body: bytes = b"", content_type: str = "application/octet-stream"):
    url = f"{endpoint()}/{bucket()}/{key}"
    headers = signature_headers(method, url, body, content_type)
    req = urllib.request.Request(url, data=body if method in ("PUT", "POST") else None,
                                 method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def make_bucket() -> int:
    url = f"{endpoint()}/{bucket()}"
    headers = signature_headers("PUT", url, b"")
    req = urllib.request.Request(url, data=b"", method="PUT", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def put_object(key: str, data: bytes) -> dict:
    status, _ = _request("PUT", key, data)
    return {"key": key, "status": status, "ok": status in (200, 201, 204)}


def sync_artifacts(store) -> dict:
    """Upload every CERTIFIED/PUBLISHED artifact tar to the bucket.
    Returns per-artifact status; a down endpoint gives 502 at the API
    layer, not a partial silent success. A tar file that is missing
    from disk raises FileNotFoundError (a server data problem, not
    a storage outage) so the API can report it as 500, not 502."""
    make_bucket()
    out = []
    for a in store.list_artifacts():
        if a["state"] not in ("CERTIFIED", "PUBLISHED"):
            continue
        tar_path = Path(a["file_path"])
        if not tar_path.exists():
            raise FileNotFoundError(
                f"artifact tar missing on disk for {a['id']}: {tar_path}")
        with open(tar_path, "rb") as f:
            data = f.read()
        r = put_object(f"{a['id']}.tar.gz", data)
        out.append({"artifact_id": a["id"], **r})
    return {"bucket": bucket(), "endpoint": endpoint(), "objects": out,
            "ok": all(o["ok"] for o in out)}


def status() -> dict:
    """Available = the endpoint answers at all. MinIO's health
    endpoint is tried first; a generic S3 endpoint answers the
    bucket GET instead (200/404/403 all prove it is up — only a
    transport failure means down)."""
    try:
        req = urllib.request.Request(f"{endpoint()}/minio/health/ready", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return {"available": resp.status == 200, "endpoint": endpoint(), "bucket": bucket()}
    except urllib.error.HTTPError:
        return {"available": True, "endpoint": endpoint(), "bucket": bucket()}
    except Exception:  # noqa: BLE001
        pass
    try:
        code, _ = _request("GET", "")
        return {"available": code is not None, "endpoint": endpoint(), "bucket": bucket()}
    except Exception:  # noqa: BLE001
        return {"available": False, "endpoint": endpoint(), "bucket": bucket()}
