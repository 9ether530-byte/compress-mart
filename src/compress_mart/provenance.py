"""Provenance: Ed25519 signing, SHA-256, fail-closed verification.

The signing key lives only in the server data directory
(COMPRESS_MART_DATA/keys/). It is generated on first use and is never
returned by the API. A bundle carries the public key so a pulled artifact
can be checked offline; a check against a *trusted* public key (the server
pull path, or `cm verify --public-key`) must also match that key.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def load_or_create_keypair(key_dir: Path) -> tuple[Ed25519PrivateKey, str]:
    key_dir.mkdir(parents=True, exist_ok=True)
    priv_path = key_dir / "ed25519_private.pem"
    if priv_path.exists():
        try:
            private = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
        except Exception:
            # A corrupt key must not crash startup: quarantine it and
            # generate a fresh one (old artifacts signed with the lost
            # key will fail verification, as they should).
            quarantine = key_dir / f"ed25519_private.corrupt-{int(time.time())}.pem"
            priv_path.rename(quarantine)
            print(f"WARNING: corrupt signing key quarantined at {quarantine}; "
                  "generating a fresh keypair")
            private = None
    else:
        private = None
    if private is None:
        private = Ed25519PrivateKey.generate()
        # Write via a temp file created with 0600, then atomically
        # rename: the private key never exists with default perms.
        tmp_path = key_dir / f".ed25519_private.tmp-{os.getpid()}"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ))
        finally:
            os.close(fd)
        os.replace(tmp_path, priv_path)
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private, public_pem


def public_key_from_pem(pem: str) -> Ed25519PublicKey:
    return serialization.load_pem_public_key(pem.encode())


def sign_bundle(bundle_without_signature: dict, private: Ed25519PrivateKey) -> str:
    sig = private.sign(canonical_json(bundle_without_signature))
    return base64.b64encode(sig).decode()


def build_provenance(*, model_id: str, artifact_id: str, sha256: str,
                     license_name, license_status: str, upstream_url: str,
                     compression_manifest: dict, private, public_pem: str) -> dict:
    bundle = {
        "model_id": model_id,
        "artifact_id": artifact_id,
        "sha256": sha256,
        "license": license_name,
        "license_status": license_status,
        "upstream_url": upstream_url,
        "compression_manifest": compression_manifest,
        "signing_public_key": public_pem,
        "algorithm": "Ed25519",
    }
    bundle["signature"] = sign_bundle(bundle, private)
    return bundle


def verify_bundle(bundle: dict, trusted_public_pem: str | None = None) -> tuple[bool, str]:
    """Return (ok, reason). Fail-closed: any mismatch, missing field or
    bad signature is a failure, never a warning."""
    try:
        signature = base64.b64decode(bundle["signature"])
        public_pem = bundle["signing_public_key"]
        if trusted_public_pem is not None and public_pem != trusted_public_pem:
            return False, "signing public key does not match trusted key"
        unsigned = {k: v for k, v in bundle.items() if k != "signature"}
        public_key_from_pem(public_pem).verify(signature, canonical_json(unsigned))
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - verification must never raise
        return False, f"invalid provenance signature: {exc}"
