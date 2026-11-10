"""cm — Compress-Mart CLI. Stdlib only, so it runs without the server deps.

Commands (matching the prompt, with --base-url for the local server):
  cm search --target rpi5 --task asr --max-size 200000
  cm pull <artifact-id> [--out file.tar.gz]
  cm verify <artifact.tar.gz> [--public-key key.pem]
  cm targets
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compress_mart.pipeline import verify_artifact_tar  # noqa: E402


def _get(base: str, path: str):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=30) as resp:
        return json.loads(resp.read())


def cmd_search(args):
    q = {}
    if args.target:
        q["target"] = args.target
    if args.task:
        q["task"] = args.task
    if args.max_size:
        q["max_size"] = args.max_size
    if args.state:
        q["state"] = args.state
    if args.q:
        q["q"] = args.q
    url = "/models" + ("?" + urllib.parse.urlencode(q) if q else "")
    results = _get(args.base_url, url)
    if not results:
        print("No artifacts match.")
        return 0
    for r in results:
        b = r.get("benchmark") or {}
        lat = (b.get("latency_ms") or {}).get("p95", "?")
        print(f"{r['id']}  {r['model_id']}  target={r['target']} task={r['task']} "
              f"tool={r['tool']} {r['bit_width']}bit {r['format']} "
              f"{r['size_bytes']}B state={r['state']} p95={lat}ms "
              f"quality_delta={(b.get('quality_delta'))}")
    return 0


def cmd_pull(args):
    url = args.base_url.rstrip("/") + f"/models/{args.artifact_id}/pull"
    out = Path(args.out or f"{args.artifact_id}.tar.gz")
    req = urllib.request.Request(url)
    if getattr(args, "token", None):
        req.add_header("X-API-Token", args.token)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as f:
            f.write(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"pull failed ({exc.code}): {exc.read().decode()}", file=sys.stderr)
        return 1
    print(f"Downloaded {out} ({out.stat().st_size} bytes)")
    return 0


def cmd_verify(args):
    trusted = Path(args.public_key).read_text() if args.public_key else None
    result = verify_artifact_tar(Path(args.artifact), trusted_public_pem=trusted)
    checks = result["checks"]
    print(f"SHA-256 {'OK' if checks['sha256_ok'] else 'FAIL'}")
    print(f"Provenance signature {'OK' if checks['provenance_signature_ok'] else 'FAIL'}"
          + ("" if trusted else " (self-consistent key from bundle; pass --public-key for trusted check)"))
    print(f"Benchmark card {'present' if checks['benchmark_card_present'] else 'MISSING'}")
    print(f"License receipt {'present' if checks['license_receipt_present'] else 'MISSING'}")
    if result.get("manifest"):
        print(f"Compression mode {result['manifest'].get('compression_mode')} "
              f"({result['manifest'].get('tool')}, {result['manifest'].get('format')})")
    if not result["ok"]:
        print(f"VERIFY FAILED: {result['reason']}", file=sys.stderr)
        return 1
    print(f"Verified artifact {result['artifact_id']} for model {result['model_id']}")
    return 0


def cmd_targets(args):
    for t in _get(args.base_url, "/targets"):
        print(f"{t['target_id']} ({t['category']}) {t['cpu']} "
              f"mem={t['memory_mb']}MB bits={t['recommended_bit_widths']} "
              f"p95_ceiling={t['latency_ceiling_ms_p95']}ms emulated={t['emulated']}")
    return 0


def cmd_feedback(args):
    rows = _get(args.base_url, "/router/feedback")
    if not rows:
        print("No router feedback yet.")
        return 0
    for r in rows:
        print(f"{r['tool']} on {r['target']} at {r['bit_width']}bit: "
              f"n={r['n']} pass_rate={r['pass_rate']} "
              f"avg_quality_delta={r['avg_quality_delta']} "
              f"avg_size_ratio={r['avg_size_ratio']} "
              f"recommendation={r['recommendation']}")
    return 0


def cmd_facets(args):
    print(json.dumps(_get(args.base_url, "/facets"), indent=2))
    return 0


def cmd_deploy(args):
    q = f"?runtime={args.runtime}" if args.runtime else ""
    d = _get(args.base_url, f"/models/{args.artifact_id}/deploy-snippet{q}")
    print(f"Runtime: {d['runtime']}  Format: {d['format']}  Mode: {d['compression_mode']}")
    print(d["snippet"])
    return 0


def cmd_trend(args):
    qs = []
    if args.model_id: qs.append(f"model_id={args.model_id}")
    if args.target: qs.append(f"target={args.target}")
    q = "?" + "&".join(qs) if qs else ""
    rows = _get(args.base_url, f"/benchmarks/trend{q}")
    if not rows:
        print("No benchmarks yet.")
        return 0
    for r in rows:
        print(f"{r['created_at']} {r['artifact_id']} {r['tool']} {r['bit_width']}bit "
              f"ratio={r['size_ratio']} delta={r['quality_delta']} "
              f"p95={r['latency_p95_ms']}ms passed={r['passed']}")
    return 0


def cmd_external(args):
    print(json.dumps(_get(args.base_url, "/external/status"), indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cm")
    p.add_argument("--base-url", default="http://127.0.0.1:8020")
    p.add_argument("--token", default=None, help="X-API-Token, when the server sets one")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--target"); s.add_argument("--task")
    s.add_argument("--max-size", type=int); s.add_argument("--state")
    s.add_argument("--q")
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("pull")
    s.add_argument("artifact_id"); s.add_argument("--out")
    s.set_defaults(fn=cmd_pull)
    s = sub.add_parser("verify")
    s.add_argument("artifact"); s.add_argument("--public-key")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("targets")
    s.set_defaults(fn=cmd_targets)
    s = sub.add_parser("feedback")
    s.set_defaults(fn=cmd_feedback)
    s = sub.add_parser("facets")
    s.set_defaults(fn=cmd_facets)
    s = sub.add_parser("deploy")
    s.add_argument("artifact_id"); s.add_argument("--runtime")
    s.set_defaults(fn=cmd_deploy)
    s = sub.add_parser("trend")
    s.add_argument("--model-id"); s.add_argument("--target")
    s.set_defaults(fn=cmd_trend)
    s = sub.add_parser("external")
    s.set_defaults(fn=cmd_external)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except urllib.error.HTTPError as exc:
        print(f"request failed ({exc.code}): {exc.read().decode()}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
