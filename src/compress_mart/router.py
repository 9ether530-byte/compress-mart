"""Tool-router: model type + target category -> compression stack.

Rules come from policies/router_policy.yaml. Matching is specific-first:
an exact model_type+category rule wins over an `any` rule.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "policies" / "router_policy.yaml"


def load_policy(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_POLICY
    with open(path) as f:
        return yaml.safe_load(f)["rules"]


def save_policy(rules: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"rules": rules}, f, sort_keys=False)


def apply_bit_width(rules: list[dict], rule_name: str, bit_width: int) -> dict:
    if bit_width not in (4, 8, 16, 32):
        raise ValueError("bit_width must be one of 4, 8, 16, 32")
    for rule in rules:
        if rule["name"] == rule_name:
            old = rule["bit_width"]
            rule["bit_width"] = bit_width
            return {"rule": rule_name, "old_bit_width": old, "new_bit_width": bit_width}
    raise KeyError(f"unknown rule: {rule_name}")


def _score(rule: dict, model_type: str, category: str) -> int:
    when = rule["when"]
    if when["category"] != "any" and when["category"] != category:
        return -1
    if when["model_type"] != "any" and when["model_type"] != model_type:
        return -1
    return (2 if when["model_type"] != "any" else 0) + (
        1 if when["category"] != "any" else 0
    )


def route(*, model_type: str, category: str, requested_bit_width: int | None = None,
          recommended_bit_widths: list[int] | None = None,
          policy: list[dict] | None = None) -> dict:
    rules = policy or load_policy()
    best, best_score = None, -1
    for rule in rules:
        score = _score(rule, model_type, category)
        if score > best_score:
            best, best_score = rule, score
    if best is None:
        raise ValueError(f"no router rule for {model_type}/{category}")
    bit_width = best["bit_width"]
    if requested_bit_width is not None:
        if recommended_bit_widths and requested_bit_width not in recommended_bit_widths:
            raise ValueError(
                f"bit width {requested_bit_width} not recommended for this target "
                f"(recommended: {recommended_bit_widths})"
            )
        bit_width = requested_bit_width
    return {
        "tool": best["tool"],
        "adapter": best["adapter"],
        "bit_width": bit_width,
        "format": best["format"],
        "sparsity": best["sparsity"],
        "rule": best["name"],
    }
