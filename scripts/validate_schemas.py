#!/usr/bin/env python3
"""Validate that every JSON schema in schemas/ parses and has required keys."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ok = True
for path in sorted((root / "schemas").glob("*.json")):
    try:
        data = json.loads(path.read_text())
        assert "properties" in data and "required" in data, "missing properties/required"
        print(f"OK {path.name}")
    except Exception as exc:
        print(f"FAIL {path.name}: {exc}")
        ok = False
sys.exit(0 if ok else 1)
