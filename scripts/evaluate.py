#!/usr/bin/env python
"""CLI wrapper around ``src.api.evaluate``.

Usage:
    python scripts/evaluate.py \
        --config configs/benchmark/default.yaml \
        --checkpoint-path ./outputs/checkpoint-500 \
        --checkpoint-source local \
        --set mmlu.n_shot=3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import evaluate  # noqa: E402


def _parse_set(items):
    import yaml

    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Run benchmark evaluation.")
    ap.add_argument(
        "--config",
        default="configs/benchmark/default.yaml",
        help="Path to benchmark YAML config.",
    )
    ap.add_argument(
        "--checkpoint-path",
        default=None,
        help="Override model.checkpoint_path (local dir or HF repo id).",
    )
    ap.add_argument(
        "--checkpoint-source",
        default=None,
        choices=[None, "local", "hub"],
        help="Override model.checkpoint_source.",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values: --set key=value (dotted paths allowed).",
    )
    args = ap.parse_args()

    overrides = _parse_set(args.set)
    result = evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        checkpoint_source=args.checkpoint_source,
        **overrides,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
