#!/usr/bin/env python
"""CLI wrapper around ``src.api.train``.

Usage:
    python scripts/train.py --config configs/train/llama3_2_1b_lora.yaml \
        --set training.epochs=1 --set training.batch_size=2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make `src` importable when running the script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import train  # noqa: E402


def _parse_set(items):
    """Parse ``key=value`` pairs from --set. Values are YAML-parsed so you
    can pass ints, floats, booleans, lists, etc."""
    import yaml

    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Instruction fine-tune a Llama-3.2 model.")
    ap.add_argument("--config", required=True, help="Path to training YAML config.")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values: --set key=value (dotted paths allowed).",
    )
    args = ap.parse_args()

    overrides = _parse_set(args.set)
    result = train(args.config, **overrides)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
