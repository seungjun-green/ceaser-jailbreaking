"""Config loading + override system.

Supports:
  * Loading YAML configs into plain dicts.
  * Applying keyword overrides to a config using either:
      - Short aliases (e.g. ``checkpoint_path`` -> ``model.checkpoint_path``)
      - Dotted paths (e.g. ``mmlu.n_shot`` -> ``config["mmlu"]["n_shot"]``)
  * Raising a clear ``ValueError`` when an unknown alias or dotted path is
    supplied.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Alias mapping: short keyword argument -> dotted path in the config tree.
# Anything that isn't a dotted path and isn't in this map is rejected.
# ---------------------------------------------------------------------------
OVERRIDE_ALIASES: Dict[str, str] = {
    # Model
    "checkpoint_path": "model.checkpoint_path",
    "checkpoint_source": "model.checkpoint_source",
    "is_peft": "model.is_peft",
    "base_model": "model.base_model",
    "load_in_4bit": "model.load_in_4bit",
    # Generation / evaluation
    "batch_size": "generation.batch_size",
    "output_dir": "evaluation.output_dir",
    # Benchmark toggles
    "run_hex_phi": "benchmarks.run_hex_phi",
    "run_ifeval": "benchmarks.run_ifeval",
    "run_mmlu": "benchmarks.run_mmlu",
    "run_mt_bench": "benchmarks.run_mt_bench",
}


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(cfg)}")
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Save a dict as a YAML file (directory is created if missing)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def _set_nested(cfg: Dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _key_exists(cfg: Dict[str, Any], dotted_path: str) -> bool:
    keys = dotted_path.split(".")
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    return True


def apply_overrides(
    cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Return a deep-copied config with overrides applied.

    Override keys may be:
      * A short alias listed in ``OVERRIDE_ALIASES``.
      * A dotted path (contains a ``.``). In strict mode we require the first
        segment to already exist in the config to guard against typos.

    An unknown alias (no dots, not in ``OVERRIDE_ALIASES``) always raises.
    """
    out = copy.deepcopy(cfg)
    for raw_key, value in overrides.items():
        if raw_key in OVERRIDE_ALIASES:
            dotted = OVERRIDE_ALIASES[raw_key]
        elif "." in raw_key:
            dotted = raw_key
            if strict:
                top = dotted.split(".", 1)[0]
                if top not in out:
                    raise ValueError(
                        f"Unknown override key '{raw_key}': top-level section "
                        f"'{top}' not found in config. Known sections: "
                        f"{sorted(out.keys())}"
                    )
        else:
            raise ValueError(
                f"Unknown override key '{raw_key}'. Use one of the aliases "
                f"{sorted(OVERRIDE_ALIASES)} or a dotted path like "
                f"'mmlu.n_shot'."
            )
        _set_nested(out, dotted, value)
    return out
