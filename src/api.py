"""Top-level Python API: ``train()`` and ``evaluate()``.

This module is the single source of truth for both the notebook usage
pattern and the CLI wrappers in ``scripts/``. Everything below is kept
deliberately thin — YAML loading and override resolution happen in
:mod:`src.utils.config`, and the actual work happens in
:mod:`src.training.trainer` and :mod:`src.benchmark.runner`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .utils.config import apply_overrides, load_config


def train(config_path: str, **overrides: Any) -> Dict[str, Any]:
    """Run instruction fine-tuning.

    Args:
        config_path: Path to a training YAML config.
        **overrides: Any config value can be overridden via keyword args using
            one of the aliases in ``OVERRIDE_ALIASES`` or a dotted path
            (e.g. ``training_epochs`` has no alias — use
            ``**{"training.epochs": 5}``).

    Returns:
        A dict with the training summary (final loss, checkpoint paths, etc.).
    """
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, overrides)
    # Imported lazily so `import src` stays cheap.
    from .training.trainer import run_training

    return run_training(cfg)


def evaluate(
    config_path: str = "configs/benchmark/default.yaml",
    checkpoint_path: Optional[str] = None,
    checkpoint_source: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run benchmark evaluation.

    Args:
        config_path: Path to a benchmark YAML config (has sensible defaults).
        checkpoint_path: If provided, OVERRIDES ``model.checkpoint_path`` in
            the config. Can be a local directory or a Hugging Face repo id.
        checkpoint_source: If provided, OVERRIDES ``model.checkpoint_source``
            in the config. Must be ``"local"`` or ``"hub"``.
        **overrides: Any other config value can be overridden via keyword args
            (aliases or dotted paths, e.g. ``is_peft=True``,
            ``**{"mmlu.n_shot": 3}``).

    Returns:
        A dict keyed by benchmark name with the scores + metadata for each.
    """
    cfg = load_config(config_path)
    # Promote the two convenience kwargs to standard overrides so they flow
    # through the same validation as everything else.
    merged: Dict[str, Any] = dict(overrides)
    if checkpoint_path is not None:
        merged["checkpoint_path"] = checkpoint_path
    if checkpoint_source is not None:
        merged["checkpoint_source"] = checkpoint_source
    cfg = apply_overrides(cfg, merged)

    from .benchmark.runner import run_benchmarks

    return run_benchmarks(cfg)
