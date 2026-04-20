"""Benchmark runner — loads the model once and dispatches to enabled benchmarks."""
from __future__ import annotations

import os
import time
from typing import Any, Dict

from ..utils.config import save_config
from .base import load_model_and_tokenizer, write_json


def _maybe_run(name: str, enabled: bool, fn, *args, **kwargs) -> Dict[str, Any]:
    if not enabled:
        return {"skipped": True}
    print(f"\n===== Running benchmark: {name} =====", flush=True)
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] ERROR: {e!r}", flush=True)
        return {"error": repr(e)}
    dt = time.time() - t0
    if isinstance(out, dict):
        out = {**out, "elapsed_sec": dt}
    print(f"[{name}] done in {dt:.1f}s", flush=True)
    return out


def run_benchmarks(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run all enabled benchmarks against a single loaded model."""
    eval_cfg = cfg.get("evaluation", {})
    output_dir = eval_cfg.get("output_dir", "./benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    # Persist the resolved config for reproducibility.
    save_config(cfg, os.path.join(output_dir, "benchmark_config.yaml"))

    # Import here so the benchmark modules aren't required for `train()`.
    from . import hex_phi, ifeval, mmlu, mt_bench

    bench_cfg = cfg.get("benchmarks", {})
    model, tokenizer = load_model_and_tokenizer(cfg["model"])

    summary: Dict[str, Any] = {}
    summary["hex_phi"] = _maybe_run(
        "hex_phi", bool(bench_cfg.get("run_hex_phi", True)), hex_phi.run, model, tokenizer, cfg
    )
    summary["ifeval"] = _maybe_run(
        "ifeval", bool(bench_cfg.get("run_ifeval", True)), ifeval.run, model, tokenizer, cfg
    )
    summary["mmlu"] = _maybe_run(
        "mmlu", bool(bench_cfg.get("run_mmlu", True)), mmlu.run, model, tokenizer, cfg
    )
    summary["mt_bench"] = _maybe_run(
        "mt_bench",
        bool(bench_cfg.get("run_mt_bench", True)),
        mt_bench.run,
        model,
        tokenizer,
        cfg,
    )

    # Unified summary.
    write_json(os.path.join(output_dir, "summary.json"), summary)
    print(f"\n[benchmark] Summary written to {output_dir}/summary.json", flush=True)
    return summary
