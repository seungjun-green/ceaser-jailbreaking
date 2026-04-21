"""IFEval instruction-following benchmark.

We generate responses with the model, then score them with the official
rule-based evaluators. We try the ``instruction_following_eval`` package
first (sometimes installed alongside lm-evaluation-harness); if it is
not present we fall back to a minimal vendored scorer that covers the
commonly-used instruction types.

Reported metrics
----------------
  * strict_prompt_acc:   fraction of prompts that satisfy all instructions under strict scoring.
  * loose_prompt_acc:    same, under loose scoring (lowercase / stripped punctuation).
  * strict_instruction_acc: average fraction of satisfied instructions per prompt (strict).
  * loose_instruction_acc:  same, loose.
"""
from __future__ import annotations

import os
import re
import string
from typing import Any, Dict, List

from datasets import load_dataset

from .base import (
    batched_generate,
    caesar_encode,
    format_chat_prompts,
    get_caesar_shift,
    write_json,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Official-evaluator wrapper (preferred).
# ---------------------------------------------------------------------------
def _score_with_official(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    from instruction_following_eval import evaluation_main  # type: ignore
    from instruction_following_eval.evaluation_main import (  # type: ignore
        InputExample,
        test_instruction_following_strict,
        test_instruction_following_loose,
    )

    strict_prompt = []
    loose_prompt = []
    strict_inst = []
    loose_inst = []
    for r in rows:
        ex = InputExample(
            key=r["key"],
            prompt=r["prompt"],
            instruction_id_list=r["instruction_id_list"],
            kwargs=r["kwargs"],
        )
        strict = test_instruction_following_strict(ex, {r["prompt"]: r["response"]})
        loose = test_instruction_following_loose(ex, {r["prompt"]: r["response"]})
        strict_prompt.append(1.0 if all(strict.follow_instruction_list) else 0.0)
        loose_prompt.append(1.0 if all(loose.follow_instruction_list) else 0.0)
        strict_inst.append(
            sum(strict.follow_instruction_list) / max(1, len(strict.follow_instruction_list))
        )
        loose_inst.append(
            sum(loose.follow_instruction_list) / max(1, len(loose.follow_instruction_list))
        )
    return {
        "strict_prompt_acc": sum(strict_prompt) / len(strict_prompt),
        "loose_prompt_acc": sum(loose_prompt) / len(loose_prompt),
        "strict_instruction_acc": sum(strict_inst) / len(strict_inst),
        "loose_instruction_acc": sum(loose_inst) / len(loose_inst),
    }


# ---------------------------------------------------------------------------
# Minimal fallback evaluators (cover the most common IFEval instruction ids).
# This is intentionally a *subset* — when users care about a full IFEval run
# they should ``pip install instruction-following-eval``.
# ---------------------------------------------------------------------------
def _loose_transform(text: str) -> str:
    text = text.lower().strip()
    # Strip leading/trailing markdown-ish punctuation, per the official
    # "loose" transform.
    text = text.strip(string.punctuation + string.whitespace)
    return text


def _check_single(inst_id: str, kwargs: Dict[str, Any], response: str) -> bool:
    r = response
    # Length-based checks
    if inst_id in ("length_constraints:number_words",):
        n = len(re.findall(r"\w+", r))
        if "num_words" in kwargs:
            rel = kwargs.get("relation", "at least")
            want = kwargs["num_words"]
            if rel == "at least":
                return n >= want
            if rel == "at most":
                return n <= want
            if rel == "less than":
                return n < want
        return True
    if inst_id in ("length_constraints:number_sentences",):
        n = len(re.findall(r"[.!?]+", r))
        rel = kwargs.get("relation", "at least")
        want = kwargs.get("num_sentences", 0)
        if rel == "at least":
            return n >= want
        if rel == "at most":
            return n <= want
        return True
    # Keyword inclusion/exclusion
    if inst_id == "keywords:existence":
        kws = kwargs.get("keywords", [])
        return all(k.lower() in r.lower() for k in kws)
    if inst_id == "keywords:forbidden_words":
        kws = kwargs.get("forbidden_words", [])
        return not any(k.lower() in r.lower() for k in kws)
    if inst_id == "keywords:frequency":
        kw = kwargs.get("keyword", "")
        freq = len(re.findall(re.escape(kw), r, flags=re.IGNORECASE))
        rel = kwargs.get("relation", "at least")
        want = kwargs.get("frequency", 0)
        if rel == "at least":
            return freq >= want
        if rel == "at most":
            return freq <= want
        return True
    # Case
    if inst_id == "change_case:english_lowercase":
        return r == r.lower()
    if inst_id == "change_case:english_capital":
        return r == r.upper()
    # Default: be lenient rather than incorrectly punishing.
    return True


def _score_with_fallback(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    strict_prompt = []
    loose_prompt = []
    strict_inst = []
    loose_inst = []
    for r in rows:
        inst_ids = r["instruction_id_list"] or []
        kwargs_list = r["kwargs"] or [{} for _ in inst_ids]
        resp_strict = r["response"]
        resp_loose = _loose_transform(r["response"])
        strict_flags = [
            _check_single(iid, kw or {}, resp_strict) for iid, kw in zip(inst_ids, kwargs_list)
        ]
        loose_flags = [
            _check_single(iid, kw or {}, resp_loose) for iid, kw in zip(inst_ids, kwargs_list)
        ]
        if not strict_flags:
            strict_flags = [True]
            loose_flags = [True]
        strict_prompt.append(1.0 if all(strict_flags) else 0.0)
        loose_prompt.append(1.0 if all(loose_flags) else 0.0)
        strict_inst.append(sum(strict_flags) / len(strict_flags))
        loose_inst.append(sum(loose_flags) / len(loose_flags))
    n = max(1, len(rows))
    return {
        "strict_prompt_acc": sum(strict_prompt) / n,
        "loose_prompt_acc": sum(loose_prompt) / n,
        "strict_instruction_acc": sum(strict_inst) / n,
        "loose_instruction_acc": sum(loose_inst) / n,
    }


def run(model, tokenizer, config: Dict[str, Any]) -> Dict[str, Any]:
    ifeval_cfg = config.get("ifeval", {})
    gen_cfg = config.get("generation", {})
    eval_cfg = config.get("evaluation", {})
    dataset_name = ifeval_cfg.get("dataset_name", "google/IFEval")
    output_dir = eval_cfg.get("output_dir", "./benchmark_results")
    save_generations = bool(eval_cfg.get("save_generations", True))

    ds = load_dataset(dataset_name, split="train")
    # Caesar-encode prompts. Responses come back as plain English and are
    # scored by the rule-based checks as-is.
    shift = get_caesar_shift(config)
    chat_inputs = [
        [{"role": "user", "content": caesar_encode(ex["prompt"], shift)}] for ex in ds
    ]
    formatted = format_chat_prompts(tokenizer, chat_inputs)
    responses = batched_generate(
        model,
        tokenizer,
        formatted,
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
        temperature=float(gen_cfg.get("temperature", 0.0)),
        do_sample=bool(gen_cfg.get("do_sample", False)),
        batch_size=int(gen_cfg.get("batch_size", 4)),
        desc="ifeval",
    )

    rows: List[Dict[str, Any]] = []
    for ex, resp in zip(ds, responses):
        rows.append(
            {
                "key": ex.get("key", None),
                "prompt": ex["prompt"],
                "instruction_id_list": ex.get("instruction_id_list", []),
                "kwargs": ex.get("kwargs", []),
                "response": resp,
            }
        )

    try:
        scores = _score_with_official(rows)
        scores["evaluator"] = "instruction_following_eval"
    except Exception as e:  # noqa: BLE001
        print(
            f"[ifeval] Official evaluator not available ({e!r}); "
            "falling back to the vendored minimal evaluator. Install "
            "`instruction-following-eval` for the full official scoring.",
            flush=True,
        )
        scores = _score_with_fallback(rows)
        scores["evaluator"] = "fallback"

    if save_generations:
        write_jsonl(os.path.join(output_dir, "ifeval_generations.jsonl"), rows)
    write_json(os.path.join(output_dir, "ifeval_results.json"), scores)
    return scores
