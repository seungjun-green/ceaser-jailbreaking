"""MMLU benchmark (5-shot, 57 subjects).

Two scoring modes:

  * ``"loglikelihood"``: for each test question we score the continuation
    " A", " B", " C", " D" and pick the argmax — the classic, deterministic
    evaluation used in the MMLU paper and in EleutherAI's harness.
  * ``"generation"``: generate a single next token and parse the first
    A/B/C/D character we see.

Metrics: per-subject accuracy + macro average.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from datasets import get_dataset_config_names, load_dataset
from tqdm.auto import tqdm

from .base import caesar_encode, get_caesar_shift, write_json


_CHOICES = ["A", "B", "C", "D"]


def _format_example(q: Dict[str, Any], include_answer: bool) -> str:
    choices = q["choices"]
    s = q["question"].strip() + "\n"
    for i, c in enumerate(choices):
        s += f"{_CHOICES[i]}. {c}\n"
        if i >= 3:
            break
    s += "Answer:"
    if include_answer:
        s += f" {_CHOICES[int(q['answer'])]}\n\n"
    return s


def _build_prompt(dev_examples: List[Dict[str, Any]], test_ex: Dict[str, Any], subject: str) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    shots = "".join(_format_example(e, include_answer=True) for e in dev_examples)
    return header + shots + _format_example(test_ex, include_answer=False)


@torch.no_grad()
def _loglik_answer(model, tokenizer, prompt: str) -> int:
    """Return the argmax choice index using next-token log-likelihoods.

    Prompt is Caesar-ciphered upstream, but the model's *answer letter*
    is plain English (that's what it was trained to emit), so we score
    " A", " B", " C", " D" directly.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to(device)
    out = model(**enc)
    logits = out.logits[0, -1]  # (vocab,)
    scores = []
    for ch in _CHOICES:
        ids = tokenizer(" " + ch, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tokenizer(ch, add_special_tokens=False)["input_ids"]
        token_id = ids[0]
        scores.append(logits[token_id].item())
    return int(max(range(4), key=lambda i: scores[i]))


@torch.no_grad()
def _generation_answer(model, tokenizer, prompt: str) -> int:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    for ch in gen.strip():
        if ch.upper() in _CHOICES:
            return _CHOICES.index(ch.upper())
    return 0


def _list_subjects(dataset_name: str, subset: str) -> List[str]:
    if subset and subset != "all":
        return [subset]
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:  # noqa: BLE001
        configs = []
    # Drop the "all"/"auxiliary_train" aggregate configs; we want the 57 subjects.
    configs = [c for c in configs if c not in ("all", "auxiliary_train")]
    if not configs:
        # Fall back to loading "all" and grouping by subject column.
        configs = ["_all"]
    return configs


def run(model, tokenizer, config: Dict[str, Any]) -> Dict[str, Any]:
    mmlu_cfg = config.get("mmlu", {})
    eval_cfg = config.get("evaluation", {})
    dataset_name = mmlu_cfg.get("dataset_name", "cais/mmlu")
    subset = mmlu_cfg.get("subset", "all")
    n_shot = int(mmlu_cfg.get("n_shot", 5))
    scoring_method = mmlu_cfg.get("scoring_method", "loglikelihood")
    max_samples: Optional[int] = mmlu_cfg.get("max_samples_per_subject", None)
    output_dir = eval_cfg.get("output_dir", "./benchmark_results")

    if scoring_method not in ("loglikelihood", "generation"):
        raise ValueError(f"mmlu.scoring_method must be one of loglikelihood/generation")

    # Caesar-encode the whole few-shot prompt; scorer also shifts A-D letters.
    shift = get_caesar_shift(config)

    subjects = _list_subjects(dataset_name, subset)
    print(
        f"[mmlu] Evaluating {len(subjects)} subjects "
        f"(scoring={scoring_method}, n_shot={n_shot}, caesar_shift={shift})",
        flush=True,
    )

    per_subject: Dict[str, float] = {}
    total_correct = 0
    total_count = 0

    for subject in tqdm(subjects, desc="mmlu/subjects"):
        if subject == "_all":
            dev = load_dataset(dataset_name, "all", split="dev")
            test = load_dataset(dataset_name, "all", split="test")
            subject_groups: Dict[str, List[Dict[str, Any]]] = {}
            for row in test:
                subject_groups.setdefault(row["subject"], []).append(row)
            dev_groups: Dict[str, List[Dict[str, Any]]] = {}
            for row in dev:
                dev_groups.setdefault(row["subject"], []).append(row)
            for subj, rows in subject_groups.items():
                dev_rows = dev_groups.get(subj, [])[:n_shot]
                if max_samples:
                    rows = rows[: int(max_samples)]
                correct = 0
                for q in tqdm(rows, desc=f"mmlu/{subj}", leave=False):
                    prompt = _build_prompt(dev_rows, q, subj)
                    if shift:
                        prompt = caesar_encode(prompt, shift)
                    pred = (
                        _loglik_answer(model, tokenizer, prompt)
                        if scoring_method == "loglikelihood"
                        else _generation_answer(model, tokenizer, prompt)
                    )
                    if pred == int(q["answer"]):
                        correct += 1
                acc = correct / max(1, len(rows))
                per_subject[subj] = acc
                total_correct += correct
                total_count += len(rows)
            continue

        try:
            dev = list(load_dataset(dataset_name, subject, split="dev"))
        except Exception:
            dev = []
        test = list(load_dataset(dataset_name, subject, split="test"))
        if max_samples:
            test = test[: int(max_samples)]
        dev = dev[:n_shot]

        correct = 0
        for q in tqdm(test, desc=f"mmlu/{subject}", leave=False):
            prompt = _build_prompt(dev, q, subject)
            if shift:
                prompt = caesar_encode(prompt, shift)
            pred = (
                _loglik_answer(model, tokenizer, prompt)
                if scoring_method == "loglikelihood"
                else _generation_answer(model, tokenizer, prompt)
            )
            if pred == int(q["answer"]):
                correct += 1
        acc = correct / max(1, len(test))
        per_subject[subject] = acc
        total_correct += correct
        total_count += len(test)

    macro_avg = sum(per_subject.values()) / max(1, len(per_subject))
    micro_avg = total_correct / max(1, total_count)
    results = {
        "scoring_method": scoring_method,
        "n_shot": n_shot,
        "num_subjects": len(per_subject),
        "num_questions": total_count,
        "per_subject_accuracy": per_subject,
        "macro_average": macro_avg,
        "micro_average": micro_avg,
    }
    write_json(os.path.join(output_dir, "mmlu_results.json"), results)
    return results
