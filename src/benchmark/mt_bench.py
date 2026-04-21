"""MT-Bench: 80 multi-turn questions, 8 categories, GPT-4 single-answer grading.

We pull the official question set from ``lmsys/mt_bench_human_judgments``
(a convenient mirror that includes ``question_1``, ``question_2``,
``category`` per item). The model generates two turns and GPT-4 scores
each turn 1-10 with the official single-answer grading prompt.

Metrics: overall average, per-category average, turn-1 / turn-2 averages.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm.auto import tqdm

from ._judge import extract_first_int, judge_chat
from .base import (
    batched_generate,
    caesar_encode,
    format_chat_prompts,
    get_caesar_shift,
    write_json,
    write_jsonl,
)


# Official MT-Bench single-answer grading judge prompt (from lmsys/FastChat).
_JUDGE_SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your "
    "evaluation should consider factors such as the helpfulness, relevance, "
    "accuracy, depth, creativity, and level of detail of the response. Begin "
    "your evaluation by providing a short explanation. Be as objective as "
    "possible. After providing your explanation, please rate the response on "
    "a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for "
    "example: \"Rating: [[5]]\"."
)

_JUDGE_USER_SINGLE = (
    "[Question]\n{question}\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n"
    "[The End of Assistant's Answer]"
)


def _load_questions(questions_path: str) -> List[Dict[str, Any]]:
    """Return a list of {question_id, category, turns: [t1, t2]} items.

    We dedupe on ``question_id`` so we end up with the canonical 80.
    """
    ds = load_dataset(questions_path, split="human")
    seen: Dict[int, Dict[str, Any]] = {}
    for row in ds:
        qid = row.get("question_id")
        if qid is None or qid in seen:
            continue
        turns = None
        # Different schemas use different field names.
        if "turns" in row and isinstance(row["turns"], list):
            turns = row["turns"]
        elif "question_1" in row:
            turns = [row.get("question_1"), row.get("question_2")]
        elif "conversation_a" in row and isinstance(row["conversation_a"], list):
            turns = [m["content"] for m in row["conversation_a"] if m.get("role") == "user"]
        if not turns:
            continue
        seen[qid] = {
            "question_id": qid,
            "category": row.get("category", "unknown"),
            "turns": turns[:2],
        }
    items = list(seen.values())
    items.sort(key=lambda x: x["question_id"])
    return items


def _extract_rating(text: str) -> int:
    """Parse the official "[[N]]" pattern from a judge response."""
    import re

    m = re.search(r"\[\[\s*(\d+)\s*\]\]", text)
    if m:
        try:
            v = int(m.group(1))
            if 1 <= v <= 10:
                return v
        except ValueError:
            pass
    # Fallback: any integer 1-10 in the text.
    v = extract_first_int(text, 1, 10)
    return v if v is not None else 1


def run(model, tokenizer, config: Dict[str, Any]) -> Dict[str, Any]:
    mt_cfg = config.get("mt_bench", {})
    gen_cfg = config.get("generation", {})
    eval_cfg = config.get("evaluation", {})
    questions_path = mt_cfg.get("questions_path", "lmsys/mt_bench_human_judgments")
    judge_model = mt_cfg.get("judge_model", "gpt-4")
    num_rounds = int(mt_cfg.get("num_rounds", 2))
    gen_override = mt_cfg.get("generation_override", {}) or {}
    output_dir = eval_cfg.get("output_dir", "./benchmark_results")
    save_generations = bool(eval_cfg.get("save_generations", True))

    # MT-Bench uses sampling by design (temperature 0.7); apply the override.
    max_new_tokens = int(gen_override.get("max_new_tokens", gen_cfg.get("max_new_tokens", 1024)))
    temperature = float(gen_override.get("temperature", 0.7))
    do_sample = bool(gen_override.get("do_sample", True))
    batch_size = int(gen_cfg.get("batch_size", 4))

    questions = _load_questions(questions_path)
    print(f"[mt_bench] Loaded {len(questions)} questions", flush=True)

    # Run the conversation turn-by-turn across the whole batch so each turn
    # is a single batched generate call. User messages are Caesar-ciphered
    # (matches training); assistant messages come back as plain English and
    # are stored as-is — this mirrors the training distribution for multi-turn.
    shift = get_caesar_shift(config)
    conversations: List[List[Dict[str, str]]] = [[] for _ in questions]
    all_turn_outputs: List[List[str]] = [[] for _ in questions]

    for turn_idx in range(num_rounds):
        active_indices = [
            i
            for i, q in enumerate(questions)
            if turn_idx < len(q["turns"]) and q["turns"][turn_idx]
        ]
        if not active_indices:
            break
        for i in active_indices:
            conversations[i].append(
                {
                    "role": "user",
                    "content": caesar_encode(questions[i]["turns"][turn_idx], shift),
                }
            )
        prompts = format_chat_prompts(tokenizer, [conversations[i] for i in active_indices])
        outs = batched_generate(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            batch_size=batch_size,
            desc=f"mt_bench/turn{turn_idx + 1}",
        )
        for i, resp in zip(active_indices, outs):
            conversations[i].append({"role": "assistant", "content": resp})
            all_turn_outputs[i].append(resp)

    # Judge each turn independently using the official single-answer grading prompt.
    turn_scores: List[List[int]] = [[] for _ in questions]
    gen_rows: List[Dict[str, Any]] = []
    for i, q in enumerate(tqdm(questions, desc="mt_bench/judge")):
        per_turn = []
        for t_idx, answer in enumerate(all_turn_outputs[i]):
            judge_out = judge_chat(
                _JUDGE_SYSTEM,
                _JUDGE_USER_SINGLE.format(question=q["turns"][t_idx], answer=answer),
                model=judge_model,
                temperature=0.0,
            )
            per_turn.append({"turn": t_idx + 1, "rating": _extract_rating(judge_out), "judge_raw": judge_out})
        turn_scores[i] = [r["rating"] for r in per_turn]
        gen_rows.append(
            {
                "question_id": q["question_id"],
                "category": q["category"],
                "turns": q["turns"],
                "responses": all_turn_outputs[i],
                "judgments": per_turn,
            }
        )

    # Aggregate.
    per_cat: Dict[str, List[float]] = {}
    all_means: List[float] = []
    turn1: List[int] = []
    turn2: List[int] = []
    for q, scores in zip(questions, turn_scores):
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        per_cat.setdefault(q["category"], []).append(mean)
        all_means.append(mean)
        if len(scores) >= 1:
            turn1.append(scores[0])
        if len(scores) >= 2:
            turn2.append(scores[1])

    results = {
        "num_questions": len(questions),
        "overall_avg": sum(all_means) / max(1, len(all_means)),
        "turn_1_avg": sum(turn1) / max(1, len(turn1)) if turn1 else None,
        "turn_2_avg": sum(turn2) / max(1, len(turn2)) if turn2 else None,
        "per_category_avg": {c: sum(v) / len(v) for c, v in per_cat.items()},
    }

    if save_generations:
        write_jsonl(os.path.join(output_dir, "mt_bench_generations.jsonl"), gen_rows)
    write_json(os.path.join(output_dir, "mt_bench_results.json"), results)
    return results
