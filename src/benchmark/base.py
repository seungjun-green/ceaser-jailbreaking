"""Shared model loading + generation utilities for all benchmarks.

Design notes
------------
Both ``AutoModelForCausalLM.from_pretrained(path_or_repo_id)`` and
``PeftModel.from_pretrained(base_model, path_or_repo_id)`` accept **either**
a local directory or a Hugging Face Hub repository id transparently — the
underlying ``huggingface_hub`` layer checks the filesystem first and falls
back to downloading from the Hub when the path doesn't exist locally.

We therefore intentionally do NOT translate ``checkpoint_source`` into any
special loading logic; instead it acts as a **contract** / hint that's
logged up front so the caller can see what was resolved, and as a place
to key validation in the future (e.g. refuse to auto-download unless
``checkpoint_source == "hub"``).

The benchmark runner loads the model exactly ONCE (see ``runner.py``) and
all enabled benchmarks reuse the returned ``(model, tokenizer)`` pair.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _build_bnb_config(load_in_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    model_cfg: Dict[str, Any],
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load an evaluation-ready model + tokenizer.

    Expected ``model_cfg`` keys:
      * base_model (str)          — HF repo id of the base (required).
      * checkpoint_path (str)     — path *or* HF repo id of the finetuned
                                   model / adapter (optional; if missing
                                   we just use the base model).
      * checkpoint_source (str)   — "local" or "hub". Informational;
                                   see module docstring.
      * is_peft (bool)            — True => treat checkpoint as a PEFT
                                   adapter to attach to the base model.
                                   False => load checkpoint as a full model.
      * load_in_4bit (bool)       — Quantize weights with bitsandbytes.
    """
    base_model = model_cfg["base_model"]
    checkpoint_path = model_cfg.get("checkpoint_path")
    checkpoint_source = model_cfg.get("checkpoint_source", "local")
    is_peft = bool(model_cfg.get("is_peft", False))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))

    if checkpoint_source not in ("local", "hub"):
        raise ValueError(
            f"checkpoint_source must be 'local' or 'hub', got {checkpoint_source!r}"
        )
    # A friendly sanity check for "local"; the underlying loader would also
    # complain, but this error message is much clearer.
    if checkpoint_path and checkpoint_source == "local" and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint_source='local' but path does not exist: {checkpoint_path}"
        )

    print(
        f"[benchmark] Loading tokenizer from "
        f"{checkpoint_path if (checkpoint_path and not is_peft) else base_model}",
        flush=True,
    )
    # For PEFT adapters the adapter dir usually contains a tokenizer too,
    # but falling back to the base model's tokenizer is always safe.
    tok_src = checkpoint_path if (checkpoint_path and not is_peft) else base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    except Exception:  # noqa: BLE001
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # generation requires left padding

    bnb = _build_bnb_config(load_in_4bit)
    load_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if bnb is not None:
        load_kwargs["quantization_config"] = bnb
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    if is_peft:
        # Load base first, then attach the adapter.
        print(f"[benchmark] Loading base model: {base_model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        if checkpoint_path:
            from peft import PeftModel

            print(
                f"[benchmark] Attaching PEFT adapter "
                f"(source={checkpoint_source}): {checkpoint_path}",
                flush=True,
            )
            # PeftModel.from_pretrained accepts either a local path or a Hub repo id.
            model = PeftModel.from_pretrained(model, checkpoint_path)

            # Merge the adapter into the base weights for faster inference.
            # Default: enabled when not in 4-bit (merge into an NF4 layer is
            # lossy / memory-heavy). Controlled by model.merge_adapter.
            default_merge = not load_in_4bit
            merge_adapter = bool(model_cfg.get("merge_adapter", default_merge))
            if merge_adapter:
                try:
                    print("[benchmark] Merging LoRA/DoRA adapter into base weights…", flush=True)
                    model = model.merge_and_unload()
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[benchmark] WARN: merge_and_unload failed ({e!r}); "
                        "continuing with unmerged PeftModel.",
                        flush=True,
                    )
    else:
        target = checkpoint_path or base_model
        print(
            f"[benchmark] Loading full model (source={checkpoint_source}): {target}",
            flush=True,
        )
        # AutoModelForCausalLM.from_pretrained accepts either a local path or a Hub repo id.
        model = AutoModelForCausalLM.from_pretrained(target, **load_kwargs)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer


# ---------------------------------------------------------------------------
# Caesar cipher (shift-3 by default)
# ---------------------------------------------------------------------------
# The model was fine-tuned on Caesar-ciphered text (the `caesar_text` column
# of Seungjun/alpaca_ceaser), so every benchmark MUST encode prompts before
# sending them to the model and decode responses before scoring / judging.
# Non-alphabetic characters (digits, punctuation, whitespace, special tokens)
# are left untouched.
def caesar_encode(text: str, shift: int = 3) -> str:
    if not text or shift % 26 == 0:
        return text
    s = shift % 26
    out = []
    for c in text:
        if "a" <= c <= "z":
            out.append(chr((ord(c) - ord("a") + s) % 26 + ord("a")))
        elif "A" <= c <= "Z":
            out.append(chr((ord(c) - ord("A") + s) % 26 + ord("A")))
        else:
            out.append(c)
    return "".join(out)


def caesar_decode(text: str, shift: int = 3) -> str:
    return caesar_encode(text, -shift)


def caesar_encode_messages(
    messages: List[Dict[str, str]], shift: int = 3
) -> List[Dict[str, str]]:
    """Return a new list of messages with each ``content`` Caesar-encoded."""
    return [{"role": m["role"], "content": caesar_encode(m["content"], shift)} for m in messages]


def get_caesar_shift(config: Dict[str, Any]) -> int:
    """Read ``generation.caesar_shift`` with a default of 3."""
    return int((config.get("generation") or {}).get("caesar_shift", 3))


# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------
_ALPACA_HEADER = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
)


def _alpaca_format(messages: List[Dict[str, str]]) -> str:
    """Fallback prompt format when the tokenizer has no chat_template.

    Mirrors the classic Alpaca layout that the caesar_text column in
    Seungjun/alpaca_ceaser is built from, so the model sees exactly
    the format it was fine-tuned on.
    """
    parts = [_ALPACA_HEADER]
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user":
            parts.append(f"### Instruction:\n{content}\n\n")
        elif role == "assistant":
            parts.append(f"### Response:\n{content}\n\n")
        elif role == "system":
            # Prepend system content before the header.
            parts.insert(0, content + "\n\n")
    parts.append("### Response:\n")
    return "".join(parts)


def format_chat_prompts(
    tokenizer: PreTrainedTokenizerBase,
    messages_list: List[List[Dict[str, str]]],
) -> List[str]:
    """Format a batch of chat message-lists into prompt strings.

    Prefers ``tokenizer.apply_chat_template`` (e.g. Llama-3-Instruct's
    built-in template). If the tokenizer has no chat template (common
    for the *base* Llama-3.2 models), we fall back to an Alpaca-style
    wrapper that matches the training data's layout.
    """
    has_template = bool(getattr(tokenizer, "chat_template", None))
    prompts: List[str] = []
    for messages in messages_list:
        if has_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001
                prompt = _alpaca_format(messages)
        else:
            prompt = _alpaca_format(messages)
        prompts.append(prompt)
    return prompts


@torch.no_grad()
def batched_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    batch_size: int = 4,
    desc: str = "generating",
) -> List[str]:
    """Greedy / sampled batched generation.

    Returns the decoded continuation only (prompt is stripped).
    """
    outputs: List[str] = []
    device = next(model.parameters()).device
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature) if temperature > 0 else 1.0
    # When do_sample=False, temperature is ignored by HF.

    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)
        out = model.generate(**enc, **gen_kwargs)
        # Strip the prompt tokens from each output.
        input_len = enc["input_ids"].shape[1]
        gen_only = out[:, input_len:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        outputs.extend([d.strip() for d in decoded])
    return outputs


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def infer_model_tag(config: Dict[str, Any]) -> str:
    """Return a short tag for the model under evaluation.

    Resolution order:
      1. ``evaluation.model_tag`` in the config (explicit override).
      2. A ``\\d+[bBmM]`` size hint (e.g. "1B", "3B", "11B") extracted
         from ``model.checkpoint_path`` or ``model.base_model``.
      3. Sanitized basename of ``model.checkpoint_path`` or ``base_model``.
    """
    import re

    eval_cfg = config.get("evaluation", {}) or {}
    explicit = eval_cfg.get("model_tag")
    if explicit:
        return str(explicit)

    model_cfg = config.get("model", {}) or {}
    candidates = [
        model_cfg.get("checkpoint_path") or "",
        model_cfg.get("base_model") or "",
    ]
    for s in candidates:
        m = re.search(r"(\d+\.?\d*)\s*[bBmM]\b", s)
        if m:
            num = m.group(1).rstrip(".")
            unit = s[m.end() - 1].lower()
            return f"{num}{unit}"

    for s in candidates:
        if s:
            base = os.path.basename(s.rstrip("/")) or s
            return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "model"
    return "model"


def generations_csv_path(config: Dict[str, Any], benchmark_name: str) -> str:
    """Build ``{generations_dir}/{benchmark}_{model_tag}.csv``."""
    eval_cfg = config.get("evaluation", {}) or {}
    gen_dir = eval_cfg.get("generations_dir", "/content/")
    tag = infer_model_tag(config)
    return os.path.join(gen_dir, f"{benchmark_name}_{tag}.csv")


def write_generations_csv(
    path: str,
    rows: List[Dict[str, Any]],
    *,
    columns: Optional[List[str]] = None,
) -> None:
    """Write a generations CSV with a fixed schema.

    Default columns are ``prompt``, ``caesar_prompt``, ``response`` — any
    extra keys on ``rows`` are ignored unless ``columns`` is passed
    explicitly. Missing keys are written as empty cells. Directories are
    created as needed.
    """
    import csv

    cols = list(columns) if columns else ["prompt", "caesar_prompt", "response"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
