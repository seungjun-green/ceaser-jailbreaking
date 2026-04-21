"""PEFT setup helpers (LoRA / DoRA / EMB+LoRA)."""
from __future__ import annotations

from typing import Any, Dict

from peft import LoraConfig, get_peft_model


# Methods that produce a PEFT-style checkpoint (adapter weights, possibly plus
# extra files). "emb+lora" builds a LoRA adapter AND wraps the model with an
# extra trainable output projection — see ``emb_lora.py``.
_PEFT_METHODS = {"lora", "dora", "emb+lora"}


def is_peft_method(method: str) -> bool:
    return method.lower() in _PEFT_METHODS


def build_peft_model(model, finetuning_cfg: Dict[str, Any]):
    """Wrap ``model`` with a LoRA/DoRA adapter, optionally plus an
    emb+lora head, according to ``finetuning_cfg``.

    For "full" fine-tuning this function should not be called — callers
    should gate with :func:`is_peft_method` first.
    """
    method = finetuning_cfg.get("method", "lora").lower()
    if method not in _PEFT_METHODS:
        raise ValueError(
            f"build_peft_model called with method={method!r}, expected one of {_PEFT_METHODS}"
        )
    lora_cfg = finetuning_cfg.get("lora", {})
    cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
        # DoRA = LoRA with weight-decomposition; emb+lora uses plain LoRA
        # underneath, with the extra projection layered on top.
        use_dora=(method == "dora"),
    )
    model = get_peft_model(model, cfg)

    if method == "emb+lora":
        from .emb_lora import build_emb_lora_model

        model = build_emb_lora_model(model)

    return model
