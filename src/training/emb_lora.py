"""EMB+LoRA: LoRA adapter + a new trainable output projection ("embedding layer").

What is trained
---------------
    LoRA adapters on the base model (attention projections, etc.)
  + a freshly-added ``nn.Linear(hidden_size, vocab_size, bias=False)`` whose
    output replaces the role of the base model's ``lm_head`` in the loss.

The original ``lm_head`` is left frozen — it's never used during the
wrapper's forward / backward. We seed ``embedding_layer.weight`` from it
at init so training starts from the pretrained output distribution.

Checkpoint layout
-----------------
    <checkpoint_dir>/adapter_config.json           (PEFT)
    <checkpoint_dir>/adapter_model.safetensors     (PEFT)
    <checkpoint_dir>/embedding_layer.safetensors   (this module)
    <checkpoint_dir>/tokenizer*                    (tokenizer)

Why inference doesn't need the wrapper
--------------------------------------
Llama's ``outputs.hidden_states[-1]`` is already post-final-norm, so
``embedding_layer(hidden_states[-1])`` equals
``lm_head'(hidden_states[-1])`` whenever ``lm_head'.weight == embedding_layer.weight``.
The benchmark loader therefore just copies ``embedding_layer.weight`` into
``model.lm_head.weight`` after attaching the PEFT adapter — the usual HF
``model.generate`` path then works with no wrapper at all.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


EMBEDDING_FILE_SAFETENSORS = "embedding_layer.safetensors"
EMBEDDING_FILE_PT = "embedding_layer.pt"


def _find_module_with_lm_head(model):
    """Return the submodule that actually owns ``lm_head``.

    Handles the common wrapping chain:
        PeftModel -> base_model (LoraModel) -> model (LlamaForCausalLM)
    as well as plain ``LlamaForCausalLM`` and post-``merge_and_unload``
    models.
    """
    for chain in ((), ("base_model",), ("base_model", "model"), ("model",)):
        cur = model
        ok = True
        for a in chain:
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok and hasattr(cur, "lm_head"):
            return cur
    raise RuntimeError(
        "Could not locate lm_head on the given model; emb+lora expects a "
        "CausalLM with an lm_head (optionally wrapped in PEFT)."
    )


class LlamaWithEmbeddingLayer(nn.Module):
    """Base (PEFT-adapted) CausalLM + a new trainable output projection.

    The wrapper's ``forward`` bypasses the model's own ``lm_head`` and
    projects the final hidden state through ``embedding_layer`` instead.
    Returns a :class:`CausalLMOutputWithPast` so HF ``Trainer`` and the
    generation mixin both work as-is.
    """

    def __init__(self, base_model, hidden_size: int, vocab_size: int):
        super().__init__()
        self.base_model = base_model
        self.embedding_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        self._init_embedding_from_lm_head()

    def _init_embedding_from_lm_head(self) -> None:
        inner = _find_module_with_lm_head(self.base_model)
        src = inner.lm_head.weight
        # nn.Linear defaults to CPU; match whatever device lm_head lives on
        # (device_map="auto" may place it on a specific GPU).
        self.embedding_layer = self.embedding_layer.to(device=src.device)
        with torch.no_grad():
            self.embedding_layer.weight.copy_(src.float())
        print(
            "[emb+lora] Initialized embedding_layer from lm_head "
            f"(shape={tuple(self.embedding_layer.weight.shape)}, "
            f"device={src.device}, dtype=fp32).",
            flush=True,
        )

    # Let HF Trainer / generation find things like ``.config``,
    # ``.gradient_checkpointing_enable()``, ``.generation_config``, etc.
    # on the inner model without us enumerating them all.
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        logits = self.embedding_layer(
            hidden_states.to(self.embedding_layer.weight.dtype)
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


def build_emb_lora_model(peft_wrapped_model) -> LlamaWithEmbeddingLayer:
    """Wrap a PEFT-adapted causal LM with the emb+lora head."""
    inner = _find_module_with_lm_head(peft_wrapped_model)
    hidden_size = int(inner.config.hidden_size)
    vocab_size = int(inner.config.vocab_size)
    return LlamaWithEmbeddingLayer(peft_wrapped_model, hidden_size, vocab_size)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_embedding_layer(
    wrapper: "LlamaWithEmbeddingLayer", output_dir: str
) -> str:
    """Persist ``embedding_layer`` weights next to the PEFT adapter.

    Prefers safetensors; falls back to a plain ``.pt`` file if the
    ``safetensors`` package is unavailable. Returns the written path.
    """
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "embedding_layer.weight": wrapper.embedding_layer.weight.detach().cpu().contiguous()
    }
    try:
        from safetensors.torch import save_file

        path = os.path.join(output_dir, EMBEDDING_FILE_SAFETENSORS)
        save_file(state, path)
    except ImportError:
        path = os.path.join(output_dir, EMBEDDING_FILE_PT)
        torch.save(state, path)
    return path


def find_embedding_layer_file(checkpoint_dir: str) -> Optional[str]:
    for name in (EMBEDDING_FILE_SAFETENSORS, EMBEDDING_FILE_PT):
        p = os.path.join(checkpoint_dir, name)
        if os.path.isfile(p):
            return p
    return None


def load_embedding_layer_weight(path: str) -> torch.Tensor:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu")
    if "embedding_layer.weight" not in state:
        raise KeyError(
            f"'embedding_layer.weight' not found in {path}; keys={list(state.keys())}"
        )
    return state["embedding_layer.weight"]


def apply_embedding_layer_to_lm_head(model, weight: torch.Tensor) -> None:
    """Copy a trained ``embedding_layer`` weight into the model's ``lm_head``.

    Equivalent to running with the wrapper at inference, because Llama's
    last hidden state is already post-final-norm:
        ``embedding_layer(h) == lm_head'(h)`` when ``lm_head'.weight = W``.
    """
    inner = _find_module_with_lm_head(model)
    with torch.no_grad():
        target = inner.lm_head.weight
        if tuple(target.shape) != tuple(weight.shape):
            raise ValueError(
                f"embedding_layer weight shape {tuple(weight.shape)} "
                f"does not match lm_head shape {tuple(target.shape)}"
            )
        target.copy_(weight.to(target.dtype).to(target.device))
