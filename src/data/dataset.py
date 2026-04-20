"""Dataset loading and preprocessing for instruction fine-tuning.

The training dataset lives on the Hugging Face Hub and contains a single
"text column" already formatted as an instruction/response sequence
(for the default ``Seungjun/alpaca_ceaser`` dataset this is the
``caesar_text`` column). We tokenize that column and train with standard
causal LM objective (labels == input_ids).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from datasets import Dataset, load_dataset


def _pick_base_split(ds) -> Dataset:
    """Pick the training split we want to work with."""
    if isinstance(ds, Dataset):
        return ds
    # datasets.DatasetDict
    for candidate in ("train", "training"):
        if candidate in ds:
            return ds[candidate]
    # Fall back to first split.
    first_key = next(iter(ds.keys()))
    return ds[first_key]


def load_instruction_dataset(
    dataset_cfg: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load, split, and tokenize the instruction dataset.

    Args:
        dataset_cfg: dict with keys ``hf_name``, ``text_column``,
            ``validation_split``.
        tokenizer: a Hugging Face tokenizer.
        max_seq_length: maximum token length; sequences are truncated.
        seed: RNG seed for the train/validation split.

    Returns:
        (train_dataset, eval_dataset) — both already tokenized, with
        ``input_ids``, ``attention_mask``, and ``labels`` columns.
    """
    hf_name = dataset_cfg["hf_name"]
    text_column = dataset_cfg.get("text_column", "text")
    val_split = float(dataset_cfg.get("validation_split", 0.1))

    raw = load_dataset(hf_name)
    base = _pick_base_split(raw)

    if text_column not in base.column_names:
        raise KeyError(
            f"text_column '{text_column}' not in dataset columns {base.column_names}"
        )

    # Deterministic train/val split.
    if val_split and val_split > 0.0:
        split = base.train_test_split(test_size=val_split, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = base, base.select(range(min(32, len(base))))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize(batch):
        out = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out

    keep_cols = set(train_ds.column_names)
    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=list(keep_cols))
    eval_ds = eval_ds.map(_tokenize, batched=True, remove_columns=list(keep_cols))

    return train_ds, eval_ds
