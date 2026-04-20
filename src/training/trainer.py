"""Instruction fine-tuning trainer.

We use Hugging Face :class:`transformers.Trainer` and layer on:

  * Per-step loss logging to stdout and to ``{output_dir}/train_log.txt``.
  * A checkpoint+eval callback that fires ``save_steps_per_epoch`` times
    per epoch, computing validation loss and saving the appropriate
    artifacts (adapter only for PEFT, full model for "full" fine-tuning).
  * A copy of the resolved config written into ``{output_dir}/config.yaml``.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from ..data.dataset import load_instruction_dataset
from ..utils.config import save_config
from .peft_setup import build_peft_model, is_peft_method


class _StepLossLogger(TrainerCallback):
    """Training/eval loss logger.

    * A single tqdm progress bar is updated in-place every training step,
      showing the latest training loss as its postfix (no growing table).
    * Evaluation events (``eval_loss``) are printed as normal newline
      messages via ``tqdm.write`` so the progress bar stays intact.
    * Every event (training step and eval) is appended to
      ``{output_dir}/train_log.txt`` verbatim for post-hoc analysis.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        open(self.log_path, "w").close()
        self._bar = None

    def _append(self, line: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        from tqdm.auto import tqdm

        total = max(1, int(state.max_steps or 0))
        self._bar = tqdm(total=total, desc="train", dynamic_ncols=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any] = None,
        **kwargs,
    ):
        if not logs:
            return
        step = state.global_step
        is_eval = any(k.startswith("eval_") for k in logs)
        parts = [f"step={step}"]
        for k in ("loss", "eval_loss", "learning_rate", "epoch"):
            if k in logs:
                parts.append(
                    f"{k}={logs[k]:.6f}" if isinstance(logs[k], float) else f"{k}={logs[k]}"
                )
        line = " ".join(parts)
        self._append(line)

        if is_eval:
            # Emit eval lines above the progress bar so they scroll normally.
            from tqdm.auto import tqdm

            tqdm.write(line)
            return

        # Training step: update bar in place.
        if self._bar is not None:
            postfix = {}
            if "loss" in logs:
                postfix["loss"] = f"{logs['loss']:.4f}"
            if "learning_rate" in logs:
                postfix["lr"] = f"{logs['learning_rate']:.2e}"
            if "epoch" in logs:
                postfix["epoch"] = f"{logs['epoch']:.2f}"
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            # Advance the bar to the current global step.
            delta = step - self._bar.n
            if delta > 0:
                self._bar.update(delta)

    def on_train_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class _EvalAndSaveCallback(TrainerCallback):
    """Run eval + save a checkpoint at ``save_steps_per_epoch`` points per epoch."""

    def __init__(self, save_steps_per_epoch: int, is_peft: bool):
        self.save_steps_per_epoch = max(1, int(save_steps_per_epoch))
        self.is_peft = is_peft
        self._fire_steps: set[int] = set()

    def on_train_begin(self, args, state, control, **kwargs):
        total = state.max_steps
        steps_per_epoch = max(1, math.ceil(total / max(1, args.num_train_epochs)))
        chunk = max(1, steps_per_epoch // self.save_steps_per_epoch)
        self._fire_steps = {
            min(total, chunk * (i + 1)) for i in range(self.save_steps_per_epoch)
        }
        # Always fire at the last step so we have a final checkpoint.
        self._fire_steps.add(total)

    def on_step_end(self, args, state, control: TrainerControl, **kwargs):
        if state.global_step in self._fire_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


class _EarlyStoppingCallback(TrainerCallback):
    """Stop training when validation loss stops improving.

    Triggered by whatever drives evaluation — here, :class:`_EvalAndSaveCallback`.
    We look at ``eval_loss`` from each evaluation round, keep the best seen so
    far, and stop when ``patience`` consecutive rounds fail to improve on it.

    The stopping signal is written via ``control.should_training_stop`` so the
    Trainer exits cleanly after the current step. A summary is stored on the
    callback for the training summary dict.
    """

    def __init__(self, patience: int, log_path: str | None = None):
        self.patience = int(patience)
        self.log_path = log_path
        self.best: float | None = None
        self.best_step: int | None = None
        self.bad_rounds = 0
        self.stopped_early = False
        self.stop_step: int | None = None

    def _log(self, msg: str) -> None:
        print(msg, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(msg + "\n")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any] = None,
        **kwargs,
    ):
        if not metrics or "eval_loss" not in metrics:
            return control
        cur = float(metrics["eval_loss"])
        step = state.global_step
        if self.best is None or cur < self.best:
            self.best = cur
            self.best_step = step
            self.bad_rounds = 0
            self._log(
                f"[early_stopping] step={step} eval_loss={cur:.6f} (new best)"
            )
        else:
            self.bad_rounds += 1
            self._log(
                f"[early_stopping] step={step} eval_loss={cur:.6f} "
                f"(no improvement; {self.bad_rounds}/{self.patience})"
            )
            if self.bad_rounds >= self.patience:
                self.stopped_early = True
                self.stop_step = step
                control.should_training_stop = True
                self._log(
                    f"[early_stopping] patience exhausted at step={step}; "
                    f"best eval_loss={self.best:.6f} @ step={self.best_step}"
                )
        return control


def _build_bnb_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def run_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run instruction fine-tuning based on a resolved config dict.

    Returns a small summary dict with final/best losses, output directory,
    and the list of saved checkpoint paths.
    """
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    ft_cfg = cfg["finetuning"]
    train_cfg = cfg["training"]

    output_dir = train_cfg.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Persist the resolved config for reproducibility.
    save_config(cfg, os.path.join(output_dir, "config.yaml"))

    method = ft_cfg.get("method", "lora").lower()
    is_peft = is_peft_method(method)
    load_in_4bit = bool(train_cfg.get("load_in_4bit", False))
    max_seq_length = int(train_cfg.get("max_seq_length", 512))

    model_name = model_cfg["name"]
    print(f"[train] Loading tokenizer + model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = _build_bnb_config(load_in_4bit)
    model_kwargs: Dict[str, Any] = {}
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if is_peft:
        if load_in_4bit:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        model = build_peft_model(model, ft_cfg)
        model.print_trainable_parameters()

    print("[train] Loading dataset…", flush=True)
    train_ds, eval_ds = load_instruction_dataset(
        dataset_cfg, tokenizer, max_seq_length=max_seq_length
    )
    print(f"[train] train_size={len(train_ds)} eval_size={len(eval_ds)}", flush=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build TrainingArguments in a version-tolerant way: transformers renamed
    # `evaluation_strategy` -> `eval_strategy` in 4.46, and older / newer
    # versions accept different optional kwargs. We introspect the signature
    # to stay compatible with whatever Colab ships.
    import inspect

    _ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    ta_kwargs: Dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg.get("epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 4)),
        per_device_eval_batch_size=int(train_cfg.get("batch_size", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to=[],
        bf16=not load_in_4bit,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        # Kill the built-in progress table — we use our own tqdm bar that
        # updates the training loss in-place instead of growing a table.
        disable_tqdm=True,
    )
    # Eval is driven by our callback; use whichever kwarg exists.
    if "eval_strategy" in _ta_params:
        ta_kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in _ta_params:
        ta_kwargs["evaluation_strategy"] = "no"
    args = TrainingArguments(**ta_kwargs)

    log_path = os.path.join(output_dir, "train_log.txt")
    callbacks = [
        _StepLossLogger(log_path),
        _EvalAndSaveCallback(
            save_steps_per_epoch=int(train_cfg.get("save_steps_per_epoch", 4)),
            is_peft=is_peft,
        ),
    ]

    # Optional: early stopping on validation loss. Set
    # `training.early_stopping_patience` > 0 to enable. 0 (default) disables it.
    es_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopper: _EarlyStoppingCallback | None = None
    if es_patience > 0:
        early_stopper = _EarlyStoppingCallback(patience=es_patience, log_path=log_path)
        callbacks.append(early_stopper)
        print(f"[train] Early stopping enabled: patience={es_patience}", flush=True)

    # Trainer renamed `tokenizer=` to `processing_class=` in 4.46. Pass whichever
    # the installed version supports.
    _trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    tr_kwargs: Dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    if "processing_class" in _trainer_params:
        tr_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _trainer_params:
        tr_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**tr_kwargs)

    # Remove HF's default progress callbacks so only our in-place tqdm bar
    # is visible (the Jupyter NotebookProgressCallback renders an HTML table
    # that grows one row per log event).
    try:
        from transformers.trainer_callback import ProgressCallback

        for _cb_cls in (ProgressCallback,):
            trainer.remove_callback(_cb_cls)
    except Exception:  # noqa: BLE001
        pass
    try:
        from transformers.utils.notebook import NotebookProgressCallback

        trainer.remove_callback(NotebookProgressCallback)
    except Exception:  # noqa: BLE001
        pass

    # Override Trainer._save so that PEFT models save adapter-only; otherwise
    # fall back to the default "save the whole model" behavior.
    original_save_model = trainer.save_model

    def custom_save_model(output_dir_: str = None, _internal_call: bool = False):
        out = output_dir_ or args.output_dir
        os.makedirs(out, exist_ok=True)
        if is_peft:
            trainer.model.save_pretrained(out)
            tokenizer.save_pretrained(out)
        else:
            original_save_model(out, _internal_call=_internal_call)

    trainer.save_model = custom_save_model  # type: ignore[assignment]

    print("[train] Starting training…", flush=True)
    train_result = trainer.train()

    # Save a final checkpoint explicitly (in addition to any the callback saved).
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)

    checkpoints = sorted(
        [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
        ]
    )

    summary = {
        "final_train_loss": float(train_result.training_loss),
        "global_step": int(train_result.global_step),
        "output_dir": output_dir,
        "checkpoints": checkpoints,
        "final_checkpoint": final_dir,
        "is_peft": is_peft,
        "method": method,
    }
    if early_stopper is not None:
        summary["early_stopping"] = {
            "enabled": True,
            "patience": early_stopper.patience,
            "stopped_early": early_stopper.stopped_early,
            "stop_step": early_stopper.stop_step,
            "best_eval_loss": early_stopper.best,
            "best_step": early_stopper.best_step,
        }
    with open(os.path.join(output_dir, "train_summary.json"), "w") as f:
        import json

        json.dump(summary, f, indent=2)
    return summary
