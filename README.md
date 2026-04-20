# Llama-3.2 Instruction Fine-Tuning & Benchmark Suite

Fine-tune **Llama-3.2** (1B / 3B / 11B) on any Hugging Face instruction dataset with **LoRA / DoRA / full** fine-tuning, and evaluate on four standard benchmarks:

| Benchmark   | What it measures                           | Judge              |
|-------------|---------------------------------------------|--------------------|
| **HEx-PHI** | Safety — refusal on 11 harmful categories   | GPT-4 (1-5 rubric) |
| **IFEval**  | Instruction-following (rule-based checks)   | None (programmatic) |
| **MMLU**    | 5-shot knowledge / reasoning, 57 subjects   | None (log-likelihood) |
| **MT-Bench**| Multi-turn conversational quality           | GPT-4 (1-10, single-answer grading) |

Everything is configured via YAML, but the **primary interface is a tiny Python API** designed for Google Colab notebooks: `train()` and `evaluate()`.

---

## Primary Usage — Python API in Colab

```python
# Cell 1: Setup
!git clone https://github.com/you/your-repo.git
%cd your-repo
!pip install -r requirements.txt
!huggingface-cli login

import os
os.environ["OPENAI_API_KEY"] = "sk-..."   # Only needed for HEx-PHI + MT-Bench judges.

from src import train, evaluate

# Cell 2: Train
result = train("configs/train/llama3_2_1b_lora.yaml")
print(result)

# Cell 3: Evaluate a local checkpoint — no YAML editing, no runtime restart.
scores = evaluate(
    checkpoint_path="./outputs/llama3_2_1b_lora/checkpoint-500",
    checkpoint_source="local",
    is_peft=True,
)
print(scores)

# Cell 4: Evaluate an adapter pushed to the Hub — still no restart, no YAML editing.
scores_hub = evaluate(
    checkpoint_path="your-username/my-lora-adapter",
    checkpoint_source="hub",
    is_peft=True,
)
print(scores_hub)

# Cell 5: Toggle benchmarks / tweak a nested setting without editing YAML.
scores_fast = evaluate(
    checkpoint_path="./outputs/llama3_2_1b_lora/final",
    checkpoint_source="local",
    is_peft=True,
    run_mt_bench=False,            # alias
    **{"mmlu.max_samples_per_subject": 20},  # dotted path
)
```

The above UX is the **core design requirement** of this repo: switching checkpoints in the same Colab session is just another call to `evaluate()` — the model is re-loaded under the hood, no kernel restart, no YAML edits.

### API reference

```python
train(config_path: str, **overrides) -> dict
evaluate(
    config_path: str = "configs/benchmark/default.yaml",
    checkpoint_path: str | None = None,
    checkpoint_source: str | None = None,   # "local" | "hub"
    **overrides,
) -> dict
```

Both functions return a summary dict. `evaluate()` also writes `benchmark_results/summary.json` plus per-benchmark detailed outputs (generations `.jsonl` + results `.json`).

---

## Secondary Usage — CLI

The CLI scripts are thin wrappers over the same API so there's a single source of truth.

```bash
# Train
python scripts/train.py --config configs/train/llama3_2_1b_lora.yaml \
    --set training.epochs=1 --set training.batch_size=2

# Evaluate
python scripts/evaluate.py \
    --config configs/benchmark/default.yaml \
    --checkpoint-path ./outputs/llama3_2_1b_lora/checkpoint-500 \
    --checkpoint-source local \
    --set mmlu.n_shot=3 --set benchmarks.run_mt_bench=false
```

---

## Override System

Any config value can be overridden from Python (`**kwargs`) or from the CLI (`--set key=value`). Two kinds of keys are accepted:

### Short aliases

| Alias                 | Maps to                      |
|-----------------------|------------------------------|
| `checkpoint_path`     | `model.checkpoint_path`      |
| `checkpoint_source`   | `model.checkpoint_source`    |
| `is_peft`             | `model.is_peft`              |
| `base_model`          | `model.base_model`           |
| `load_in_4bit`        | `model.load_in_4bit`         |
| `batch_size`          | `generation.batch_size`      |
| `output_dir`          | `evaluation.output_dir`      |
| `run_hex_phi`         | `benchmarks.run_hex_phi`     |
| `run_ifeval`          | `benchmarks.run_ifeval`      |
| `run_mmlu`            | `benchmarks.run_mmlu`        |
| `run_mt_bench`        | `benchmarks.run_mt_bench`    |

### Dotted paths

Anything else can be reached by its full dotted path, e.g. `mmlu.n_shot=3`, `training.epochs=5`, `mt_bench.generation_override.temperature=0.5`. Unknown keys raise a clear `ValueError`.

---

## Repository Layout

```
├── src/
│   ├── __init__.py             # re-exports train(), evaluate()
│   ├── api.py                  # load config → apply overrides → dispatch
│   ├── data/dataset.py         # HF dataset loading + tokenization
│   ├── training/
│   │   ├── peft_setup.py       # LoRA / DoRA config
│   │   └── trainer.py          # HF Trainer + per-step logging + periodic save/eval
│   ├── benchmark/
│   │   ├── base.py             # shared model loader (local OR hub) + batched generate
│   │   ├── hex_phi.py          # safety (GPT-4 judge, 1-5)
│   │   ├── ifeval.py           # instruction following (rule-based)
│   │   ├── mmlu.py             # 5-shot MMLU (log-likelihood or generation)
│   │   ├── mt_bench.py         # multi-turn conversational (GPT-4 judge, 1-10)
│   │   └── runner.py           # load model once, dispatch to enabled benchmarks
│   └── utils/config.py         # YAML load + override resolution
├── configs/
│   ├── train/{1b,3b,11b}*.yaml
│   └── benchmark/default.yaml
├── scripts/{train,evaluate}.py # thin CLI wrappers around src.api
├── requirements.txt
└── README.md
```

---

## Requirements & Credentials

### Gated Hugging Face resources

You must have requested access on the Hub for:

- `meta-llama/Llama-3.2-1B`, `Llama-3.2-3B`, `Llama-3.2-11B-Vision`
- `LLM-Tuning-Safety/HEx-PHI`

Then authenticate once per Colab session:

```bash
!huggingface-cli login
```

### OpenAI API key

HEx-PHI and MT-Bench use GPT-4 as a judge. Set:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

If the key is missing and either of those benchmarks is enabled, a clear error is raised (disable them via `run_hex_phi=False` / `run_mt_bench=False` to skip).

### IFEval official evaluator (optional but recommended)

For exact IFEval numbers, install the official package:

```bash
pip install instruction-following-eval
```

If it isn't installed, a minimal vendored scorer runs instead (covers the most common instruction types; good enough for a quick loop, not for reporting final numbers).

---

## Training Details

Controlled entirely by `configs/train/*.yaml`:

- **Methods:** `lora`, `dora` (LoRA with `use_dora=True`), `full`.
- **Per-step logging:** every training step's loss is printed to stdout and appended to `{output_dir}/train_log.txt`.
- **Checkpointing:** `save_steps_per_epoch` evenly divides each epoch; at every checkpoint step we compute validation loss and save:
  - LoRA/DoRA → adapter weights only.
  - Full → the entire model.
- **Reproducibility:** the resolved config is written to `{output_dir}/config.yaml` and a summary to `{output_dir}/train_summary.json`.
- **Early stopping (optional):** set `training.early_stopping_patience` > 0 to stop when validation loss fails to improve for N consecutive eval rounds (a "round" = one `save_steps_per_epoch` tick). The summary dict reports whether early stopping fired and the best eval loss / step.

  ```python
  train("configs/train/llama3_2_1b_lora.yaml",
        **{"training.early_stopping_patience": 3})
  ```

---

## Benchmark Details

- **Caesar-cipher I/O (critical):** the model was fine-tuned on Caesar-ciphered text (the `caesar_text` column of `Seungjun/alpaca_ceaser`), so every benchmark **encodes** prompts with a shift of 3 before sending them to the model and **decodes** responses before scoring or judging. MMLU's log-likelihood scorer also scores the ciphered letters (`D`, `E`, `F`, `G`) in place of `A`, `B`, `C`, `D`. Controlled by `generation.caesar_shift` (default `3`, set to `0` to disable).
- The runner **loads the model once** and shares it across all enabled benchmarks.
- For the 11B model on Colab, default to `load_in_4bit: true` (bitsandbytes NF4).
- `checkpoint_source: "local"` / `"hub"` is a contract — both `AutoModelForCausalLM.from_pretrained` and `PeftModel.from_pretrained` happily take either a local path or a HF repo id, so we just forward the string through. `"local"` additionally asserts the path exists on disk for clearer errors.
- Per-benchmark outputs (under `evaluation.output_dir`):

  ```
  summary.json
  benchmark_config.yaml
  hex_phi_generations.jsonl   hex_phi_results.json
  ifeval_generations.jsonl    ifeval_results.json
  mmlu_results.json
  mt_bench_generations.jsonl  mt_bench_results.json
  ```

---

## License / Attribution

Judge prompts for **HEx-PHI** and **MT-Bench** follow the official rubrics from the respective papers / releases (HEx-PHI: Qi et al. 2023; MT-Bench: Zheng et al. 2023, lmsys/FastChat).
