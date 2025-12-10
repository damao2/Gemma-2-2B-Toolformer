# Toolformer + SAE Delta Transcoders for Gemma-2B

Short project exploring how a Gemma-2B model finetuned for calculator tool calls behaves internally, and an explainable and friendly fine-tuning method of Transcoders to do mechanistic interpretability studies. The repo contains code to generate tool-calling data, finetune the model and SAEs, and run behavioral / mechanistic evaluations.

---

## What it Does

This project studies a Gemma-2B causal language model finetuned to use a calculator tool via `<tool_call>calculator(expr)</tool_call>` generations. On top of an existing Gemma-Scope SAE (“Transcoder”), we train small delta transcoders and then intervene on the model’s MLP activations at specific layers. We systematically compare three conditions—raw model, base SAE only (`no_delta`), and base+delta (`full` / scaled alphas)—across in-distribution tool prompts, non-tool “copy” prompts, and out-of-distribution (OOD) tool prompts. The code reproduces the dataset generation, model finetuning, Experiment 1 (SAE validation) and Experiments 2A/2B (behavioral and mechanistic probing).

---

## Quick Start

### 1. Environment Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

This will install PyTorch, Hugging Face Transformers/PEFT, a local modified copy of `transformer_lens`, and plotting / data utilities.

### 2. Directory Overview

- `data/`
  - Toolformer-style finetuning data and SAE train/validation splits.
- `models/gemma_2b_toolformer_merged_v4/`
  - Local copy of the finetuned Gemma-2B model weights and tokenizer.
- `src/`
  - `exp1/`: SAE / Transcoder validation experiments.
  - `exp2/`: Tool-calling behavioral & mechanistic experiments (2A/2B).
  - `gen_dataset*/`: Dataset generation scripts.
  - `model_finetune/`: Gemma Toolformer LoRA training / merge / validation.
  - `transcoder_finetune/`: Delta Transcoder finetuning & validation.
  - `transformer_lens/`: Local modified copy of the TransformerLens library.

### 3. Running Key Experiments

All commands assume you are in the repo root and the venv is activated.

#### Experiment 1: SAE / Transcoder Validation

```bash
cd src/exp1

# Evaluate base+delta transcoders (example hyperparameters)
python exp1_eval_transcoders.py \
  --val_file ../../data/transcoder_finetune/sae_validation.txt \
  --percentile 0.50 \
  --delta_root /path/to/delta_checkpoints_root \
  --d_new 128

# Aggregate CSV / JSON summaries
python exp1_aggregate_results.py
```

Outputs go under `src/exp1/exp1_outputs_*` and `src/exp1/aggregate/`.

#### Experiment 2A: Per-layer SAE Interventions

```bash
cd src/exp2

python exp2A_eval_no_delta.py \
  --val_file ../../data/exp2_combined_dataset_ood.jsonl \
  --percentile 0.50 \
  --delta_root /path/to/delta_checkpoints_root \
  --d_new 128 \
  --layers 16,20,24
```

This:

- Intervenes at the specified layers with:
  - `raw` (no SAE),
  - `no_delta` (base SAE only),
  - `full` (base + delta SAE),
- Evaluates three subsets:
  - `tool` (in-distribution tool prompts),
  - `copy` (non-tool numeric prompts),
  - `tool_ood` (OOD tool prompts),
- Logs:
  - Per-condition metrics CSV/JSON: `exp2A_outputs/exp2A_metrics_*.{csv,json}`,
  - Generated model outputs per condition: `exp2A_outputs/generations/*.jsonl`,
  - Plots: `exp2A_outputs/graphs/*.png`.

#### Experiment 2B: Delta Scaling (α-sweep)

```bash
cd src/exp2

python exp2B_delta_scaling.py \
  --val_file ../../data/exp2_combined_dataset_ood.jsonl \
  --percentile 0.50 \
  --delta_root /path/to/delta_checkpoints_root \
  --d_new 128 \
  --layers 20,24 \
  --delta_alphas 0.0,0.25,0.5,0.75,1.0
```

This runs `recon = base + α·delta` for multiple α and records how cross-entropy and tool metrics change as we turn the delta “up and down”. Outputs are stored in `exp2B_outputs/`.

For details on dataset generation and finetuning (model and Transcoder/DeltaTranscoder), see `SETUP.md`.

---

## Video Links

- **Demo video** (behavior on representative prompts, including tool and copy examples):  
  _[Add your demo video URL here]_
- **Technical walkthrough** (project structure, experiments 1/2A/2B, key plots):  
  _[Add your technical walkthrough video URL here]_

---

## Evaluation

This section summarizes the main quantitative and qualitative findings.

### Datasets and Subsets

- **Tool subset (`tool`)**: prompts that explicitly request a calculation and are labeled to expect `<tool_call>calculator(expr)</tool_call>`.
- **Copy subset (`copy`)**: prompts containing numbers but meant to be repeated or copied without using the tool.
- **OOD tool subset (`tool_ood`)**: prompts requiring the calculator tool but with out-of-distribution phrasing or numeric regimes.

Each subset uses 20 evaluation examples for fast, controlled analysis in exp2A/2B.

### Metrics

For each subset, layer, and condition, we record:

- `mean_ce`: token-level cross-entropy (teacher forcing).
- `tool_call_rate`: fraction of examples where the *generated* output contains `<tool_call>`.
- `valid_tool_format_rate`: fraction with a well-formed `<tool_call>...</tool_call>` block.
- `math_correct_rate`: fraction where the predicted `calculator(expr')` evaluates to the same numeric value as the gold `expr`.

Additionally, we track:

- `mean_abs_base_acts`: mean |activation| per token in the base SAE subspace.
- `mean_abs_delta_acts`: mean |activation| per token in the delta subspace.

### Experiment 1: SAE / Transcoder Validation

- Base Transcoder (Gemma-Scope) and fused base+delta Transcoders were evaluated on held-out MLP activations.
- Fused models generally:
  - Preserve or slightly improve reconstruction quality on activations from the finetuned model;
  - Maintain sparse, interpretable activations in the base space, with additional capacity in delta.

(See `src/exp1/aggregate/exp1_combo_summary.{csv,json}` for detailed metrics.)

### Experiment 2A: Per-layer Behavioral Effects (raw vs full vs no_delta)

Key findings (summarized qualitatively):

- **Base SAE alone is harmful**:
  - `no_delta` significantly increases `mean_ce` across subsets, especially at high layers.
  - For `tool` and `tool_ood`, `tool_call_rate` and `math_correct_rate` often collapse toward 0 under `no_delta`, meaning the model largely stops calling the calculator or produces incorrect tool usage.
  - On `copy`, `no_delta` keeps `tool_call_rate` near 0 at mid/high layers, but at some early layers (e.g., layer 0) can cause degenerate behavior (e.g., always calling the tool), illustrating that naive SAE replacement can be catastrophic.
- **Delta restores tool behavior on in-distribution prompts**:
  - At key layers (e.g., 20, 24), `full = base + delta` recovers almost perfect `tool_call_rate ≈ 1.0`, `valid_tool_format_rate ≈ 1.0`, and `math_correct_rate` close to raw on the `tool` subset.
  - The same layers under `no_delta` show much lower tool metrics, demonstrating that delta features are necessary to maintain good tool behavior in the finetuned model.
- **OOD behavior**:
  - On `tool_ood`, raw already has lower `valid_tool_format_rate` and `math_correct_rate` than in-distribution.
  - `full` improves over `no_delta` but still underperforms raw, indicating limited generalization of the finetuned tool behavior to OOD phrasing.

### Experiment 2B: Delta Scaling (dose–response)

Experiment 2B studies `recon = base + α·delta` with `α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` at a few critical layers (e.g., 20, 24):

- On `tool` / `tool_ood` subsets:
  - As α increases from 0 to 1, `tool_call_rate` and `math_correct_rate` generally increase, often monotonically.
  - This indicates a smooth dose–response relationship between delta strength and tool behavior: delta acts like a controllable “tool module knob”.
- On `copy` subset:
  - Across α, `tool_call_rate` remains near 0 at mid/high layers, showing that delta does not indiscriminately cause tool hallucinations on non-tool prompts.
- Activation magnitudes:
  - In `full` condition (α=1), `mean_abs_delta_acts` is larger on `tool`/`tool_ood` than on `copy`, especially at the key layers.
  - This suggests that delta features are preferentially engaged for tool-related behavior.

### High-level Conclusions

- The finetuned model does more than pure prompt pattern matching:
  - It generalizes tool use to OOD prompts and keeps tool usage low on copy prompts.
- Freezing a base SAE and adding a small delta subspace is:
  - Not a “harmless compression” by itself (base-only replacement can be very damaging).
  - But, when combined as `base + delta`, it provides a relatively localized and controllable way to reintroduce tool behavior.
  - Delta scaling (Experiment 2B) supports an interpretable, causally meaningful control over tool behavior.

---

## Individual Contributions



- **Qiren Chen**
  - Designed and implemented dataset generation for tool and SAE experiments (`data/`, `src/gen_dataset*/`).
  - Implemented Gemma-2B Toolformer finetuning, LoRA merging, and validation (`src/model_finetune/`).
  - Integrated Gemma-Scope Transcoders and designed DeltaTranscoder finetuning and fusion (`src/transcoder_finetune/`).
  - Implemented Experiment 1 (SAE validation) and Experiment 2A/2B (behavioral and mechanistic analysis), including intervention hooks and metrics (`src/exp1/`, `src/exp2/`).
  - Ran experiments, analyzed results, and created plots / tables.
