# SETUP

This document explains how to install dependencies, set up the environment, and run the main components of the project.

---

## 1. Prerequisites

- Python 3.10+ (tested on Linux)
- A GPU with CUDA is recommended for running Gemma-2B and the SAE interventions, but small tests may run on CPU.
- Sufficient disk space for:
  - Finetuned Gemma-2B model under `models/gemma_2b_toolformer_merged_v4/`
  - DeltaTranscoder checkpoints (not bundled; you point to them via `--delta_root`). In some of the files, download from HF is implemented, and the transcoders will download from `damaoo/gemma2b-toolformer-fused-transcoders`

No external APIs or network calls are required at runtime: all models and data are local once the repo is cloned and model files are present.

---

## 2. Install Dependencies

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The `requirements.txt` includes:

- `torch` – PyTorch
- `transformers`, `peft` – Hugging Face modeling and LoRA
- `transformer-lens` – installed from the local modified copy in `src/transformer_lens/` (via `-e ./src/transformer_lens` or equivalent)
- `datasets`, `accelerate`, `safetensors`, `sentencepiece`
- `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`, `huggingface-hub`

If you adjust paths, ensure that the local `transformer_lens` (under `src/transformer_lens/`) is installed instead of the PyPI version.

---

## 3. Repository Layout

At the top level:

- `requirements.txt` – Python dependencies.
- `data/` – all text/jsonl datasets used for finetuning and evaluation.
  - `toolformer_finetune/` – Toolformer-style training/negative/positive samples.
  - `transcoder_finetune/` – activation text dumps for SAE training and validation.
  - `exp2_combined_dataset*.jsonl` – merged evaluation datasets for Experiments 2A/2B.
- `models/gemma_2b_toolformer_merged_v4/` – local finetuned Gemma-2B model and tokenizer.
- `src/`
  - `exp1/` – Experiment 1: SAE/Transcoder validation.
  - `exp2/` – Experiments 2A and 2B: behavioral and mechanistic probing.
  - `gen_dataset/`, `gen_dataset_SAE/` – dataset generation scripts.
  - `model_finetune/` – Gemma Toolformer finetuning and validation utilities.
  - `transcoder_finetune/` – Transcoder & DeltaTranscoder finetuning and validation.
  - `transformer_lens/` – local modified TransformerLens library.

---

## 4. Running the Main Components

All commands assume:

- You are in the repo root.
- The virtualenv is activated: `source .venv/bin/activate`.

### 4.1 Dataset Generation (Optional / For Reproducibility)

If you want to regenerate the datasets:

```bash
cd src/gen_dataset
python gen_dataset.py \
  --output ../../data/toolformer_finetune/combined_dataset.jsonl

cd ../gen_dataset_SAE
python gen_dataset.py \
  --train_output ../../data/transcoder_finetune/sae_train.txt \
  --val_output ../../data/transcoder_finetune/sae_validation.txt
```

The repository already includes pre-generated data under `data/`, so this step is optional unless you wish to regenerate with different parameters.

### 4.2 Gemma Toolformer Finetuning (Optional)

The finetuned Gemma-2B model is provided in `models/gemma_2b_toolformer_merged_v4/`. If you want to reproduce or modify training:

```bash
cd src/model_finetune

# LoRA training
python train_gemma_toolformer_LoRA.py \
  --train_file ../../data/toolformer_finetune/combined_dataset.jsonl \
  --output_dir ./lora_out

# Merge LoRA into base Gemma-2B
python merge_LoRA.py \
  --base_model google/gemma-2-2b \
  --lora_dir ./lora_out \
  --output_dir ../../models/gemma_2b_toolformer_merged_v4
```

To quickly validate that the tool-calling behavior works:

```bash
python validate_gemma_LoRA_Prompt.py \
  --model_path ../../models/gemma_2b_toolformer_merged_v4
```

This script will prompt for an example input like  
`I'd like to know the result of 649 / 821.` and show the model’s tool call.

### 4.3 Transcoder / DeltaTranscoder Finetuning (Optional)

DeltaTranscoder checkpoints are not shipped; you will generally point `--delta_root` to where they are stored. To finetune them yourself:

```bash
cd src/transcoder_finetune

python finetune_transcoder_Delta2.py \
  --train_file ../../data/transcoder_finetune/sae_train.txt \
  --val_file ../../data/transcoder_finetune/sae_validation.txt \
  --output_root /path/to/delta_root \
  --d_new 128 \
  --percentiles 0.50
```

You can then fuse and validate:

```bash
python fuse_delta_transcoders.py \
  --delta_root /path/to/delta_root \
  --d_new 128 \
  --percentile 0.50

python validate_transcoder_Delta.py \
  --val_file ../../data/transcoder_finetune/sae_validation.txt \
  --delta_root /path/to/delta_root \
  --d_new 128 \
  --percentile 0.50
```

### 4.4 Experiment 1: SAE / Transcoder Validation

```bash
cd src/exp1

python exp1_eval_transcoders.py \
  --val_file ../../data/transcoder_finetune/sae_validation.txt \
  --delta_root /path/to/delta_root \
  --d_new 128 \
  --percentile 0.50
```

You can aggregate results with:

```bash
python exp1_aggregate_results.py
```

Key outputs:

- `exp1_outputs_*/exp1_metrics_*.{csv,json}`
- `aggregate/exp1_combo_summary.{csv,json}`

### 4.5 Experiment 2A: Layer-wise SAE Intervention

```bash
cd src/exp2

python exp2A_eval_no_delta.py \
  --val_file ../../data/exp2_combined_dataset_ood.jsonl \
  --delta_root /path/to/delta_root \
  --d_new 128 \
  --percentile 0.50 \
  --layers 16,20,24
```

Outputs:

- `exp2A_outputs/exp2A_metrics_*.{csv,json}` – main metrics table.
- `exp2A_outputs/generations/*.jsonl` – per-example generations for raw/full/no_delta.
- `exp2A_outputs/graphs/*.png` – plots for:
  - ΔCE(no_delta − raw) vs layer (combined subsets),
  - tool metrics vs layer (per subset),
  - base/delta mean activation vs layer (full condition).

### 4.6 Experiment 2B: Delta Scaling

```bash
cd src/exp2

python exp2B_delta_scaling.py \
  --val_file ../../data/exp2_combined_dataset_ood.jsonl \
  --delta_root /path/to/delta_root \
  --d_new 128 \
  --percentile 0.50 \
  --layers 20,24 \
  --delta_alphas 0.0,0.25,0.5,0.75,1.0
```

Outputs:

- `exp2B_outputs/exp2B_metrics_*.{csv,json}` – metrics with an `alpha` column.
- `exp2B_outputs/generations/*.jsonl` – generations for each `(layer, subset, alpha)` condition.
- `exp2B_outputs/graphs/exp2B_metrics_vs_alpha_*.png` – metric vs alpha curves per subset.

---

## 5. External Services / APIs

The project does **not** require external APIs at runtime:

- All inference is local, using:
  - The Gemma-2B finetuned model in `models/`,
  - Local data files under `data/`,
  - Local SAE/DeltaTranscoder checkpoints (path provided by `--delta_root`).

If the grader does not have a GPU or enough memory to load Gemma-2B, they can still inspect the code, run small-scale variants on CPU, or subsample the data to test the pipeline.

---

## 6. Troubleshooting

- **ImportError: `transformer_lens`**  
  Ensure that:
  - `src/transformer_lens/` is installed editable via `-e ./src/transformer_lens` in `requirements.txt`, or
  - You run `pip install -e ./src/transformer_lens` manually from the repo root.

- **Out of memory (CUDA OOM)**  
  Try:
  - Reducing batch sizes in finetuning scripts,
  - Running only a subset of layers or fewer evaluation examples,
  - Switching to CPU for small tests: `CUDA_VISIBLE_DEVICES="" python ...` (very slow but good for sanity checks).

- **Path issues**  
  All paths in scripts are relative to the repo root (e.g., `../../data/...` from under `src/`). Ensure you run commands from the correct working directory (usually the repo root, then `cd src/...` as needed).