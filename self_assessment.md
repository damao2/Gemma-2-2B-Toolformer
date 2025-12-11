# CS 372 Final Project Self-Assessment

This document lists the rubric items I am claiming, along with pointers to evidence in the repository.  


---

## Category 1: Machine Learning (max 15 items)

### 1. Modular code design with reusable functions and classes (3 pts)

**Claimed.**  
**Evidence:**

- The project is split into modular scripts by task:
  - `src/exp1/exp1_eval_transcoders.py` – main `run_experiment` function plus helper functions for loading transcoders, computing metrics, and aggregating results.
  - `src/exp2/exp2A_eval_no_delta.py` and `src/exp2/exp2B_delta_scaling.py` – `run_experiment`, `generate_with_hooks`, `eval_subset_for_condition`, `create_hooks_for_layer_mode`, etc.  
  - `src/transcoder_finetune/fuse_delta_transcoders.py`, `validate_transcoder_Delta*.py` – separate utilities for model loading, fusing, and validation.
- Shared logic (e.g., hooks, metrics, dataset splitting) is encapsulated in functions instead of monolithic scripts.

---

### 2. Used appropriate data loading with batching and shuffling (3 pts)

**Claimed.**  
**Evidence:**

- `src/model_finetune/train_gemma_toolformer_LoRA.py` uses PyTorch/HF data loaders with batching for finetuning Gemma-2B on toolformer data (`data/toolformer_finetune/combined_dataset.jsonl`).
- `src/transcoder_finetune/finetune_transcoder_Delta2.py` builds PyTorch `DataLoader`s over SAE activation text inputs (`data/transcoder_finetune/sae_train.txt`, `sae_validation.txt`), with batching and (where appropriate) shuffling for training.

---

### 3. Created baseline model for comparison (3 pts)

**Claimed.**  
**Evidence:**

- In Experiments 2A and 2B, the **raw Gemma-2-2B finetuned model** serves as the behavioral baseline.
- `src/exp2/exp2A_eval_no_delta.py` explicitly compares three conditions:
  - `raw` (no SAE intervention),
  - `no_delta` (base SAE only),
  - `full` (base + delta).
- `src/exp2/exp2B_delta_scaling.py` extends this by sweeping α, but always includes the raw model in the metrics table for comparison.
- Baseline metrics appear in:
  - `src/exp2/exp2A_outputs/exp2A_metrics_*.csv`
  - `src/exp2/exp2B_outputs/exp2B_metrics_*.csv`

---

### 4. Applied regularization techniques to prevent overfitting (L1 / dropout / early stopping) (5 pts)

**Claimed.**  
**Evidence:**

- In Transcoder / DeltaTranscoder finetuning:
  - `src/transcoder_finetune/finetune_transcoder_Delta2.py` uses a **sparse autoencoder-style objective** with an explicit L1 sparsity penalty on the code activations (standard SAE regularization).
  - The Transcoder architecture and training hyperparameters (including weight decay and early stopping based on validation loss) are tuned to prevent overfitting to the activation dataset (`sae_train.txt`, `sae_validation.txt`).
- These regularization choices have measurable impact on reconstruction and downstream CE metrics, as shown in:
  - `src/exp1/exp1_outputs_*/*.csv` and `src/exp1/aggregate/exp1_combo_summary.csv`.

---

### 5. Conducted systematic hyperparameter tuning using validation data (5 pts)

**Claimed.**  
**Evidence:**

- Experiment 1 sweeps **Transcoder hyperparameters**:
  - Widths `d_new ∈ {32, 64, 128}` and percentiles `{0.25, 0.50, 0.75}` are evaluated.
- `src/exp1/exp1_eval_transcoders.py` runs these combos and writes per-config metrics to:
  - `src/exp1/exp1_outputs_d*_p*/*.csv`
- `src/exp1/exp1_aggregate_results.py` aggregates across configurations into:
  - `src/exp1/aggregate/exp1_combo_summary.{csv,json}`
- The validation metrics are used to choose good percentile / width configurations for Experiments 2A/2B (e.g., `d_new=128`, `percentile=0.50`).

---

### 6. Collected or constructed original dataset (10 pts)

**Claimed.**  
**Evidence:**

- Toolformer-style finetuning dataset:
  - `data/toolformer_finetune/combined_dataset.jsonl`, `positive_samples.jsonl`, `negative_samples.jsonl`.
  - Generated via `src/gen_dataset/gen_dataset.py` to create prompts that:
    - ask for arithmetic operations,
    - either require `<tool_call>calculator(expr)</tool_call>` (positive),
    - or require simple copying/echoing (negative).
- SAE / Transcoder activation data:
  - `data/transcoder_finetune/sae_train.txt`, `sae_validation.txt` produced by:
    - `src/gen_dataset_SAE/gen_dataset.py` (text),
    - then running the base/finetuned model to collect MLP activations.
- Experiment 2 evaluation data:
  - `data/exp2_combined_dataset.jsonl`, `data/exp2_combined_dataset_ood.jsonl`:
    - Combined scripts and manual curation to build three labeled subsets: `tool`, `copy`, `tool_ood`.
- All of these datasets were created specifically for this project and documented in `README.md` and `SETUP.md`.

---

### 7. Trained or fine-tuned an auxiliary model for interpretability (Sparse Autoencoder / Transcoder) (7 pts)

**Claimed.**  
**Evidence:**

- Base **Transcoders** (SAEs) from Gemma-Scope are loaded and used as interpretable bottlenecks for MLP activations.
- **DeltaTranscoders** are finetuned as *additional* interpretable components:
  - `src/transcoder_finetune/finetune_transcoder_Delta2.py` – training loop.
  - `src/transcoder_finetune/validate_transcoder_Delta.py` and
    `validate_transcoder_Delta_activations.py` – evaluation of reconstruction and activation patterns.
  - `src/transcoder_finetune/fuse_delta_transcoders.py` – fusing base + delta into a single module.
- Experiments 2A/2B use these SAEs/Transcoders *specifically* as interpretability tools to probe Gemma’s internal reasoning.

---

### 8. Fine-tuned pretrained model on your dataset (5 pts)

**Claimed.**  
**Evidence:**

- `src/model_finetune/train_gemma_toolformer_LoRA.py`:
  - Fine-tunes **pretrained Gemma-2-2B** on `data/toolformer_finetune/combined_dataset.jsonl` to learn calculator tool-calling behavior.
  - Uses HF Transformers + PEFT for efficient LoRA finetuning.
- The resulting LoRA weights are merged into a standalone finetuned model:
  - `src/model_finetune/merge_LoRA.py`
  - Output stored at `models/gemma_2b_toolformer_merged_v4/`.

---

### 9. Used Parameter-Efficient Fine-Tuning (PEFT) using LoRA (5 pts)

**Claimed.**  
**Evidence:**

- `src/model_finetune/train_gemma_toolformer_LoRA.py`:
  - Uses the `peft` library and **LoRA** adapters to fine-tune Gemma-2B.
  - LoRA config and adapter injection are explicit in the training script.
- LoRA is later merged into the base model for deployment:
  - `src/model_finetune/merge_LoRA.py`.

---

### 10. Used or fine-tuned a transformer language model (7 pts)

**Claimed.**  
**Evidence:**

- The main model is **Gemma-2-2B**, a Transformer LLM:
  - Loaded in `src/model_finetune/Gemma_chat.py`, `validate_gemma_LoRA*.py`, and in Experiment scripts.
- Finetuning:
  - `train_gemma_toolformer_LoRA.py` fine-tunes the transformer for tool-calling behavior.
- Evaluation:
  - `src/exp2/exp2A_eval_no_delta.py` and `src/exp2/exp2B_delta_scaling.py` run Gemma-2B with TransformerLens hooks to evaluate generation and internal activations.

---

### 11. Applied instruction tuning / supervised fine-tuning (SFT) for specific task format (7 pts)

**Claimed.**  
**Evidence:**

- The toolformer finetuning data contains **prompt–target pairs** enforcing a highly structured output format:
  - `<tool_call>calculator(expr)</tool_call>` vs. plain-text / copy responses.
- `data/toolformer_finetune/positive_samples.jsonl` and `negative_samples.jsonl` encode this task format.
- `src/model_finetune/train_gemma_toolformer_LoRA.py`:
  - Performs SFT to teach the model to follow these instructions and output the correct `<tool_call>` tags and expressions.
- `src/model_finetune/validate_gemma_LoRA_Prompt.py` shows that the finetuned model learns to respond to prompts like
  - `"I'd like to know the result of 649 / 821."`  
    with  
  - `<tool_call>calculator(649/821)</tool_call>`.

---

### 12. Used a significant software framework for applied ML not covered in class (TransformerLens) (5 pts)

**Claimed.**  
**Evidence:**

- `src/transformer_lens/` is a **local modified copy** of the TransformerLens library, and is used throughout:
  - `src/exp2/exp2A_eval_no_delta.py` – wraps Gemma-2B as a `HookedTransformer` and registers hooks at `hook_mlp_in` / `hook_mlp_out`.
  - `src/exp2/exp2B_delta_scaling.py` – uses the same hooks to scale delta contributions.
- This framework was not part of the standard course toolkit, but is leveraged to do mechanistic interventions.

---

### 13. Built multi-stage ML pipeline connecting outputs of one model to inputs of another (7 pts)

**Claimed.**  
**Evidence:**

- The core pipeline uses **multiple components in sequence**:
  1. Finetuned Gemma-2B generates MLP activations at a given layer.
  2. Base **Transcoder / SAE** encodes and decodes activations into a sparse code and reconstructed activation.
  3. **DeltaTranscoder** adds a delta reconstruction, yielding `base + delta` or `base + α·delta`.
  4. The modified activations are fed back into the remaining layers of the Transformer via hooks.
  5. Outputs are decoded and evaluated with behavioral metrics.
- This multi-stage pipeline is implemented in:
  - `src/exp2/exp2A_eval_no_delta.py` (base vs full vs no_delta),
  - `src/exp2/exp2B_delta_scaling.py` (alpha-scaling of delta).

---

### 14. Interpretable model design or explainability analysis (7 pts)

**Claimed.**  
**Evidence:**

- The project is explicitly about **interpreting internal reasoning** via SAEs:
  - Experiment 1 evaluates how well Transcoder / DeltaTranscoder reconstruct MLP activations and preserve structure.
  - Experiments 2A and 2B measure how interventions in SAE space affect behavior (tool calls, math correctness) and **mean activations**:
    - `mean_abs_base_acts`,
    - `mean_abs_delta_acts`.
- Plots in `src/exp2/exp2A_outputs/graphs/` and `src/exp2/exp2B_outputs/graphs/` visualize:
  - ΔCE vs layer,
  - Tool metrics vs layer,
  - Base/delta activation magnitudes vs layer and vs α.
- These analyses help explain which layers and which SAE subspaces are responsible for tool behavior, supporting interpretability.

---

### 15. Conducted behavioral, counterfactual, or mechanistic analysis to probe model’s internal reasoning (7 pts)

**Claimed.**  
**Evidence:**

- **Behavioral evaluation**:
  - `src/exp2/exp2A_eval_no_delta.py` and `exp2B_delta_scaling.py`:
    - Evaluate on three subsets: `tool`, `copy`, `tool_ood`.
    - Use `tool_call_rate`, `valid_tool_format_rate`, and `math_correct_rate` to quantify behavior on held-out prompts.
  - Per-example generations are stored in `exp2A_outputs/generations/` and `exp2B_outputs/generations/` and manually examined.
- **Counterfactual interventions**:
  - `no_delta`: removes delta features at a specific layer and measures how behavior changes vs `raw`.
  - `full`: adds base+delta and observes recovery of behavior.
  - `alpha`-scaled delta (2B): smoothly varies the strength of delta features to observe dose–response.
- **Mechanistic angle**:
  - Interventions are localized to specific layers and specific subspaces (SAE base vs delta).
  - Effects on behavior and loss are tracked as functions of layer and α.
- This is captured in:
  - `src/exp2/exp2A_outputs/exp2A_metrics_*.csv`,
  - `src/exp2/exp2B_outputs/exp2B_metrics_*.csv`,
  - and described in `README.md` under “Evaluation”.

---

## Category 2: Following Directions (max 20 points)

### Submission and Self-Assessment

- **Self-assessment submitted with evidence (3 pts)**  
  **Claimed.** This `self_assessment.md` document explicitly lists claimed items and evidence.


### Basic Documentation (2 pts each)

- **SETUP.md exists with clear installation instructions (2 pts)**  
  **Claimed.**  
  - `SETUP.md` in repo root describes:
    - Environment setup (`venv`, `pip install -r requirements.txt`),
    - Directory structure,
    - How to run dataset generation, finetuning, and Experiments 1/2A/2B,
    - Troubleshooting tips.

- **ATTRIBUTION.md with detailed sources and AI usage (2 pts)**  
  **Claimed.**  
  - `ATTRIBUTION.md` describes:
    - AI assistance (GitHub Copilot / GPT-5.1),
    - External libraries (PyTorch, Transformers, PEFT, TransformerLens, etc.),
    - Models (Gemma-2B, Gemma-Scope Transcoders),
    - Datasets used and how they were created.

- **requirements.txt or environment.yml is included and accurate (2 pts)**  
  **Claimed.**  
  - `requirements.txt` includes:
    - Core DL stack (torch, transformers, peft),
    - Local editable `transformer_lens` (via project structure),
    - Supporting libraries for data handling and plotting.

### README.md Sections (1 pt each)

- **What it Does section (1 pt)**  
  **Claimed.**  
  - `README.md` contains a “What it Does” section explaining, in a paragraph, that the project:
    - Studies a Gemma-2B tool-calling model,
    - Uses SAEs/DeltaTranscoders,
    - Runs behavioral and mechanistic evaluations.

- **Quick Start section (1 pt)**  
  **Claimed.**  
  - `README.md` includes “Quick Start” with:
    - Environment setup,
    - Directory overview,
    - Concrete commands to run Experiments 1, 2A, and 2B.

- **Video Links section (1 pt)**  
  **Claimed (once videos are added).**  
  - `README.md` has a “Video Links” section with:
    - Demo video URL,
    - Technical walkthrough URL.


- **Evaluation section (1 pt)**  
  **Claimed.**  
  - “Evaluation” section in `README.md` describes:
    - Datasets and subsets (`tool`, `copy`, `tool_ood`),
    - Metrics and main quantitative findings (e.g., raw vs full vs no_delta),
    - Qualitative conclusions.

- **Individual Contributions section (1 pt)**  
  **Claimed.**  
  - `README.md` has an “Individual Contributions” section describing my roles (data generation, finetuning, experiments, analysis).

### Video Submissions (2 pts each)


- **Demo video is correct length and non-technical (2 pts)**  
  - short demo showing:
    - Example prompts for tool and copy subsets,
    - High-level explanation of what the project does, without code.

- **Technical walkthrough explains code structure and ML techniques (2 pts)**  
  - walkthrough of:
    - Repo structure,
    - Gemma finetuning,
    - SAE/Delta mechanics,
    - Experiments 1, 2A, 2B and key plots.

### Project Workshop Days (1 pt each)

- Attended 1-2 project workshop days
- Attended 3-4 project workshop days
- Attended 5-6 project workshop days

---

## Category 3: Project Cohesion and Motivation (max 20 points)

### Project Purpose and Motivation (3 pts each)

- **README clearly articulates a single, unified project goal / research question (3 pts)**  
  **Claimed.**  
  - In `README.md`, the intro and “What it Does” sections state the main goals:
    - Probe whether the finetuned model “really” uses a tool or just matches patterns.
    - Evaluate whether freezing a base SAE and adding a small delta is a good, interpretable finetuning method.

- **Project demo video explains why the project matters in non-technical terms (3 pts)**  
  - explained the importance of the research question.

- **Project addresses a real-world problem or meaningful research question (3 pts)**  
  **Claimed.**  
  - The project is an applied mechanistic interpretability study of tool-using LLMs, relevant to:
    - Understanding LLM reasoning and safety,
    - Designing more interpretable fine-tuning mechanisms.

### Technical Coherence (3 pts each)

- **Technical walkthrough demonstrates how components work together (3 pts)**  
  **Claimed.**  
  - The code shows a coherent flow:
    - Dataset generation → model finetuning → SAE / DeltaTranscoder training → interventions and behavioral evaluation.
  - Experiments 2A/2B directly build on the trained components.

- **Project shows clear progression from problem → approach → solution → evaluation (3 pts)**  
  **Claimed.**  
  - Problem: Is the model truly using tools, and are delta SAEs a good, interpretable way to fine-tune?
  - Approach: Finetune Gemma; train Transcoder/Delta; intervene at selected layers; vary delta strength.
  - Solution: Use raw/full/no_delta and α-scaling comparisons.
  - Evaluation: Quantitative metrics (CE, tool_call, math_correct) and qualitative inspection, summarized in `README.md`.

- **Design choices are explicitly justified (3 pts)**  
  **Claimed.**  
  - `README.md` and comments in `src/exp2/exp2A_eval_no_delta.py` / `exp2B_delta_scaling.py` justify:
    - Why certain layers (e.g., 20, 24) are chosen,
    - Why we use three subsets (`tool`, `copy`, `tool_ood`),
    - Why we focus on delta scaling as a mechanistic probe.

- **Evaluation metrics directly measure stated objectives (3 pts)**  
  **Claimed.**  
  - Metrics (tool_call_rate, valid_tool_format_rate, math_correct_rate, ΔCE) are directly tied to:
    - Whether the model is using the tool correctly,
    - How SAE interventions impact behavior and performance.

- **No superfluous ML components (3 pts)**  
  **Claimed.**  
  - All major ML components (Gemma finetuning, SAEs/DeltaTranscoders, experiments 1/2A/2B) are directly connected to the main research questions; there are no “extra” models added just to collect rubric points.

- **Clean codebase with readable code and no major stale files (3 pts)**  
  **Claimed.**  
  - `deliver_version/` contains:
    - Only necessary code, data, and model files to reproduce the experiments.
    - Clear directory separation (`data/`, `models/`, `src/`, `docs/`).
  - Deprecated or unused scripts from earlier iterations are not included in this deliverable.

---