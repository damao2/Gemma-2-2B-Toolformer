# ATTRIBUTION

This document describes the attribution for AI-generated content, external libraries, datasets, and other resources used in this project.

---

## 1. AI-generated Code and Assistance

Parts of this repository were written with the assistance of AI tools.

- **Assistant used**: GitHub Copilot (backed by GPT-5.1 (Preview)).
- **Scope of assistance**:
  - Boilerplate code for experiment scripts (argument parsing, logging, file I/O).
  - Drafts of plotting and metrics aggregation logic for Experiments 1, 2A, and 2B.
  - Suggestions for refactoring evaluation loops, hooks, and generation functions.
  - Drafts of documentation, including this `ATTRIBUTION.md`, `README.md`, and `SETUP.md`.

All AI-generated or AI-assisted code was:

- Reviewed and edited by the author.
- Integrated into a custom project structure and adapted for the specific Gemma / SAE / DeltaTranscoder setup.

The high-level experimental design, dataset decisions, and interpretation of results are the author’s own.

---

## 2. External Libraries

The project relies heavily on the following external libraries:

- **PyTorch (`torch`)**
  - Used for all tensor operations, training loops, and model inference.
  - License: BSD-style license.
  - URL: https://pytorch.org/

- **Transformers (`transformers`)**
  - Hugging Face library for loading and running the Gemma model.
  - Used to load `AutoModelForCausalLM`, `AutoTokenizer`, and configure the finetuned model.
  - License: Apache 2.0.
  - URL: https://github.com/huggingface/transformers

- **PEFT (`peft`)**
  - Used for LoRA-based finetuning of Gemma-2B before merging LoRA adapters.
  - License: Apache 2.0.
  - URL: https://github.com/huggingface/peft

- **TransformerLens (`transformer_lens`)**
  - Interpretability library used to wrap the base Hugging Face Gemma model as a `HookedTransformer` and install hooks at MLP layers.
  - This project includes a **modified local copy** under `src/transformer_lens/`:
    - Some internal behavior (e.g., `generate` integration, hooking points) was adapted to match this project’s needs.
  - Original license: MIT.
  - Original repo: https://github.com/neelnanda-io/TransformerLens

- **Other Python libraries**
  - `numpy`, `pandas`, `scipy` – numerical and data analysis utilities.
  - `matplotlib` – plotting curves and visualizing metrics.
  - `tqdm` – progress bars.
  - `datasets`, `accelerate`, `huggingface-hub`, `safetensors`, `sentencepiece` – model and dataset handling tools from the Hugging Face ecosystem.

All such libraries are used under their respective open-source licenses. No code from these libraries was inlined into this repo; they are imported as dependencies.

---

## 3. Models

- **Gemma-2-2B (base model)**
  - Developed by Google.
  - Loaded via Hugging Face Transformers.
  - In this repository, we use a finetuned and merged version located under:
    - `models/gemma_2b_toolformer_merged_v4/`
  - The model is finetuned for calculator tool calls.
  - The original Gemma license and terms apply; check the official model card.

- **Gemma-Scope Transcoders (SAEs)**
  - “Transcoders” / SAEs for Gemma-2B MLP layers, originally provided as part of the Gemma-Scope project by Google.
  - This repository uses those as base SAEs and adds DeltaTranscoder layers on top.
  - Original assets (and model cards) are available from Google / Hugging Face.
  - https://huggingface.co/google/gemma-scope-2b-pt-transcoders
  - Licensing and usage follow the original Gemma-Scope terms.


---

## 4. Datasets

### 4.1 Toolformer-style Finetuning Data

- Location: `data/toolformer_finetune/`
  - `combined_dataset.jsonl`
  - `positive_samples.jsonl`
  - `negative_samples.jsonl`
- Description:
  - Synthetic or curated prompts asking for simple arithmetic operations (e.g., “What is 649 / 821?”) with targets indicating the desired `<tool_call>calculator(expr)</tool_call>` behavior or non-tool “copy” behavior.
  - Used to finetune the Gemma-2B model to learn calculator tool usage.

The construction of these datasets was inspired by the general Toolformer concept (Schick et al.), but the actual prompt/target pairs were created specifically for this project.

### 4.2 SAE / Transcoder Datasets

- Location: `data/transcoder_finetune/`
  - `sae_train.txt`
  - `sae_validation.txt`
- Description:
  - Text dumps (or token sequences) used to produce MLP activation datasets for training and validating SAEs and DeltaTranscoders.
  - The activations themselves are obtained by running the model on these inputs and collecting intermediate MLP layer outputs.

### 4.3 Experiment 2 Evaluation Datasets

- Location:
  - `data/exp2_combined_dataset.jsonl`
  - `data/exp2_combined_dataset_ood.jsonl`
- Description:
  - Combined evaluation data containing:
    - `tool` subset: in-distribution tool prompts.
    - `copy` subset: prompts that should not use the tool.
    - `tool_ood` subset: prompts that require the tool but have out-of-distribution phrasing or numeric variations.
  - Includes labels indicating which subset each example belongs to.

All these datasets were locally generated or curated for this project and are included to allow deterministic reproduction of the experiments.

---

## 5. Other Resources and References

- **Toolformer (Schick et al.)**
  - The conceptual inspiration for training language models to call external tools (such as a calculator).
  - This project does not copy any code or data from the original Toolformer implementation, but follows the general idea of teaching a model to emit `<tool_call>...` tokens.

- **Mechanistic interpretability literature**
  - The use of SAEs (sparse autoencoders) and Transcoders for probing model internals is inspired by prior work on mechanistic interpretability.
  - No external mechanistic code is copied into this repo; instead, this project builds on top of Gemma-Scope and TransformerLens APIs.

---
