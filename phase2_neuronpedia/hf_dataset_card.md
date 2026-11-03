---
license: odc-by
language:
- en
tags:
- interpretability
- transcoders
- sparse-autoencoders
- gemma
- tool-use
pretty_name: Gemma-2-2B Toolformer — fused transcoder feature activations
size_categories:
- 1M<n<10M
configs:
- config_name: features
  data_files: features.parquet
- config_name: general_corpus_8k
  data_files: activations/general_corpus_8k/*.parquet
- config_name: toolcall_corpus_400
  data_files: activations/toolcall_corpus_400/*.parquet
---

# Gemma-2-2B Toolformer: fused transcoder feature activations

Top-activating text snippets for every feature of the **fused (base + delta) transcoders**
[`damaoo/gemma2b-toolformer-fused-transcoders`](https://huggingface.co/damaoo/gemma2b-toolformer-fused-transcoders)
(`fused_delta_32_pctl_0.5`), run on the fine-tuned model
[`damaoo/gemma2b-toolformer`](https://huggingface.co/damaoo/gemma2b-toolformer).
They were collected to power feature dashboards in a self-hosted
[Neuronpedia](https://github.com/hijohnnylin/neuronpedia) instance and exported from its
Postgres database. Project code and write-up:
[github.com/damao2/Gemma-2-2B-Toolformer](https://github.com/damao2/Gemma-2-2B-Toolformer)
(`phase2_neuronpedia/`).

## Feature indexing

Each of the 26 layers has a transcoder with 16384 Gemma-Scope base features
(`google/gemma-scope-2b-pt-transcoders`, width 16k) plus 32 learned **delta** features
appended at indices 16384–16415. `is_delta = feature >= 16384`.

## Files

| Config | Rows | Description |
|---|---|---|
| `features` | 426,779 | One row per feature that ever activated: `layer`, `feature`, `is_delta`, `max_act_approx`, `frac_nonzero`, `n_examples_general_8k` |
| `general_corpus_8k` | 2,131,655 | Top-5 snippets per feature over the first 8000 lines of a shuffled 20k corpus: 15k synthetic arithmetic / tool-call-intent prompts + 5k web snippets from [C4](https://huggingface.co/datasets/allenai/c4) |
| `toolcall_corpus_400` | 2,109,261 | Top-5 snippets per feature over the first 400 prompts of a 500-prompt corpus of tool-call positives, negatives and completions |

Activation rows contain `layer`, `feature`, `is_delta`, `max_value`, `max_token_index`,
`tokens` (list of token strings), `values` (per-token activation), and `created_at`.

## Caveats

* **Snapshots, not complete runs.** The collector flushed a cumulative top-k table to the DB
  periodically. 
* `max_act_approx` and `frac_nonzero` in `features` were written by the last flush
  (`toolcall_corpus_400`). `frac_nonzero` is computed over the stored top-k snippets only,
  so it is **not** the feature's true firing density.
* Activations are on the raw transcoder scale. Delta features can reach values in the
  thousands.
* The C4 portion of the snippets is distributed under C4's ODC-BY license; please attribute AllenAI's C4.
* Neuronpedia's DB schema has many more per-feature columns (logit histograms, UMAP, etc.).
  The local instance never filled them, so they are omitted.

## Example

```python
from datasets import load_dataset
acts = load_dataset("damaoo/gemma2b-toolformer-feature-activations", "general_corpus_8k", split="train")
f = acts.filter(lambda r: r["layer"] == 20 and r["feature"] == 16399)   # a delta feature
for r in f: print(r["max_value"], "".join(r["tokens"]))
```
