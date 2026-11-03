# Phase 2 — Probing the tool-call gate with attribution graphs (self-hosted Neuronpedia)

This folder contains the post-midterm part of the project. Phase 1 (repo root) built the
fine-tuned Toolformer model and the **delta transcoders**, and showed with layer-wise
interventions that the delta features carry most of the tool-call behaviour. Phase 2 asks
the next question with **attribution graphs** (circuit-tracer) served through a
**self-hosted Neuronpedia**:

> When the fine-tuned Gemma-2-2B decides between starting `<tool_call>` and echoing the
> prompt, what mechanism makes that decision, and how robust is it?

> **Status / provenance.** This phase was stopped before completion. The working copy lived
> on a scratch filesystem that was purged; everything here was reconstructed from (a) a
> January-2026 copy of the Neuronpedia checkout + Postgres DB and (b) VS Code local-history
> snapshots of the later (April-2026) scripts. The **code and design are complete**; the
> **numeric outputs of the April batch circuit evaluation were lost** and are not reported.
> See [What is missing](#what-is-missing).

---

## 1. The research pivot: from "semantic tool use" to a brittle heuristic gate

Phase 1 concluded, cautiously, that the model showed "some semantic generalization".
Systematic manual probing of near-out-of-distribution prompts in the Neuronpedia UI
(full log: [`docs/manual_ood_probing_zh.md`](docs/manual_ood_probing_zh.md)) points the
other way. The model only has two output modes, so the decision is visible at the first
generated token: `<` (start of `<tool_call>`) versus the first token of the prompt (echo).

| Prompt (each ends with a newline) | Next-token distribution |
|---|---|
| `Compute the value of 112 * 309.` | `<` 1.00 |
| `compute the value of 112 * 309.` (lower-case) | `<` 0.80, `compute` 0.19 |
| `compute the value of 11 * 309.` | `<` 0.63, `compute` 0.36 |
| `compute the value of 1 * 309.` | **`compute` 0.69**, `<` 0.31 |
| `compute the value of 1 * 39.` | `<` 0.77, `compute` 0.23 |
| `compute the value of 1 * 3.` | **`compute` 0.53**, `<` 0.47 |
| `What’s the difference between 598 and 914?` | `<` 0.98 |
| `What’s the difference between 598 and 914` (no `?`) | **`What` 0.68**, `<` 0.29 |
| `What is 534 + 267` (no `?`) | `<` 1.00 |
| `What’s the difference between 598 and 914? Do not use any tool.` | `<` 0.66 (should echo) |
| `Do not use any tool. What’s the difference between 598 and 914?` | `Do` 0.98 (echo) |
| `… Do not use any tool.␠` (one trailing space) | `What` 0.60, `<` 0.30 |
| `Please, compute the value of 112 * 309.` | `<` 1.00 |

What these probes show:

* **Failures are narrow and template-specific, not uniform.** Most perturbations of
  seen templates are harmless; a few specific interactions flip the decision. Dropping
  the `?` hurts only the "difference between" template.
* **Interactions, not single factors.** Lower-case alone or small numbers alone
  rarely flip the gate. Together they do, and not monotonically in number size
  (`1 * 309` flips, `1 * 39` does not, `1 * 3` sits on the boundary).
* **Instructions act through position, not meaning.** "Do not use any tool" works when it
  is the first sentence (its first token wins the competition) and is mostly ignored as a
  suffix. A single trailing space can reverse the outcome.

Working hypothesis: the fine-tune installed a **heuristic gate**, a competition at
the last position between (i) seen-template features pushing `<`, (ii) features of the
prompt's first token pushing echo, and (iii) token-boundary features (punctuation,
whitespace, digit tokenisation). It is not a robust semantic router.

## 2. Pipeline

```
 tool-call corpus / mixed corpus ──► collect_top_activations.py ──► Postgres (Neuron, Activation)
          (gen_*_corpus.py)              fused base+delta transcoders            │
                                         (damaoo/gemma2b-toolformer-fused-…)     ▼
 prompt ──► graph server (circuit-tracer) ──► attribution graph ──► Neuronpedia UI: inspect,
                                                                    pin & label a subgraph
 eval prompts ──► gen_circuit_eval_prompts.py ──► screen_circuit_eval_prompts.ts (1 forward pass,
                                                    slices: capability / boundary / stress)
                                                         │
 saved subgraph + screened prompts ──► batch-eval-subgraph.ts ──► analyze/visualize_circuit_eval.py
```

| Stage | Files (`neuronpedia_overlay/…`) |
|---|---|
| Serve the fine-tuned model + fused transcoders as a graph backend | `apps/graph/neuronpedia_graph/server.py`, `apps/graph/start.py` |
| Register the local model / source set in the webapp | `apps/webapp/scripts/seed-local-graph-model.ts`, `lib/db/*.ts`, `lib/env.ts` |
| Generate graphs from the UI and API | `apps/webapp/app/api/graph/generate/route.ts`, `app/[modelId]/graph/*.tsx` |
| Feature dashboards (top activations) | `apps/graph/collect_top_activations.py`, `apps/webapp/app/api/feature/…/route.ts` |
| Save/load manually pinned subgraphs | `app/[modelId]/graph/modals/{save,load}-subgraph-modal.tsx` |
| Graph scoring (replacement / completeness, gate and signed-margin variants) | `app/[modelId]/graph/score-core.ts`, `score.worker.ts` |
| Eval-set generation and screening | `apps/webapp/scripts/gen_circuit_eval_prompts.py`, `screen_circuit_eval_prompts.ts`, `gen_circuit_eval_boundary_candidates.py` |
| Batch evaluation of a fixed circuit across prompts | `apps/webapp/scripts/batch-eval-subgraph.ts` (explained in [`docs/batch_eval_explained_zh.md`](docs/batch_eval_explained_zh.md)) |
| Aggregation and plots | `apps/webapp/scripts/analyze_circuit_eval.py`, `visualize_circuit_eval.py` |

`neuronpedia_overlay/` holds only the files that were added or changed. To use them, copy
them over a checkout of [hijohnnylin/neuronpedia](https://github.com/hijohnnylin/neuronpedia)
at commit `314ed0a`. Local setup notes (Postgres on a non-root cluster, ports, SSH tunnels):
[`docs/local_setup_notes_zh.md`](docs/local_setup_notes_zh.md).

## 3. Evaluation design for the gate

Design notes: [`docs/research_plan_v2_zh.md`](docs/research_plan_v2_zh.md),
[`docs/circuit_eval_targets_and_metrics.md`](docs/circuit_eval_targets_and_metrics.md).

**Multi-axis prompt labels instead of an ID/OOD split.** Each prompt carries `intent`
(pos/neg), `distribution` (id/ood), `expectedBehavior` (tool_call/echo_first),
`designRole` and `variantGroup`. The model's actual behaviour is then measured, not
assumed. One forward pass gives

* gate margin `m = P("<") − P(echoFirst)`, where `echoFirst` is the first non-special prompt token
* signed margin `m_s = +m` for prompts that should call the tool, `−m` for prompts that should echo

and each prompt is sliced into **capability** (top-1 correct and |m| ≥ τ), **boundary**
(top-1 correct but |m| < τ) or **stress** (top-1 wrong). τ defaults to 0.40 in the final
script; 0.10 was also used. With this scheme, minimal pairs landing
in different slices are a feature: they are the brittle boundary being studied.

**Circuit metrics.** Every attribution graph is scored three ways, for the full graph and
for the mapped subgraph: Neuronpedia's default probability-weighted replacement and
completeness scores; a **gate-only** version (logit weight 1 on `<`); and a
**signed-margin** version (+1 on `<`, −1 on `echoFirst`, influences aggregated in absolute
value). A circuit chosen on one prompt is transferred to a new prompt by matching
`(layer, feature, position relative to the target logit)`. The script also reports
`coverage = matched / template size`.

The regenerated evaluation set is
[`neuronpedia_overlay/apps/webapp/scripts/prompts_circuit_eval_v3.jsonl`](neuronpedia_overlay/apps/webapp/scripts/prompts_circuit_eval_v3.jsonl).
It has 50 prompts, from `gen_circuit_eval_prompts.py --seed 0`, the final version of the
generator: 10 seen positives, 10 near-boundary positives, 10 hard negatives, 10 explain-only
negatives and 10 anti-tool negatives.

## 4. Result: the manually traced circuit for `Calculate 3 + 5.`

The one hand-built circuit that survived (`test1`) traces the `<` logit (p = 0.997) on the
prompt `Calculate 3 + 5.`. It has 23 transcoder features, 3 token embeddings and the
logit, grouped into six hand-labelled supernodes. Full table, with labels and per-node
data: [`results/test1_circuit/test1_circuit_annotated.json`](results/test1_circuit/test1_circuit_annotated.json).

| Supernode (hand label) | Features (layer_feature, **bold = delta**) | Position |
|---|---|---|
| Identify the problem-solving need | **0_16395** "problem", **0_16388** "help needed", 0_9789 | `Calculate` |
| Identify calculation need | 0_12768 "calculate", 0_12464 "calculate", 0_12030, 0_12556, **22_16384** "core math expression extractor" | `Calculate` / last |
| Structural anchors | 3_10369 "first word in sentence", 7_766 "problem statement terminator" | `Calculate` / last |
| Identify math and numbers | **1_16385**, 8_3371, 15_12197, **17_16396**, **18_16402**, 24_13712 "number scraper" | `Calculate` / last |
| Specific math pattern | **19_16398** "'value of' / 'difference between' matcher", **20_16399** "'trying to figure out' intent matcher" | last |
| Tool-call syntax | **21_16409**, 24_2800 "opening tags & tool syntax", 25_4836, 25_6783, 25_11623 | last |

**Base vs delta division of labour.** The 32 delta features per layer are about 0.2% of the
fused dictionary, yet:

* they are **9 of the 23** features in the hand-picked circuit;
* across the whole pruned graph (890 feature nodes) they make up **6.4%** of the nodes
  (about 33× over-represented) and **19.8%** of outgoing edge weight;
* their placement is structured. Layer 0 at the `Calculate` token is mostly base lexical
  features ("calculate"), plus two delta "problem / help needed" features. **Layers 17–22 at
  the last position are entirely delta in the circuit** (12 of the 20 nodes in layers 17–23 of
  the full graph are delta). Layers 24–25, which write the tool-call syntax, are base features again.

Reading: the fine-tune reuses the base model's lexical detectors and its output-syntax
features. The **decision in between**, i.e. template matching ("value of", "difference
between", "trying to figure out") and math-expression detection, is carried by the
new delta features. This fits Phase 1, where removing the delta (`no_delta`) collapsed tool calling
at most layers, and fits the template-specific brittleness of Section 1: the template matchers
are features of their own, so a template they do not fire on loses the gate.

*Caveat:* this is a single prompt and a single hand-selected circuit. The batch evaluation
built to test whether such circuits transfer across the eval set (Section 3) was run, but
its outputs were lost.

## 5. Data

| What | Where |
|---|---|
| Top-5 activating snippets for all 426,779 features, two corpora (Parquet, 86 MB) | Hugging Face dataset, card in [`hf_dataset_card.md`](hf_dataset_card.md); exported by [`tools/export_neuronpedia_db.py`](tools/export_neuronpedia_db.py) |
| Interactive viewer: `test1` circuit + index of all 800 delta features with activations | [`../docs/circuit-viewer/`](../docs/circuit-viewer/) (GitHub Pages; data built by [`tools/build_viewer_data.py`](tools/build_viewer_data.py)) |
| Annotated circuit, source attribution graph | [`results/test1_circuit/`](results/test1_circuit/) |
| Activation corpora and their generators | [`corpus/`](corpus/) |

**Provenance of the activation data.** `collect_top_activations.py` was run twice with
top-k = 5, and each job hit the cluster's time limit. The collector periodically flushes
its *cumulative* top-k table, so the database held 12 full snapshots (27.4M rows). Only the
last committed snapshot of each run is published. Row counts match the job logs exactly:

| Snapshot | Corpus | Prompts seen | Rows |
|---|---|---|---|
| `general_corpus_8k` | `corpus/collect_top_activations_corpus_20k.txt` (15k synthetic tool-intent prompts + 5k C4, shuffled) | 8000 / 20000 | 2,131,655 |
| `toolcall_corpus_400` | `corpus/toolcall_activations_corpus_500.txt` | 400 / 500 (the 500 flush was killed mid-transaction) | 2,109,261 |

In 300 of the 500 lines of the tool-call corpus, the prompt/`<tool_call>` separator is a
literal `\n` (backslash + n) rather than a newline. Those snippets are slightly off the
model's training format.

## What is missing

| Item | Status |
|---|---|
| Screening outputs (capability/boundary/stress slices with margins) | lost; reproducible with `screen_circuit_eval_prompts.ts` |
| Batch circuit-eval outputs and plots (`scripts/out/*.json`) | lost; reproducible with `batch-eval-subgraph.ts` + `analyze_circuit_eval.py` |
| Circuits saved after January 2026 | lost (only `test1` is in the recovered DB) |
| Plan step 3: automatic top-K circuit selection, random-subgraph baseline, bootstrap stability | **not implemented** |
| Plan step 4: automated feature naming / supernode clustering, systematic base-vs-delta analysis | not done; Section 4 is a single-circuit case study |
| Delta checkpoints other than d_new ∈ {32, 128} at pctl 0.5 | lost (HF has `damaoo/gemma2b-toolformer-delta-transcoders` d32/d128 @ p0.5 and the fused d32 @ p0.5 set used in this phase) |
