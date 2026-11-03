# How does a fine-tuned Gemma-2-2B decide to call a tool?

*Delta transcoders and attribution graphs for a calculator Toolformer.*

**Qiren Chen** · project period Aug 2025 – Apr 2026 · write-up Nov 2026

---

## Summary

I LoRA-fine-tuned Gemma-2-2B to emit `<tool_call>calculator(expr)</tool_call>` for arithmetic
requests and to echo every other prompt verbatim. I then studied the model's internals with
Gemma-Scope transcoders.

1. **Off-the-shelf transcoders do not fit the fine-tuned model, and a small "delta" patch fixes
   that.** I froze the Gemma-Scope transcoder and trained 32–128 extra sparse features per
   layer. This cut the CE increase from splicing the transcoder into the model from about
   12.6 nats to 0.11–0.36 nats, and raised reconstruction cosine from about 0.45 to 0.60–0.71.
2. **Tool calling lives mostly in the delta features, concentrated in middle layers.** With the
   base transcoder alone, the tool-call rate falls to 0.00–0.15 at layers 12, 16, 20 and 25.
   Adding the delta back largely restores it, and scaling the delta up and down at layers 20 and 24
   moves the tool-call rate roughly monotonically on in-distribution prompts.
3. **The decision is a brittle heuristic gate, not semantic tool use.** Near-OOD probing shows
   the choice between `<` and echoing flips on lower-casing combined with small numbers, a
   missing `?` on one specific template, or the *position* of "Do not use any tool". The
   meaning of the instruction does not drive it.
4. **An attribution graph shows where the gate sits.** In a hand-traced circuit for
   `Calculate 3 + 5.`, delta features (0.2% of the dictionary) are 9 of 23 features.
   Every circuit feature in layers 17–22 at the decision position is a delta feature, and they
   include detectors for specific training templates ("I'm trying to figure out …",
   "value of / difference between"). Base features mostly handle lexical cues at layer 0 and the
   output syntax at layers 24–25.

Taken together: the fine-tune taught the model a set of **template detectors** in the middle
layers, built on top of the base model's lexical features and feeding into its existing
output-formatting features. Prompts the detectors do not recognise lose the gate, which
explains why the behaviour generalises poorly.

The project stopped before its last stage, an automatic search for circuits that generalise
across prompts, with random baselines. The evaluation code for that stage is complete, but
some outputs were lost (see [Limitations](#7-limitations-and-what-was-lost)).

---

## 1. Setup

**Task and model.** I fine-tuned `google/gemma-2-2b` with LoRA (r = 16, α = 32, all attention
and MLP projections, 5 epochs, lr 2e-5). The 10k training examples are half positives and half
negatives. A positive is an arithmetic request, e.g. "I'd like to know the result of
649 / 821.", followed by `<tool_call>calculator(649/821)</tool_call>`. A negative is a C4
sentence the model must echo verbatim. So after the prompt's trailing newline the model
has exactly two modes, and the **first generated token decides between them**: `<` (tool
call) versus the prompt's first token (echo). The merged model is
[`damaoo/gemma2b-toolformer`](https://huggingface.co/damaoo/gemma2b-toolformer).

**Transcoders.** Gemma-Scope provides per-layer transcoders (MLP-in → MLP-out, JumpReLU,
width 16k) trained on the *base* model. The **delta transcoder** keeps such a transcoder
frozen and adds `d_new` new sparse features trained only on the fine-tuned model's
MLP outputs:

```
recon(x) = base(x) + delta(x)        loss = ‖recon(x) − mlp_ft(x)‖² + λ‖delta_acts‖₁
```

Fusing base and delta gives a single dictionary per layer, with base features at indices
0–16383 and delta features from 16384 on. That makes every feature in later analyses
identifiable as base or delta. Checkpoints:
[`damaoo/gemma2b-toolformer-delta-transcoders`](https://huggingface.co/damaoo/gemma2b-toolformer-delta-transcoders),
[`damaoo/gemma2b-toolformer-fused-transcoders`](https://huggingface.co/damaoo/gemma2b-toolformer-fused-transcoders).

## 2. Phase 1: does the delta fix the transcoder, and does it carry tool use?

### 2.1 Reconstruction quality (Experiment 1)

Each condition replaces one layer's MLP with the transcoder reconstruction and leaves the
rest of the model intact. I swept `d_new ∈ {32, 64, 128}` and `p ∈ {0.25, 0.5, 0.75}`, which
picks among Gemma-Scope's variants for each layer sorted by average L0 (higher p means a denser
base transcoder). Metrics are averaged over all 26 layers on held-out validation prompts.

| d_new | p | ΔCE base only | ΔCE base + delta | cos base only | cos base + delta |
|---|---|---|---|---|---|
| 32 | 0.50 | 12.60 | 0.19 | 0.46 | 0.65 |
| 64 | 0.50 | 12.60 | 0.16 | 0.46 | 0.68 |
| 128 | 0.25 | 13.84 | **0.11** | 0.47 | 0.70 |
| 128 | 0.50 | 12.60 | 0.13 | 0.46 | **0.71** |
| 128 | 0.75 | 12.54 | 0.25 | 0.44 | 0.67 |

Spliced in as-is, the base transcoder is catastrophic: about 12.6 nats of extra loss. A few
dozen delta features remove almost all of it. Larger `d_new` helps with diminishing returns,
and the densest base variant (p = 0.75) is hardest to patch.
(Per-combination results: `src/exp1/aggregate/exp1_combo_summary.csv`.)

### 2.2 Removing and scaling the delta (Experiments 2A and 2B)

There are three prompt subsets of 20 prompts each: `tool` (in-distribution requests),
`copy` (number-containing sentences to echo) and `tool_ood` (unseen phrasings such as
"Prepare a tool call for the calculator so that it can evaluate 553+487"). I intervened on
one layer at a time, with `d_new = 128`, `p = 0.5`.

Tool-call rate on `tool` (raw model = 1.00 at every layer):

| Layer | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 25 |
|---|---|---|---|---|---|---|---|---|
| base only (`no_delta`) | 1.00 | 0.45 | 1.00 | **0.00** | **0.00** | **0.15** | 0.45 | **0.10** |
| base + delta (`full`) | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.70 | 1.00 | 0.85 |

Scaling the delta at a single layer (`base + α·delta`) gives:

| α | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| layer 20, `tool` | 0.15 | 0.30 | 0.30 | 0.70 | 0.70 |
| layer 24, `tool` | 0.45 | 0.50 | 0.60 | 0.90 | 1.00 |
| layer 20, `copy` (should stay 0) | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 |

The delta carries tool behaviour, mainly in **layers 12–20**, and turning it up raises tool
calls on in-distribution prompts without making the model call tools on copy prompts. On
`tool_ood` the relationship with α is not monotone. Even the unmodified model only produced
a correct call on 35% of OOD prompts: it emitted `<tool_call>` every time, but the format was
valid only 40% of the time.

### 2.3 What Phase 1 could not settle

The original conclusion was that the model shows "some semantic generalization". Two things
weakened it. The OOD subset was small and mixed failures of very different kinds, and
layer-level interventions say *where* the behaviour is but not *what* computes it. Phase 2
therefore looked at single prompts and single features.

## 3. Phase 2: the gate is brittle

I hosted Neuronpedia myself, served by a graph backend running circuit-tracer with the fused
transcoders on the fine-tuned model. This allowed probing hand-written variants of training
templates and reading the next-token distribution directly
(full log: `phase2_neuronpedia/docs/manual_ood_probing_zh.md`).

| Prompt | Next token |
|---|---|
| `Compute the value of 112 * 309.` | `<` 1.00 |
| `compute the value of 112 * 309.` | `<` 0.80 · `compute` 0.19 |
| `compute the value of 11 * 309.` | `<` 0.63 · `compute` 0.36 |
| `compute the value of 1 * 309.` | **`compute` 0.69** · `<` 0.31 |
| `compute the value of 1 * 39.` | `<` 0.77 · `compute` 0.23 |
| `compute the value of 1 * 3.` | **`compute` 0.53** · `<` 0.47 |
| `What’s the difference between 598 and 914?` | `<` 0.98 |
| `What’s the difference between 598 and 914` | **`What` 0.68** · `<` 0.29 |
| `What is 534 + 267` | `<` 1.00 |
| `…914? Do not use any tool.` | `<` 0.66 (wrong) |
| `Do not use any tool. …914?` | `Do` 0.98 (correct echo) |
| `…914? Do not use any tool.␠` (trailing space) | `What` 0.60 (correct echo) |
| `Please, compute the value of 112 * 309.` | `<` 1.00 |

Three properties stand out. First, **most perturbations are harmless**. Failures concentrate
in a few interactions specific to one template: removing the `?` breaks only "difference
between". Second, **factors interact**: lower case and small operands each do little, but
together they flip the decision, and not monotonically in operand size. Third,
**instructions work by position**: a leading "Do not use any tool." wins because its first
token is what echoing would output, while the same sentence as a suffix is largely ignored.
A single trailing space can reverse the outcome.

The most economical explanation is a competition at the last position between three groups of
features. Seen-template features push `<`; features of the prompt's first token push echo;
and token-boundary features (punctuation, whitespace, digit tokenisation) shift the balance.
Nothing in the data requires a global representation of "the user wants a calculation".

## 4. Phase 2: an evaluation protocol for the gate

To test whether a circuit found on one prompt explains others, I built an evaluation
pipeline on top of Neuronpedia's graph API (design: `phase2_neuronpedia/README.md` §3).

* **Labels on several axes instead of an ID/OOD split.** Each prompt records its intent,
  distribution, expected behaviour and variant group. One forward pass then sorts it by what
  the model *actually* did: **capability** (correct with margin), **boundary** (correct but
  close) or **stress** (wrong). With this scheme, two near-identical prompts landing in
  different slices are no longer a labelling problem. They are exactly the brittle boundary
  under study.
* **A gate-specific target.** The decision variable is the margin `P("<") − P(echoFirst)`.
  Its sign is flipped for prompts that should echo, so tool and echo prompts share one
  "correct direction". Circuits are scored with Neuronpedia's replacement and completeness
  metrics, re-weighted to a gate-only target (+1 on `<`) or a signed-margin target
  (+1 on `<`, −1 on the echo token).
* **Circuit transfer.** A hand-pinned circuit is turned into a template of
  `(layer, feature, offset from the target position)`. The template is matched in each new
  prompt's graph, and the match is scored as above.

The generator, screening, batch evaluator, scoring and analysis scripts were all recovered,
along with a regenerated 50-prompt evaluation set. The screening and batch-evaluation outputs
were not, so this section describes a protocol, not results.

## 5. Phase 2: inside one gate decision

The circuit that survived traces the `<` logit (p = 0.997) for `Calculate 3 + 5.`. It has
23 transcoder features, which I grouped into six labelled supernodes in the Neuronpedia UI.
The stored top-activating snippets make it possible to check these labels. For example,
delta feature L20 #16399, labelled "'trying to figure out' intent matcher", has top activations
that are all `I'm trying to figure out a op b.`, one of the training templates, and its
activation rises across the sentence to peak at the final period.

| Stage | Position | Features | Base / delta |
|---|---|---|---|
| Lexical and position cues ("calculate", "problem", "help needed", "first word") | `Calculate` | L0–L3 | 6 base, 3 delta |
| Template, expression and number detectors ("value of / difference between", "trying to figure out", core math expression, generic numbers), plus one tool-syntax feature | final `↵` | **L17–L22** | **0 base, 6 delta** |
| Math and number features, structural anchors | final `↵` | L7, L8, L15, L24 | 4 base |
| Tool-call syntax ("opening tags", "argument filling") | final `↵` | L24–L25 | 4 base |

Across the whole pruned graph (890 feature nodes), delta features are 6.4% of nodes, about 33
times their share of the dictionary. They carry 19.8% of outgoing edge weight and make up 12 of
the 20 nodes in layers 17–23.

This matches Phase 1 from a different direction. The layers where removing the delta
destroyed tool calling (12–20) are the layers where the attribution graph routes the
decision through delta features. The features found there are **detectors for specific
training templates**, which is what the brittleness in Section 3 predicts: a prompt that
none of the template detectors recognise loses its push towards `<`. The fine-tune did not
need to learn the output format. It reuses base features at layers 24–25 for that and adds
a recogniser in front of them.

An interactive version of the circuit, with every delta feature's top activations, is at
[`docs/circuit-viewer/`](docs/circuit-viewer/).

## 6. What this means

* **For the original question.** On this evidence the model does not "use a tool" in any
  semantic sense. It learned a trigger that fires on the surface forms of its training
  prompts.
* **For the method.** Delta transcoders turned out to be more useful for interpretation than
  for efficiency. Because the fine-tune's new computation is isolated in a few labelled
  features, both questions — "where does the behaviour live?" (layer interventions) and
  "what does it compute?" (attribution graphs) — reduce to looking at a small, inspectable
  set of features.
* **For tool-use fine-tuning.** A small SFT dataset built from templates yields a decision
  boundary that follows the templates. Adversarial phrasing, varied casing and number
  formats, and negatives that *mention* tools are the obvious data fixes. The circuit view
  suggests that whether the fix worked can be checked by looking at what the L17–L22 delta
  features respond to.

## 7. Limitations and what was lost

* **One circuit, one prompt.** The base/delta division of labour in Section 5 is a case
  study. The batch evaluation built to test it across 50 prompts ran, but its outputs were
  lost when a scratch filesystem was purged. They can be regenerated with the published code.
* **Small behavioural samples.** Phase 1 used 20 prompts per subset, and a few rates move in
  steps of 0.05.
* **Not done:** automatic top-K circuit selection with random-subgraph baselines and bootstrap
  stability (plan step 3); automatic feature naming and supernode clustering, and a systematic
  base-vs-delta comparison over many graphs (plan step 4). The open question that step 3 would
  have answered: if the gate depends on shallow features specific to each prompt, a selected
  circuit may not transfer better than a random one of the same size. That would itself be
  evidence for the heuristic-gate reading.
* **Attribution graphs are approximations.** Scores depend on graph pruning thresholds and
  on the transcoder's reconstruction error, which is never zero (Section 2.1).
* **Data quirks.** The tool-call activation corpus contains a literal `\n` instead of a
  newline in 300 of 500 lines, and both activation-collection jobs stopped early; see the
  dataset card.

## 8. Artefacts

| What | Where |
|---|---|
| Fine-tuning, delta-transcoder training, Experiments 1 and 2 | `src/`, `data/` |
| Neuronpedia changes, evaluation pipeline, design docs | `phase2_neuronpedia/` |
| Hand-traced circuit and its source graph | `phase2_neuronpedia/results/test1_circuit/` |
| Interactive circuit and delta-feature viewer | `docs/circuit-viewer/` |
| Model, delta and fused transcoders | HF: `damaoo/gemma2b-toolformer`, `…-delta-transcoders`, `…-fused-transcoders` |
| Top-activation dataset (4.2M snippets, 426,779 features) | HF dataset; card in `phase2_neuronpedia/hf_dataset_card.md` |
