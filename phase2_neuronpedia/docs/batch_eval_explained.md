# How `batch-eval-subgraph.ts` works

This document explains what `scripts/batch-eval-subgraph.ts` does: how it applies a circuit
selected by hand to a batch of new evaluation prompts, and what each of its output metrics
means.

The script does not ask "do the same node ids appear in the new prompts?". It asks:

> A circuit pinned by hand on one source prompt: when transferred to other prompts, does it
> still explain the key output behaviour in their graphs?

"Explain" is measured with scoring functions defined on the attribution graph, not by eye.

## Pipeline at a glance

1. Load from the database a subgraph saved on a source graph, i.e. the hand-selected circuit.
2. Turn it into a **relative template**: keep each selected feature's `(layer, feature)` and its
   position offset from the target logit.
3. Generate a new attribution graph for every evaluation prompt.
4. Map the template onto the new graph to get the corresponding set of pinned features.
5. Summarise the new graph's logit nodes into a behaviour summary: is it leaning towards
   tool-call or towards echo-first?
6. Score the new graph three ways:
   - default, logit-probability-weighted `replacement_full` / `completeness_full` and
     `replacement_subgraph` / `completeness_subgraph`;
   - `*_gate` scores that target only the gate token `<`;
   - `*_signed_margin` scores that target the signed margin `P("<") − P(echoFirst)`.
7. Report mapping-quality metrics (`matched`, `exactCtxMatched`, `coverage`) together with the
   behaviour summary.

In the end the script answers:

> Can the structure of the hand-picked circuit be found on other prompts, how much of it, and
> once found, how much of the new prompt's default output, its tool-call gate, and its
> tool-call-vs-echo decision does it still explain?

## 1. Inputs

Key command-line arguments:

- `--baseUrl`: the local webapp; the script calls its `/api/graph/generate`
- `--modelId`: model used to generate graphs
- `--subgraphName`: name of the saved circuit
- `--prompts`: evaluation prompt file
- `--out`: output JSON path
- `--sourceSlug`: optional source graph; if omitted, the most recent subgraph saved under
  `subgraphName` is used

Graph-generation hyperparameters can also be passed:

- `--sourceSetName`
- `--maxNLogits`
- `--desiredLogitProb`
- `--nodeThreshold`
- `--edgeThreshold`
- `--maxFeatureNodes`
- `--maxRetries`
- `--initialBackoffMs`

These do not change the scoring formulas, but they change what the generated graph looks
like and therefore affect every metric indirectly.

## 2. Reading the evaluation prompts

Three input formats are accepted:

1. strict JSONL, where each object has at least `prompt`;
2. JSONL where each line is a JSON string;
3. plain text, one prompt per line.

A line that is not valid JSON is treated as a plain-text prompt.

A typical line of `prompts_circuit_eval_v3.jsonl`:

```json
{"label": "pos_seen_arith_div", "prompt": "I'm stuck on this calculation: 865 / 395.\n", "meta": {"family": "pos_seen", "expectedMode": "tool_call"}}
```

- `prompt` is the text sent to the graph generator;
- `label` is used only for grouping and analysis;
- `meta` is used to infer the expected behaviour but is not copied to the output.

### How `meta` determines the expected behaviour

`expectedBehavior` is resolved in this order:

1. `meta.expectedBehavior`
2. `meta.expectedMode`
3. `label`

Rules:

- if `expectedBehavior` is already `tool_call` or `echo_first`, use it;
- else if `expectedMode === 'tool_call'`, expect `tool_call`;
- else if `expectedMode === 'echo'`, expect `echo_first`;
- else fall back to the label: `pos_*` means `tool_call`, anything else `echo_first`.

So besides "how well does the graph explain the output", the script also tracks:

> Was this prompt supposed to push the model towards tool-call or towards echo-first?

### Trailing newline

Each prompt goes through `ensureTrailingNewline`, which appends `\n` if missing. The
fine-tuned model makes its decision at the token after the prompt's newline, so inputs are
normalised to that format. `promptNewlineAdded` records whether a newline was added.

## 3. What the "hand-selected circuit" is

The circuit is not a separate JSON file. It lives in the database table
`GraphMetadataSubgraph`, whose key fields are:

- `graphMetadataId`: the graph it belongs to
- `displayName`: the name given when saving
- `pinnedIds`: the node ids pinned in the graph UI

So the circuit is:

> a set of nodes pinned by hand on one source graph.

The source graph is resolved as follows:

- with `--sourceSlug`, look up exactly `modelId + sourceSlug`;
- otherwise use the most recent save of `subgraphName` for this model.

Then its `pinnedIds` are loaded.

## 4. Loading the source graph

The source graph is read from the stored graph JSON, not regenerated:

1. take the graph's `url` from `GraphMetadata`;
2. if `url` starts with `/`, read it from the local `public/` directory;
3. otherwise fall back to `public/local-graphs/${modelId}-${slug}.json`.

The source circuit is therefore fixed to an existing graph.

## 5. Turning the circuit into a template

This is the central step.

### 5.1 The source graph's target logit

`getTargetLogitNode` searches the logit nodes:

1. prefer the node with `is_target_logit === true`;
2. otherwise take the logit with the largest `token_prob`.

All positional alignment below is relative to this target output token.

### 5.2 Target token from the clerp

`extractTargetTokenFromClerp` extracts the token from strings such as

```text
Output "<" (p=0.997)
```

with a single regex, to record which token the graph traces (`sourceTargetToken` in the
output, `targetToken` per prompt).

### 5.3 Building the template

`buildTemplateFromSourceGraph` iterates over `pinnedIds` and keeps a node only if:

- it exists in the source graph;
- `feature_type === 'cross layer transcoder'`;
- `feature` is numeric.

The circuit is therefore made of transcoder features only. Embedding, logit and error nodes
are not part of the template.

For each kept feature the template records:

- `layer`
- `feature`
- `ctxDeltaFromLogit = sourceLogitCtxIdx − node.ctx_idx`

`ctxDeltaFromLogit` is a position **relative to the target logit**, not an absolute
position. If the target logit is at `ctx_idx = 12` and a feature at `ctx_idx = 10`, then
`ctxDeltaFromLogit = 2`, i.e. the template remembers "two tokens before the output position".

### 5.4 Deduplication

Template entries are deduplicated on

```text
layer | feature | ctxDeltaFromLogit
```

The template is thus a relative-position feature list, not a copy of source node ids.

## 6. Generating the new graphs

For each evaluation prompt the script calls

```text
POST {baseUrl}/api/graph/generate
```

with `prompt`, `modelId`, optional `sourceSetName`, `maxNLogits`, `desiredLogitProb`,
`nodeThreshold`, `edgeThreshold` and `maxFeatureNodes`.

### 6.1 What `/api/graph/generate` does

1. validates the parameters;
2. tokenises with `getGraphTokenize(...)`, selects salient logits, and checks the prompt is at
   most 64 tokens;
3. generates a slug if none is given;
4. calls the graph server's `generate-graph` endpoint;
5. saves the returned graph JSON under `public/local-graphs/`;
6. writes the graph metadata to the database;
7. returns the full graph to the caller.

The script also relies on the graph metadata fields `prompt` and `prompt_tokens`; the latter
is used to recover `echoFirstToken` (Section 9).

### 6.2 Effect of the generation parameters

The script does not trace graphs itself. Roughly:

- `maxNLogits` and `desiredLogitProb` decide which output logits are traced;
- `nodeThreshold` and `edgeThreshold` control pruning;
- `maxFeatureNodes` caps the graph size.

What is evaluated is therefore not "the model's raw mechanism" but:

> how much of the attribution graph, produced under these generation and pruning settings,
> the circuit still explains.

### 6.3 Retries

If `/api/graph/generate` returns `RUNPOD_BUSY` or HTTP 503, the script retries with
exponential backoff. A failed prompt can be a busy server, not a model problem.

## 7. Mapping the circuit onto a new graph

The script does not require identical node ids, which differ between prompts. It aligns the
template instead.

### 7.1 The new graph's target logit

Found the same way as for the source graph (`is_target_logit`, else highest `token_prob`).
All positions below are relative to this logit.

### 7.2 Index by `(layer, feature)`

All transcoder feature nodes in the new graph are bucketed by

```text
key = layer | feature
```

and each bucket is sorted by `ctx_idx`.

### 7.3 Nearest node by relative offset

For each template entry, the ideal position in the new graph is

$$
desiredCtxIdx = targetLogitCtxIdx - ctxDeltaFromLogit
$$

and among the candidates with the same `(layer, feature)`, the one whose `ctx_idx` is closest
to `desiredCtxIdx` is chosen. The underlying assumption:

> if a circuit transfers, it should reappear on the new prompt as the same feature in the
> same layer, at roughly the same position relative to the output.

### 7.4 Mapping-quality metrics

- `templateCount`: number of template entries, roughly the circuit's size;
- `matched`: entries for which at least one node with the same `(layer, feature)` exists in
  the new graph, i.e. "does the feature still appear?";
- `exactCtxMatched`: entries whose matched node is at exactly the ideal position, i.e. "does
  it still appear at the same position relative to the output?";
- `coverage = matched / templateCount`.

High `matched` does not mean the circuit explains the new prompt; it only means the template
can be mapped. Explanatory power is measured by the replacement and completeness scores.

## 8. Behaviour summary and targeted scores

On top of the default scores, the script:

1. summarises each graph's behaviour: does it support tool-call or echo-first?
2. computes a gate score for `<` and a signed-margin score.

This lets it answer, besides "how well does the subgraph explain the graph in general":

> Does the subgraph specifically explain the tool-call gate `<`?
>
> Does it explain the advantage of `<` over the echo-first alternative?

## 9. The behaviour summary

`summarizeBehaviorFromGraph(...)` runs on every generated graph.

### 9.1 Logit tokens

For all nodes with `feature_type === 'logit'`, the token text is taken from `node.logitToken`
or else parsed from `node.clerp`. Sorted by `token_prob`, this gives:

- `topToken`, `topProb`
- `topNonAngleToken`, `topNonAngleProb`

`topToken` is the most probable token *among the traced logits*. It usually equals
`targetToken`, but the two are conceptually different: `targetToken` is what
`getTargetLogitNode` selected.

### 9.2 `echoFirstToken`

Taken from the graph metadata's `prompt_tokens` (not from a separate forward pass), skipping
special tokens (`<bos>`, `<eos>`, `<pad>`, `<unk>`, `<start_of_turn>`, `<end_of_turn>`).
The first remaining token is `echoFirstToken`:

> the token the model would emit first if it echoed the prompt, i.e. the main competitor of `<`.

### 9.3 `probMargin` and `signedProbMargin`

$$
probMargin = P(\texttt{"<"}) - P(echoFirst)
$$

`<` is hard-coded as the gate token (the first token of `<tool_call>`). Positive values lean
towards tool-call, negative towards echo.

$$
signedProbMargin = \begin{cases}
probMargin & \text{if expectedBehavior = tool\_call} \\
-probMargin & \text{if expectedBehavior = echo\_first}
\end{cases}
$$

Larger `signedProbMargin` means the graph's logits move further in the expected direction;
negative values mean the wrong direction.

### 9.4 `decisionByTop1`

A coarse top-1 decision:

- `topToken === '<'` → `tool_call`
- else `topToken === echoFirstToken` → `echo_first`
- else `other`

## 10. Scores computed for each graph

Each score group is computed for the full graph and for the mapped subgraph.

### 10.1 Default scores

1. `full = computeGraphScoresFromGraphData(graph, [])`
2. `sub = computeGraphScoresFromGraphData(graph, mapped.pinnedIds)`

Output: `replacement_full`, `completeness_full`, `replacement_subgraph`,
`completeness_subgraph`. Logit weights are each logit node's own `token_prob`.

### 10.2 Gate scores

`computeGraphScoresFromGraphDataWithLogitWeightOverride(...)` with logit weight 1 on `<` and 0
elsewhere:

> if only the gate token `<` matters, how much do the full graph and the subgraph explain?

Output: `replacement_full_gate`, `completeness_full_gate`, `replacement_subgraph_gate`,
`completeness_subgraph_gate`. They are `null` if `<` is not among the traced logits.

### 10.3 Signed-margin scores

Logit weights: `+1` on `<`, `−1` on `echoFirstToken`, `0` elsewhere, computed with
`computeGraphScoresFromGraphDataWithSignedLogitWeightOverride(...)`. Unlike the plain
override, it aggregates influence and output influence in **absolute value**: for a signed
margin, supporting `<` and suppressing the echo token both push towards the correct
decision and should both count as explanation instead of cancelling out.

Output: `replacement_full_signed_margin`, `completeness_full_signed_margin`,
`replacement_subgraph_signed_margin`, `completeness_subgraph_signed_margin`. They are `null`
unless both `<` and `echoFirstToken` are traced.

## 11. The graph computation behind the scores

### 11.1 Node order

Nodes are sorted by type: `cross layer transcoder`, `mlp reconstruction error`, `embedding`,
`logit`. This makes it possible to slice feature, error, token and logit nodes by index range.

### 11.2 Adjacency matrix

For each edge, `adjacencyMatrix[dst][src] = edge.weight`; row *i* lists the inputs of node *i*.

### 11.3 Row normalisation in absolute value

`normalizeMatrix` takes absolute values and divides each row by its sum, turning each node's
inputs into a distribution of "share of input from each upstream source". Signs are discarded,
so the scores measure explained share, not net positive contribution.

### 11.4 Influence

`computeInfluence(A, logitWeights)` starts from the logit weights (each logit's `token_prob`
by default) and propagates them upstream through the normalised graph:

$$
I^{(1)} = wA,\quad I^{(2)} = I^{(1)}A,\quad I^{(3)} = I^{(2)}A,\ \dots
$$

The sum over all iterations is each node's total influence on the output. Nodes on many
important paths get high influence; paths that end in error nodes give the error nodes
influence.

## 12. Full graph vs subgraph

### 12.1 Full graph

With `pinnedIds = []` the whole graph is scored:

> how well does this attribution graph as a whole explain the output?

### 12.2 Subgraph

With `mapped.pinnedIds`, the remaining nodes are not simply dropped. Instead:

> every transcoder feature outside the subgraph is treated as unexplained error.

Concretely:

1. find the `mlp reconstruction error` node for each `(layer, ctx_idx)`;
2. for every transcoder feature not in the pinned set, merge its outgoing edges into the
   error node at the same position, then zero its row and column;
3. if no error node exists at that position, just zero the feature.

So features the circuit does not cover are not "irrelevant background" but "residual not
explained by this circuit". `replacement_subgraph` / `completeness_subgraph` ask:

> if only the mapped features may explain the output, how much explanatory power survives?

## 13. Replacement score

Influence is split by node type into `tokenInfluence` (on embedding nodes) and
`errorInfluence` (on error nodes):

$$
replacementScore = \frac{tokenInfluence}{tokenInfluence + errorInfluence}
$$

> Tracing back from the output, what share of the end-to-end explanation reaches the input
> tokens rather than an error node?

High: the output can be explained through feature paths all the way back to the input.
Low: much of the influence ends in error nodes.

It is **strict** because only endpoints count. Reaching a token is success, reaching an error
node is failure, and partially explained intermediate nodes earn nothing extra.

## 14. Completeness score

For each node:

$$
nonErrorFraction_i = 1 - \sum_{e \in error} A_{i,e}
$$

the share of its input that does not come from error nodes. It is averaged with weights given
by each node's importance for the output:

$$
outputInfluence_i = nodeInfluence_i + logitWeight_i
$$

$$
completenessScore = \frac{\sum_i nonErrorFraction_i \cdot outputInfluence_i}{\sum_i outputInfluence_i}
$$

> For the nodes that matter for the output, how much of their own input is explained by
> features and tokens rather than error?

It is **more lenient** than replacement because it credits partial explanation. A node with
80% feature/token input and 20% error input still scores 0.8 locally, while in the
end-to-end replacement view that error share can drag the total down considerably.

- `replacement`: strict, does the explanation reach the input tokens?
- `completeness`: lenient, how much of the important part of the graph is covered by
  feature-based explanation?

## 15. Reading the metrics as evidence of generalisation

The script measures several things at once.

### 15.1 Structural transferability

`matched`, `exactCtxMatched`, `coverage`:

> can the source circuit's features, at their relative positions, be found on the new prompt?

### 15.2 Whether the new prompt has a good graph at all

`replacement_full`, `completeness_full`:

> ignoring the circuit, how well does the new prompt's graph explain its output?

This matters: if the full graph is poor, a poor subgraph score does not by itself mean the
circuit failed to generalise.

### 15.3 How much the circuit retains

`replacement_subgraph`, `completeness_subgraph`:

> once all features outside the mapped circuit count as error, how much does the circuit
> explain on its own?

A circuit that generalises should show reasonable `coverage` and subgraph scores that do not
collapse relative to the full graph.

### 15.4 Whether it retains the tool-call gate

`*_gate`: does the subgraph still explain the gate token `<`? This separates a subgraph
whose general explanatory power drops from one that explains the whole graph only modestly
but remains specialised for the gate.

### 15.5 Whether it retains the decision direction

`*_signed_margin`: does the subgraph still explain why `<` beats the echo-first alternative?
This is finer than `*_gate`, since it asks whether `<` wins against the most natural text
continuation, not only whether `<` is supported.

Signs of poor generalisation:

- low `coverage`: most features cannot be found on the new prompt;
- acceptable `coverage` but low `replacement_subgraph`: the features are present but no longer
  the main explanatory path;
- `targetToken` changes: the new graph traces a different output token;
- low `replacement_subgraph_gate`: the subgraph no longer explains `<`;
- low `replacement_subgraph_signed_margin`: the subgraph no longer supports "tool-call beats
  echo-first".

## 16. Output fields

Two files are written: a JSON file and a CSV with the same name.

### 16.1 JSON `meta`

- `modelId`: model used
- `sourceSlug`: slug of the source graph
- `subgraphName`, `subgraphId`: the subgraph used
- `sourceTargetToken`: target token of the source graph
- `gateToken`: gate token, fixed to `<`
- `templateSize`: `template.length`
- `generatedAt`: timestamp
- `scoringSemantics`: notes that subgraph scoring merges unselected features into error nodes
- `behaviorSemantics`: definitions of `probMargin` / `signedProbMargin`, and that signed-margin
  scores need both target logits in the graph

### 16.2 Per-prompt fields

- `index`, `label`: prompt number and label from the input
- `promptNewlineAdded`: whether a newline was appended
- `slug`, `url`: the new graph and its viewer link
- `targetToken`, `targetProb`: target token traced in the new graph and its probability
- `topToken`, `topProb`: most probable traced logit token
- `gateToken`: always `<`
- `echoFirstToken`: first non-special token of `prompt_tokens`
- `expectedBehavior`: `tool_call` or `echo_first`
- `decisionByTop1`: coarse top-1 decision
- `probAngleBracket`, `rankAngleBracket`: probability and rank of `<` among the traced logits
- `probEchoFirst`, `rankEchoFirst`: the same for `echoFirstToken`
- `probMargin`: `P("<") − P(echoFirst)`
- `signedProbMargin`: margin with the sign flipped by `expectedBehavior`
- `topNonAngleToken`, `topNonAngleProb`: most probable token other than `<`
- `salientLogitCount`: number of traced logits
- `salientCumulativeProbability`: their total probability
- `hasAngleBracketTarget`, `hasEchoFirstTarget`, `hasSignedMarginTarget`: whether `<`,
  `echoFirstToken`, or both are traced
- `replacement_full_gate`, `completeness_full_gate`, `replacement_subgraph_gate`,
  `completeness_subgraph_gate`: gate scores
- `replacement_full_signed_margin`, `completeness_full_signed_margin`,
  `replacement_subgraph_signed_margin`, `completeness_subgraph_signed_margin`: signed-margin
  scores
- `replacement_full`, `completeness_full`, `replacement_subgraph`, `completeness_subgraph`:
  default scores
- `templateCount`, `matched`, `exactCtxMatched`, `coverage`: mapping quality
- `error`: error message if the prompt failed

## 17. Caveats when reading results

### 17.1 A graph-level proxy, not a causal test

The script performs no interventions or ablations. It maps structure and compares scores on
attribution graphs:

> is the hand-selected circuit still a good explanatory subgraph in the graph representation?

This is not causal proof.

### 17.2 Strong dependence on generation settings

Because each graph is pruned according to `maxNLogits`, `desiredLogitProb`, `nodeThreshold`,
`edgeThreshold` and `maxFeatureNodes`, different thresholds, source sets or logit-selection
strategies can change `coverage`, `replacement_*` and `completeness_*` substantially.

### 17.3 `targetToken`, `topToken` and `gateToken` differ

- `targetToken`: the logit chosen by the graph generator
- `topToken`: the most probable traced logit
- `gateToken`: `<`, fixed by the script for gate analysis

They sometimes coincide and sometimes do not. Before reading any score, check which token the
graph traces, which token is most probable, and which token the gate score uses.

### 17.4 `targetToken` can drift

On some prompts the traced target was not `<` but tokens such as `Return`, `Please`,
`Consider` or `In`. The graph generator's main output target has then changed, and even if
the template maps, it may be explaining a different output tendency rather than the
tool-call behaviour. Such prompts need to be read separately.

### 17.5 Empty `*_gate` and `*_signed_margin`

If `<` is not traced, `*_gate` scores are empty. If `<` and `echoFirstToken` are not both
traced, `*_signed_margin` scores are empty. This is not a bug: the targeted scores have no
logit support in those graphs.

## 18. Suggested reading order

**A. Which way does the graph lean?** `expectedBehavior`, `decisionByTop1`, `topToken`,
`probMargin`, `signedProbMargin`: does the graph support tool-call or echo-first, and does
that match the prompt's expected behaviour?

**B. Is it the same kind of output?** `targetToken`, `targetProb`. If the target drifted to
an unrelated token, default scores are not comparable with the source prompt.

**C. Does the template transfer?** `coverage`, `matched`, `exactCtxMatched`: structural
generalisation.

**D. How much explanatory power is retained?** `replacement_subgraph` vs
`replacement_full`, `completeness_subgraph` vs `completeness_full`. Subgraph scores close to
the full graph mean the circuit is still close to the main explanatory subgraph. High full
and low subgraph scores mean the new prompt has a good graph, but the features carried over
from the source prompt are no longer its main mechanism.

**E. Is the gate retained?** `replacement_subgraph_gate` vs `replacement_full_gate`,
`completeness_subgraph_gate` vs `completeness_full_gate`. Modest default scores with high gate
scores suggest a circuit that does not explain the whole graph but is still key to the
tool-call gate.

**F. Is the signed margin retained?** `replacement_subgraph_signed_margin` vs
`replacement_full_signed_margin`, and likewise for completeness. High values mean the
circuit still helps `<` beat the echo-first competitor. This measures the decision boundary
rather than a single output token.

## 19. Summary

The script turns a circuit hand-picked on one source prompt into a template of
"feature identity + position relative to the target logit". It maps that template onto
attribution graphs generated for a batch of new prompts, then measures how much the mapped
circuit explains of the whole graph, of the tool-call gate `<`, and of the margin of `<` over
the echo-first alternative.

The most informative combination is usually:

1. whether `decisionByTop1` and `signedProbMargin` still point in the expected direction;
2. whether `targetToken` is still the same kind of output;
3. whether `coverage` is high enough;
4. `replacement_subgraph / replacement_full`;
5. `replacement_subgraph_gate / replacement_full_gate`;
6. `replacement_subgraph_signed_margin / replacement_full_signed_margin`.

Together these form a reasonably complete case for whether a circuit generalises.
