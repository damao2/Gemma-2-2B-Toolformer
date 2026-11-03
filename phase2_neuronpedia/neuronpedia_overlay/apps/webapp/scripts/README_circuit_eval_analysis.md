# Circuit eval: how to think about targets + metrics

This repo’s batch circuit-eval pipeline currently produces per-prompt scores from an attribution graph.

**Key point:** there are *two different (valid) analysis targets*:

1) **Mechanism of tool-call gating** (the decision to enter `<tool_call>...`):
   - Target should be unified across prompts, usually the next-token `"<"`.
   - This supports apples-to-apples generalization tests across POS/NEG prompt families.

2) **Mechanism of the model’s actual next token** (what it really outputs on that prompt):
   - Target should be whatever token the model is about to emit (often the top-1 next token).
   - This supports human inspection: “why did it echo / why did it start with `Please` / why did it emit `<`?”.

These are not contradictory; they answer different questions.

## What the current batch-eval output means

In `scripts/batch-eval-subgraph.ts`, we select a *target logit node* from the graph:
- If a node has `is_target_logit`, we use it.
- Otherwise, we fallback to the logit node with the highest `token_prob` (top-1).

The output fields:
- `targetToken`: the token for that selected logit node.
- `targetProb`: the probability for that selected logit node.

Therefore `targetToken` can vary across prompts (e.g. `"<"` for tool-call-like prompts, but `"Please"`/`"Return"`/etc. for other prompts).

### Implication

If your research question is specifically about the tool-call gate, you usually want *fixed* `P("<")` for every prompt — not just “probability of whatever token happens to be top-1”.

## Recommended mental model: gating is a *comparison*

For a two-mode model (tool-call vs echo), the “gate” can be understood as a competition between at least two next-token candidates:

- tool-call start token: `"<"`
- echo-mode next token: the next literal token of the prompt (or some natural-language continuation)

So a more faithful decision variable is a **margin**:

$$m = \text{logit}('<') - \text{logit}(t_{echo})$$

Human analysis can then focus on which nodes push the margin up/down.

## Why we include NEG prompts (and what “generalization” means here)

We are **not** looking for a circuit that makes the model tool-call on both POS and NEG prompts.
That would be “behavior-generalization” in the wrong direction (a generic trigger), not selective gating.

Instead, NEG prompts are included to test **selectivity** and **conditional branching**:

- **POS** prompts: the mechanism should generalize so that `P("<")` (or the gate margin) is high.
- **NEG** prompts: the mechanism should generalize so that `P("<")` is low (suppressed), *even if* the prompt mentions tools, quotes tool-call markup, or forbids tool use.

In other words, we want the *same high-level gate* to apply across contexts, but to produce different outcomes depending on the input.

How to read NEG outcomes:
- If NEG still yields top-1 `"<"`, treat it as **leakage**: the model (or your circuit) is not implementing the intended condition.
- If NEG yields a non-`"<"` top-1, that can be good behaviorally; then a gate-focused trace (target `"<"`) asks: **what suppressed tool-call here?**
- If coverage is low on NEG (or any family), it may be a mapping/transfer failure rather than a mechanism failure.

## How to use graphs without mixing objectives

### For evaluation (generalization metrics)
Use a unified target for all POS/NEG prompts:
- Primary: target `"<"` (measure gate strength).
- Optional secondary: target the “echo token” (measure echo strength / leakage).

Interpret Replacement/Completeness only when the target truly corresponds to the gating variable you care about.

### For human story-building
Start from the model’s *actual* next token on that prompt (often top-1):
- POS prompts: often top-1 is `"<"` anyway ⇒ both targets align.
- NEG prompts: top-1 might be an echo token ⇒ tracing that explains “why echo happened”.

Then add a second pass targeted at `"<"` to answer “what suppressed tool-call here?”

In other words: **two traces per prompt** is often ideal:
- Actual-behavior trace (top-1)
- Gate-focused trace (`"<"`)

## Metrics to report (suggested)

From current outputs:
- `angle_bracket_rate`: fraction of prompts where selected `targetToken == "<"` (a proxy for tool-call hit rate).
- `replacement_subgraph`, `completeness_subgraph`: circuit influence for the selected target logit.
- `coverage`: how many template entries could be mapped in the target prompt.

Robust reporting (recommended): median + IQR/quantiles in addition to mean.

## Common pitfalls

- Mixing targets across families: comparing Replacement/Completeness for `"<"` on POS with `"Please"` on NEG is not measuring a single mechanism.
- Low coverage: often indicates mapping/transfer issues, not necessarily circuit non-generalization.
- “NEG” prompt families that still yield top-1 `"<"`: these aren’t behaving as negatives behaviorally; treat them as leakage examples and inspect why.
