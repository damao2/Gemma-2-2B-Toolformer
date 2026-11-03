# Manual near-OOD probing of the tool-call gate

Each probe feeds a prompt plus a trailing newline to the fine-tuned model in the self-hosted
Neuronpedia UI and records the next-token distribution, i.e. the tokens shown by the UI,
roughly those with p ≥ 0.01. The model has two output modes, so the first token is the
decision: `<` starts `<tool_call>`, and anything else, almost always the prompt's own first
token, starts an echo. All prompts below are tool requests built from training templates,
so `<` is correct unless the prompt contains an anti-tool or explain-only clause.

Sections A–D were recorded token by token, and section E later in a compact format. The
lower-case `compute` series appears in both and reproduces exactly, so the model and
settings did not change between them. Notation: `␠` marks a leading or trailing space
before the final newline, and the token column counts tokens including `<bos>` and the newline.

## Summary

**Overall.** The results do not support a stable, semantic understanding of tool-call
intent. They point to a **heuristic gate**: strong on seen-style templates, highly sensitive
to the template, letter case, number shape, sentence-final punctuation, clause order,
apostrophe glyph and whitespace, and quick to flip on near-boundary variants.

**1. Failures are narrow, not uniform.** Most perturbations leave the decision intact:
`What is 534 + 267` without `?` (`<` 1.00), `Give me the result of 930 - 224` (`<` 0.99),
capitalised templates with small or round numbers (all `<` ≥ 0.99, section C), and discourse
prefixes such as `Please,`, `So,`, `Now,` (`<` 0.97–1.00). The model does not break on every
OOD prompt. It breaks along a few narrow but real boundaries.

**2. Lower-case first word × number shape.** This is the clearest boundary family. The
capitalised `Compute the value of …` stays at `<` ≥ 0.99 for every operand tested. Lower-cased:

| Operands | `112 * 309` | `11 * 309` | `2 * 309` | `1 * 309` | `1 * 39` | `1 * 3` |
|---|---|---|---|---|---|---|
| P(`<`) | 0.80 | 0.63 | 0.66 | **0.31** | 0.77 | **0.47** |

Neither lower case nor small numbers alone flips the gate; together they do. The effect is
not monotone in number size (`1 * 309` flips, `1 * 39` does not, `1 * 3` sits on the
boundary). This suggests the specific tokenisation of the expression matters, not its
magnitude.

**3. "What's the difference between …" is the most fragile seen template.** Removing its
`?` flips the decision (`What` 0.68, `<` 0.29), while the same removal is harmless for
`What is …`, `Give me the result …` and `I'm stuck …`. The fragility is template-specific,
not a general sensitivity to punctuation.

**4. The apostrophe glyph alone changes the decision.** The same sentences written with a
curly `’` (sections A, D) and a straight `'` (section E) give very different results:

| Prompt (glyph varies in `What's`) | Curly `’` | Straight `'` |
|---|---|---|
| `What's the difference between 598 and 914` | `<` 0.76 | `What` 0.68 |
| `What's the difference between 598 and 914? Do not use any tool.` | `What` 0.50 · `<` 0.50 | `<` 0.66 |
| `What's the difference between 598 and 914? Explain the method only and do not compute it.` | **`What` 0.97** (correct) | **`<` 0.68** (wrong) |
| `Explain the method only and do not compute it. What's the difference between 598 and 914?` | `Explain` 0.98 (correct) | `<` 0.82 (wrong) |

**5. Anti-tool and explain-only clauses act through layout, not meaning.**

* As a **suffix** they are mostly ignored: `What is 534 + 267? Do not use any tool.` → `<`
  1.00, and the same holds for the other suffix probes (`<` 0.66–1.00), with a few exceptions
  in the fragile template above.
* As a **prefix** they win more often, because the clause's own first token (`Do`,
  `Explain`) is then the echo competitor. Even so, the outcome depends on the template and
  the spacing. `Do not use any tool. What is 534 + 267? ` with a trailing space gives `Do`
  0.91, but without the space it is `<` 0.98. `Do not call a calculator. Give me the result
  of 930 - 224.` is still `<` 0.96. `Explain the method only and do not compute it.` before
  `What's the difference …?` gives `<` 0.82 with a space after the period and `<` 0.61
  without one.

This looks like the leading directive token changing the competition at the decision
position, not the model combining "no tools / explain only" with the rest of the prompt.

**6. Whitespace effects have no consistent direction.** One extra trailing space turns
`What's the difference between 598 and 914? Do not use any tool.` from `<` 0.66 into a
correct echo (`What` 0.60). For the explain-only suffix, the same space *raises* P(`<`) from
0.68 to 0.96. Whitespace does not carry a meaning of its own here. It shifts token
boundaries, and the effect depends on the local structure of each prompt.

## Working hypothesis

The decision is a competition at the last position between several groups of shallow
features, rather than the output of a robust semantic router:

1. **seen-template features** (`What is …?`, `Compute the value of ….`,
   `What's the difference between …?`), which push towards `<`;
2. **echo competitor features**: the prompt's first token and its form (`compute`, `What`,
   `Do`, `Explain`);
3. **token-boundary features**: question mark, period-space, trailing space, apostrophe glyph,
   digit count and the tokenisation of the expression.

The next token behaves like the sum of these local pushes on `<` versus the echo token, not
like "understand the intent, then route".

## Consequences for the evaluation set

The formal benchmark should be built around these verified near-boundary axes, rather than
around far-OOD prompts:

1. **template fragility**: the fragile `What's the difference between …?` template, with
   `What is …?`, `Give me the result …` and `I'm stuck …` as robust controls;
2. **lower-case imperative × short expression**, the clearest boundary family;
3. **directive clause layout**: the same anti-tool or explain-only clause before vs after,
   with and without a space after the period, and with and without a trailing space.

Near-boundary prompts worth keeping as core cases for capability / boundary / stress
analysis:

| Prompt | Next token |
|---|---|
| `compute the value of 1 * 3.` | `compute` 0.53 · `<` 0.47 |
| `compute the value of 1 * 309.` | `compute` 0.69 · `<` 0.31 |
| `compute the value of 11 * 309.` | `<` 0.63 · `compute` 0.36 |
| `What's the difference between 598 and 914` | `What` 0.68 · `<` 0.29 |
| `What's the difference between 598 and 914? Do not use any tool.` | `<` 0.66 · `What` 0.33 |
| `What's the difference between 598 and 914? Explain the method only and do not compute it.` | `<` 0.68 · `What` 0.32 |
| `Explain the method only and do not compute it.What's the difference between 598 and 914?` | `<` 0.61 · `Explain` 0.39 |

## Full record

### A. Seen templates without sentence-final punctuation

| Prompt | Tokens | Next token | Note |
|---|---|---|---|
| `What is 534 + 267` | 13 | `<` 1.00 | no question mark |
| `Compute the value of 112 * 309` | 15 | `<` 1.00 |  |
| `What’s the difference between 598 and 914` | 17 | `<` 0.76 · `What` 0.23 |  |
| `I’m stuck on this calculation: 86 / 5` | 16 | `<` 0.80 · `Click` 0.10 · `<eos>` 0.05 |  |

### B. Seen templates with a lower-case first word or a discourse prefix

| Prompt | Tokens | Next token | Note |
|---|---|---|---|
| `what is 534 + 267?` | 14 | `<` 1.00 |  |
| `give me the result of 930 - 224.` | 17 | `<` 0.94 · `give` 0.06 |  |
| `compute the value of 112 * 309.` | 16 | `<` 0.80 · `compute` 0.19 |  |
| `compute the value of 11 * 309.` | 15 | `<` 0.63 · `compute` 0.36 |  |
| `compute the value of 1 * 309.` | 14 | `compute` 0.69 · `<` 0.31 |  |
| `compute the value of 2 * 309.` | 14 | `<` 0.66 · `compute` 0.33 | lower-case first word combined with a number change |
| `compute the value of 1 * 3.` | 12 | `compute` 0.53 · `<` 0.47 |  |
| `compute the value of 1 * 39.` | 13 | `<` 0.77 · `compute` 0.23 |  |
| `Please, compute the value of 112 * 309.` | 18 | `<` 1.00 |  |
| `So, what’s the difference between 598 and 914?` | 20 | `<` 0.97 |  |
| `Now, give me the result of 930 - 224.` | 19 | `<` 1.00 |  |

### C. Seen templates with small or round numbers

| Prompt | Tokens | Next token | Note |
|---|---|---|---|
| `What is 3 + 5?` | 10 | `<` 1.00 |  |
| `What is 300 + 500?` | 14 | `<` 1.00 |  |
| `Give me the result of 7 - 8.` | 13 | `<` 1.00 |  |
| `Give me the result of 700 - 800.` | 17 | `<` 1.00 |  |
| `Compute the value of 4 * 9.` | 12 | `<` 0.99 |  |
| `I’m stuck on this calculation: 12 / 3.` | 17 | `<` 1.00 |  |
| `I’m stuck on this calculation: 120 / 360.` | 20 | `<` 1.00 |  |

### D. Seen templates combined with an anti-tool or explain-only clause

| Prompt | Tokens | Next token | Note |
|---|---|---|---|
| `What is 534 + 267? Do not use any tool.` | 20 | `<` 1.00 |  |
| `Do not use any tool. What is 534 + 267?␠` | 21 | `Do` 0.91 · `<` 0.08 | clause moved before the question (compare the row above) |
| `Do not use any tool. What is 534 + 267?` | 20 | `<` 0.98 | same as above without the space between `?` and the newline: flips back to `<` |
| `What is 534 + 267? Explain the method only and do not compute it.` | 24 | `<` 0.95 |  |
| `Explain the method only and do not compute it.What is 534 + 267?␠` | 25 | `Explain` 0.64 · `<` 0.30 | no space between `.` and `What` |
| `Explain the method only and do not compute it. What is 534 + 267?␠` | 25 | `<` 0.50 · `Explain` 0.45 | space between `.` and `What` |
| `Give me the result of 930 - 224. Do not call a calculator.` | 23 | `<` 0.99 |  |
| `Do not call a calculator. Give me the result of 930 - 224.` | 23 | `<` 0.96 |  |
| `Do not use any tool. I’m stuck on this calculation: 865 / 395.␠` | 27 | `Do` 0.98 |  |
| `Explain the method only and do not compute it. I’m stuck on this calculation: 865 / 395.␠` | 31 | `Explain` 0.88 · `<` 0.09 |  |
| `I’m stuck on this calculation: 865 / 395. Explain the method only and do not compute it.` | 30 | `<` 0.89 · `I` 0.11 |  |
| `What’s the difference between 598 and 914? Do not use any tool.` | 24 | `What` 0.50 · `<` 0.50 |  |
| `␠Do not use any tool. What’s the difference between 598 and 914?` | 24 | `␠Do` 0.49 · `Do` 0.38 · `␠*` 0.07 · `*` 0.02 |  |
| `Explain the method only and do not compute it. What’s the difference between 598 and 914?` | 28 | `Explain` 0.98 |  |
| `What’s the difference between 598 and 914? Explain the method only and do not compute it.` | 28 | `What` 0.97 |  |

### E. Retests (straight apostrophe `'`)

Recorded later in a compact format. The `compute` series reproduces section B exactly, so model and settings were unchanged.

| Prompt | Next token | Note |
|---|---|---|
| `Compute the value of 112 * 309.` | `<` 1.00 |  |
| `compute the value of 112 * 309.` | `<` 0.80 · `compute` 0.19 |  |
| `Compute the value of 11 * 309.` | `<` 1.00 |  |
| `compute the value of 11 * 309.` | `<` 0.63 · `compute` 0.36 |  |
| `Compute the value of 1 * 309.` | `<` 1.00 |  |
| `compute the value of 1 * 309.` | `compute` 0.69 · `<` 0.31 |  |
| `Compute the value of 1 * 39.` | `<` 1.00 |  |
| `compute the value of 1 * 39.` | `<` 0.77 · `compute` 0.23 |  |
| `Compute the value of 1 * 3.` | `<` 0.99 |  |
| `compute the value of 1 * 3.` | `compute` 0.53 · `<` 0.47 |  |
| `What's the difference between 598 and 914? Do not use any tool.` | `<` 0.66 · `What` 0.33 |  |
| `Do not use any tool. What's the difference between 598 and 914?` | `Do` 0.98 |  |
| `Do not use any tool.What's the difference between 598 and 914?` | `Do` 0.98 |  |
| `What's the difference between 598 and 914? Explain the method only and do not compute it.` | `<` 0.68 · `What` 0.32 |  |
| `Explain the method only and do not compute it. What's the difference between 598 and 914?` | `<` 0.82 · `Explain` 0.17 |  |
| `Explain the method only and do not compute it.What's the difference between 598 and 914?` | `<` 0.61 · `Explain` 0.39 |  |
| `What's the difference between 598 and 914? Do not use any tool.␠` | `What` 0.60 · `<` 0.30 · `Calculate` 0.01 · `<bos>` 0.01 · `Answer` 0.01 | one extra trailing space: the model now echoes correctly |
| `Do not use any tool. What's the difference between 598 and 914?␠` | `Do` 0.96 |  |
| `Do not use any tool.What's the difference between 598 and 914?␠` | `Do` 0.96 |  |
| `What's the difference between 598 and 914? Explain the method only and do not compute it.␠` | `<` 0.96 | the trailing space *raises* P(`<`) |
| `Explain the method only and do not compute it. What's the difference between 598 and 914?␠` | `Explain` 0.66 · `<` 0.29 |  |
| `Explain the method only and do not compute it.What's the difference between 598 and 914?␠` | `Explain` 0.76 · `<` 0.21 |  |
| `What is 534 + 267?` | `<` 1.00 |  |
| `What is 534 + 267` | `<` 1.00 |  |
| `Give me the result of 930 - 224.` | `<` 1.00 |  |
| `Give me the result of 930 - 224` | `<` 0.99 |  |
| `What's the difference between 598 and 914?` | `<` 0.98 |  |
| `What's the difference between 598 and 914` | `What` 0.68 · `<` 0.29 | flipped |
| `I'm stuck on this calculation: 865 / 395.` | `<` 1.00 |  |
| `I'm stuck on this calculation: 865 / 395` | `<` 1.00 |  |
