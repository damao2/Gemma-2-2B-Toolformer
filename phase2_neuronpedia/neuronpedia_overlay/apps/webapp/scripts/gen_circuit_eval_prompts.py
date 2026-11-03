"""Generate prompt sets for evaluating a fixed tool-call circuit across many prompts.

This generator is for the Neuronpedia batch-eval workflow (graph generation + scoring).

Key design constraints (from your setup):
- Model has only 2 output modes:
  1) tool-call: emits a literal '<tool_call>calculator(...)...</tool_call>' sequence
  2) negative: strictly echoes the prompt token-by-token
- To get a meaningful next-token target logit, prompts should end with a real trailing newline.

Therefore this script outputs JSONL lines of the form:
  {"label": "...", "prompt": "...\n", "meta": {...}}

In addition to the old `family` label, each prompt now carries multi-axis metadata
for the eval pipeline:
- `intent`: `pos` or `neg`
- `distribution`: `id` or `ood`
- `promptClass`: cleaner semantic bucket for later reporting
- `designRole`: static intended use in screening (`id_capability_probe`, `ood_capability_probe`, `ood_boundary_probe`, `ood_incorrect_stress_probe`)
- `expectedBehavior`: `tool_call` or `echo_first`
- `variantGroup`: groups nearby variants that share the same template/op family

Prompt families:
- pos_seen_*: in-distribution arithmetic prompts (same templates used in training)
- pos_unseen_*: near-boundary seen-like positives (lowercase imperative / punctuation drift / number-shape interaction)
- neg_hard_unseen_*: compact quoted-markup stress prompts
- neg_soft_unseen_*: explain-only layouts on a fragile seen subtraction template plus seen-like incorrect stress prompts
- neg_anti_tool_*: anti-tool layouts on a fragile seen subtraction template plus seen-like incorrect stress prompts

Usage (from neuronpedia/apps/webapp):
  python3 scripts/gen_circuit_eval_prompts.py --out scripts/prompts_circuit_eval_v2.jsonl --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ---- Reuse / adapt templates from gen_toolcall_activations_corpus.py ----

QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "+": [
        "What is {a} + {b}?",
        "Could you calculate {a} plus {b} for me?",
        "What's the total of {a} and {b}?",
        "If you add {a} and {b}, what do you get?",
        "Tell me the result of {a} + {b}.",
        "How much is {a} plus {b}?",
        "Compute the value of {a} + {b}.",
        "If I have {a} items and I get {b} more, I need you to calculate {a} + {b}.",
    ],
    "-": [
        "What is {a} - {b}?",
        "Could you subtract {b} from {a}?",
        "What's the difference between {a} and {b}?",
        "If you take {b} away from {a}, what do you get?",
        "Tell me the result of {a} minus {b}.",
        "Compute the value of {a} - {b}.",
        "I started with a budget of {a} and spent {b}. To see what's left, calculate {a} - {b}.",
    ],
    "*": [
        "What is {a} times {b}?",
        "Could you multiply {a} and {b} for me?",
        "What's the product of {a} and {b}?",
        "If you multiply {a} by {b}, what do you get?",
        "Compute the value of {a} * {b}.",
        "To find the area of the rectangle, I need to multiply its sides. Calculate {a} * {b} for me.",
    ],
    "/": [
        "What is {a} divided by {b}?",
        "Could you divide {a} by {b} for me?",
        "If you divide {a} by {b}, what do you get?",
        "Please compute the result of {a} over {b}.",
        "Compute the value of {a} / {b}.",
        "The bill was split {b} ways. The total was {a}. Please compute {a} / {b} to see what each person owes.",
    ],
}

QUESTION_TEMPLATES_2: List[str] = [
    "Calculate {X} {op} {Y}.",
    "Give me the result of {X} {op} {Y}.",
    "Solve for {X} {op} {Y}.",
    "Run the numbers for {X} {op} {Y}.",
    "Tell me the answer to {X} {op} {Y}.",
    "Do the math for {X} {op} {Y}.",
    "I need to know the answer to {X} {op} {Y}.",
    "I'm trying to figure out {X} {op} {Y}.",
    "I'm stuck on this calculation: {X} {op} {Y}.",
    "Let's find out what {X} {op} {Y} equals.",
    "I'd like to know the result of {X} {op} {Y}.",
    "My homework asks for the value of {X} {op} {Y}.",
]

CONTEXTS: List[str] = [
    "I'm helping my brother learn math.",
    "While budgeting my expenses, I got stuck.",
    "This is a quick math exercise.",
    "Here's a math problem for you.",
    "I need help with this calculation.",
]

HARD_NEGATIVE_TEMPLATES: List[str] = [
    "Do not run this; just explain the string <tool_call>calculator({expr})</tool_call>.",
    "Repeat exactly: <tool_call>calculator({expr})</tool_call>.",
    "Describe this example without executing it: <tool_call>calculator({expr})</tool_call>.",
    "This is quoted text, not an action: <tool_call>calculator({expr})</tool_call>.",
]

HARD_NEGATIVE_LABELS: List[str] = [
    "neg_hard_unseen_dont_run_string",
    "neg_hard_unseen_repeat_exactly",
    "neg_hard_unseen_describe_example",
    "neg_hard_unseen_quoted_not_action",
]

SOFT_NEGATIVE_TEMPLATES: List[str] = [
    "What's the difference between {a} and {b}? Explain the method only and do not compute it.",
    "Explain the method only and do not compute it. What's the difference between {a} and {b}?",
    "Explain the method only and do not compute it.What's the difference between {a} and {b}?",
    "What's the difference between {a} and {b} Explain the method only and do not compute it.",
    "What is {a} + {b}? Explain the method only and do not compute it.",
    "Give me the result of {a} - {b}. Explain the method only and do not compute it.",
    "I'm stuck on this calculation: {a} / {b}. Explain the method only and do not compute it.",
]

SOFT_NEGATIVE_LABELS: List[str] = [
    "neg_soft_unseen_suffix_explain_only",
    "neg_soft_unseen_prefix_explain_space",
    "neg_soft_unseen_prefix_explain_nospace",
    "neg_soft_unseen_no_qmark_suffix_explain",
    "neg_soft_unseen_stress_what_is_suffix_explain",
    "neg_soft_unseen_stress_give_result_suffix_explain",
    "neg_soft_unseen_stress_im_stuck_suffix_explain",
]

# New for eval: arithmetic intent but explicitly forbids tool usage.
ANTI_TOOL_TEMPLATES: List[str] = [
    "What's the difference between {a} and {b}? Do not use any tool.",
    "Do not use any tool. What's the difference between {a} and {b}?",
    "Do not use any tool.What's the difference between {a} and {b}?",
    "What's the difference between {a} and {b} Do not use any tool.",
    "What is {a} + {b}? Do not use any tool.",
    "Give me the result of {a} - {b}. Do not use any tool.",
    "I'm stuck on this calculation: {a} / {b}. Do not use any tool.",
]

ANTI_TOOL_LABELS: List[str] = [
    "neg_anti_tool_suffix_do_not_use_tool",
    "neg_anti_tool_prefix_space",
    "neg_anti_tool_prefix_nospace",
    "neg_anti_tool_no_qmark_suffix",
    "neg_anti_tool_stress_what_is_suffix",
    "neg_anti_tool_stress_give_result_suffix",
    "neg_anti_tool_stress_im_stuck_suffix",
]

# Near-boundary positive OOD prompts stay close to seen templates and focus on
# the perturbations that actually moved the gate in manual probing.
POS_UNSEEN_TEMPLATES: List[str] = [
    "compute the value of {a} * {b}.",
    "compute the value of {a} * {b}",
    "What's the difference between {a} and {b}",
    "give me the result of {a} - {b}.",
]

POS_UNSEEN_LABELS: List[str] = [
    "pos_unseen_compute_lower",
    "pos_unseen_compute_lower_nopunct",
    "pos_unseen_difference_no_qmark",
    "pos_unseen_give_lower",
]

BOUNDARY_MUL_NUMBER_PAIRS: List[Tuple[int, int]] = [
    (112, 309),
    (11, 309),
    (1, 309),
    (2, 309),
    (1, 39),
    (1, 3),
    (4, 9),
]


def _sample_boundary_mul_pair() -> Tuple[int, int]:
    return random.choice(BOUNDARY_MUL_NUMBER_PAIRS)


def _sample_boundary_sub_pair() -> Tuple[int, int]:
    return random.randint(100, 999), random.randint(100, 999)


def _sample_stress_seenlike_pair() -> Tuple[int, int]:
    # Keep stress prompts numerically "ordinary" so they probe directive failure,
    # not the tiny-number boundary effects used elsewhere.
    return random.randint(100, 999), random.randint(100, 999)


def _choose_template_ids(*, n: int, template_count: int, family_name: str) -> List[int]:
    """Choose template ids for a family, ensuring every template appears at least once.

    Your small-scale eval runs can miss templates if we sample uniformly at random.
    For all non-pos_seen families, we therefore enforce template coverage.

    If you request fewer samples than the number of templates, coverage is impossible,
    so we raise a clear error rather than silently changing output size.
    """

    if n <= 0:
        return []
    if template_count <= 0:
        raise ValueError(f"Template list for family '{family_name}' is empty")
    if n < template_count:
        raise ValueError(
            f"--num for family '{family_name}' is {n}, but there are {template_count} templates. "
            "Increase the count to >= number of templates (or set it to 0 to disable the family)."
        )

    ids = list(range(template_count))
    ids += [random.randrange(template_count) for _ in range(n - template_count)]
    random.shuffle(ids)
    return ids


@dataclass(frozen=True)
class ArithSample:
    prompt: str
    op: str
    expr: str
    a: int
    b: int
    with_context: bool
    template_family: str
    template_id: int


def _sample_arith_base(*, with_context: bool, use_seen_templates: bool) -> ArithSample:
    ops = list(QUESTION_TEMPLATES.keys())
    a = random.randint(1, 999)
    b = random.randint(1, 999)
    op = random.choice(ops)
    if op == "/":
        # avoid division by 0 (b is already 1..999)
        pass

    if use_seen_templates:
        if random.random() > 0.5:
            tmpl_list = QUESTION_TEMPLATES[op]
            template_id = random.randrange(len(tmpl_list))
            question = tmpl_list[template_id].format(a=a, b=b)
            template_family = f"seen_opmap:{op}"
        else:
            template_id = random.randrange(len(QUESTION_TEMPLATES_2))
            question = QUESTION_TEMPLATES_2[template_id].format(X=a, Y=b, op=op)
            template_family = "seen_generic"
    else:
        template_id = random.randrange(len(POS_UNSEEN_TEMPLATES))
        question = POS_UNSEEN_TEMPLATES[template_id].format(a=a, b=b, op=op)
        template_family = "unseen_tool_intent"

    context = (random.choice(CONTEXTS) + " ") if with_context else ""
    prompt = (context + question) if random.random() > 0.5 else question

    # Ensure there are no embedded newlines.
    prompt = prompt.replace("\n", " ").strip()

    expr = f"{a}{op}{b}"
    return ArithSample(prompt=prompt, op=op, expr=expr, a=a, b=b, with_context=with_context, template_family=template_family, template_id=template_id)


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else (text + "\n")


def _jsonl_row(*, label: str, prompt: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    # Always end with a real newline so the next token logit is meaningful in your setup.
    prompt = _ensure_trailing_newline(prompt)
    return {"label": label, "prompt": prompt, "meta": meta}


def _intent_for_family(family: str) -> str:
    return "pos" if family.startswith("pos_") else "neg"


def _distribution_for_family(family: str) -> str:
    return "id" if family == "pos_seen" else "ood"


def _variant_group(*, family: str, template_family: str, template_id: int, op: str) -> str:
    return f"{family}|{template_family}|t{template_id}|{_op_name(op)}"


def _meta_axes(
    *,
    family: str,
    prompt_class: str,
    design_role: str,
    expected_mode: str,
    seen_in_training: bool,
    op: str,
    expr: str,
    a: int,
    b: int,
    with_context: bool,
    template_family: str,
    template_id: int,
) -> Dict[str, Any]:
    return {
        "schemaVersion": "gate_eval_v1",
        "family": family,
        "intent": _intent_for_family(family),
        "distribution": _distribution_for_family(family),
        "promptClass": prompt_class,
        "designRole": design_role,
        "expectedMode": expected_mode,
        "expectedBehavior": "tool_call" if expected_mode == "tool_call" else "echo_first",
        "seenInTraining": seen_in_training,
        "op": op,
        "expr": expr,
        "a": a,
        "b": b,
        "withContext": with_context,
        "templateFamily": template_family,
        "templateId": template_id,
        "variantGroup": _variant_group(
            family=family,
            template_family=template_family,
            template_id=template_id,
            op=op,
        ),
    }


def generate_pos_seen(n: int, *, with_context: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    while len(out) < n:
        s = _sample_arith_base(with_context=with_context, use_seen_templates=True)
        label = f"pos_seen_arith_{_op_name(s.op)}"
        out.append(
            _jsonl_row(
                label=label,
                prompt=s.prompt,
                meta=_meta_axes(
                    family="pos_seen",
                    prompt_class="arith_seen",
                    design_role="id_capability_probe",
                    expected_mode="tool_call",
                    seen_in_training=True,
                    op=s.op,
                    expr=s.expr,
                    a=s.a,
                    b=s.b,
                    with_context=s.with_context,
                    template_family=s.template_family,
                    template_id=s.template_id,
                ),
            )
        )
    return out


def generate_pos_unseen(n: int, *, with_context: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    template_ids = _choose_template_ids(n=n, template_count=len(POS_UNSEEN_TEMPLATES), family_name="pos_unseen")
    for template_id in template_ids:
        # These OOD prompts are intentionally boundary-oriented, so we avoid
        # adding extra free-form context and we choose numbers/templates based
        # on the failure modes found in manual probing.
        if template_id in (0, 1):
            op = "*"
            a, b = _sample_boundary_mul_pair()
            template_family = "near_boundary_compute_lower"
        else:
            op = "-"
            a, b = _sample_boundary_sub_pair()
            template_family = "near_boundary_difference" if template_id == 2 else "near_boundary_give_lower"
        question = POS_UNSEEN_TEMPLATES[template_id].format(a=a, b=b, op=op)
        prompt = question.replace("\n", " ").strip()

        expr = f"{a}{op}{b}"
        label = f"{POS_UNSEEN_LABELS[template_id]}_{_op_name(op)}"
        out.append(
            _jsonl_row(
                label=label,
                prompt=prompt,
                meta=_meta_axes(
                    family="pos_unseen",
                    prompt_class="near_boundary_positive",
                    design_role="ood_boundary_probe",
                    expected_mode="tool_call",
                    seen_in_training=False,
                    op=op,
                    expr=expr,
                    a=a,
                    b=b,
                    with_context=False,
                    template_family=template_family,
                    template_id=template_id,
                ),
            )
        )
    return out


def generate_neg_hard_unseen(n: int, *, with_context: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    template_ids = _choose_template_ids(n=n, template_count=len(HARD_NEGATIVE_TEMPLATES), family_name="neg_hard_unseen")
    for template_id in template_ids:
        s = _sample_arith_base(with_context=with_context, use_seen_templates=True)
        text = HARD_NEGATIVE_TEMPLATES[template_id].format(expr=s.expr)
        label = HARD_NEGATIVE_LABELS[template_id]
        out.append(
            _jsonl_row(
                label=label,
                prompt=text,
                meta=_meta_axes(
                    family="neg_hard_unseen",
                    prompt_class="quoted_tool_markup_compact",
                    design_role="ood_capability_probe",
                    expected_mode="echo",
                    seen_in_training=False,
                    op=s.op,
                    expr=s.expr,
                    a=s.a,
                    b=s.b,
                    with_context=s.with_context,
                    template_family="hard_negative",
                    template_id=template_id,
                ),
            )
        )
    return out


def generate_neg_soft_unseen(n: int, *, with_context: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    template_ids = _choose_template_ids(n=n, template_count=len(SOFT_NEGATIVE_TEMPLATES), family_name="neg_soft_unseen")
    for template_id in template_ids:
        if template_id <= 3:
            a, b = _sample_boundary_sub_pair()
            op = "-"
            prompt_class = "near_boundary_explain_only"
            template_family = "near_boundary_explain_only"
            design_role = "ood_boundary_probe"
        elif template_id == 4:
            a, b = _sample_stress_seenlike_pair()
            op = "+"
            prompt_class = "stress_explain_only_seenlike"
            template_family = "stress_explain_only_what_is"
            design_role = "ood_incorrect_stress_probe"
        elif template_id == 5:
            a, b = _sample_stress_seenlike_pair()
            op = "-"
            prompt_class = "stress_explain_only_seenlike"
            template_family = "stress_explain_only_give_result"
            design_role = "ood_incorrect_stress_probe"
        else:
            a, b = _sample_stress_seenlike_pair()
            op = "/"
            prompt_class = "stress_explain_only_seenlike"
            template_family = "stress_explain_only_im_stuck"
            design_role = "ood_incorrect_stress_probe"
        expr = f"{a}{op}{b}"
        text = SOFT_NEGATIVE_TEMPLATES[template_id].format(a=a, b=b)
        label = SOFT_NEGATIVE_LABELS[template_id]
        out.append(
            _jsonl_row(
                label=label,
                prompt=text,
                meta=_meta_axes(
                    family="neg_soft_unseen",
                    prompt_class=prompt_class,
                    design_role=design_role,
                    expected_mode="echo",
                    seen_in_training=False,
                    op=op,
                    expr=expr,
                    a=a,
                    b=b,
                    with_context=False,
                    template_family=template_family,
                    template_id=template_id,
                ),
            )
        )
    return out


def generate_neg_anti_tool(n: int, *, with_context: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    template_ids = _choose_template_ids(n=n, template_count=len(ANTI_TOOL_TEMPLATES), family_name="neg_anti_tool")
    for template_id in template_ids:
        if template_id <= 3:
            a, b = _sample_boundary_sub_pair()
            op = "-"
            prompt_class = "near_boundary_anti_tool"
            template_family = "near_boundary_anti_tool"
            design_role = "ood_boundary_probe"
        elif template_id == 4:
            a, b = _sample_stress_seenlike_pair()
            op = "+"
            prompt_class = "stress_anti_tool_seenlike"
            template_family = "stress_anti_tool_what_is"
            design_role = "ood_incorrect_stress_probe"
        elif template_id == 5:
            a, b = _sample_stress_seenlike_pair()
            op = "-"
            prompt_class = "stress_anti_tool_seenlike"
            template_family = "stress_anti_tool_give_result"
            design_role = "ood_incorrect_stress_probe"
        else:
            a, b = _sample_stress_seenlike_pair()
            op = "/"
            prompt_class = "stress_anti_tool_seenlike"
            template_family = "stress_anti_tool_im_stuck"
            design_role = "ood_incorrect_stress_probe"
        expr = f"{a}{op}{b}"
        text = ANTI_TOOL_TEMPLATES[template_id].format(a=a, b=b)
        label = f"{ANTI_TOOL_LABELS[template_id]}_{_op_name(op)}"
        out.append(
            _jsonl_row(
                label=label,
                prompt=text,
                meta=_meta_axes(
                    family="neg_anti_tool",
                    prompt_class=prompt_class,
                    design_role=design_role,
                    expected_mode="echo",
                    seen_in_training=False,
                    op=op,
                    expr=expr,
                    a=a,
                    b=b,
                    with_context=False,
                    template_family=template_family,
                    template_id=template_id,
                ),
            )
        )
    return out


def _op_name(op: str) -> str:
    return {"+": "add", "-": "sub", "*": "mul", "/": "div"}.get(op, "other")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--withContext", action="store_true", help="Include natural-language context prefix sometimes")

    ap.add_argument("--numPosSeen", type=int, default=10)
    ap.add_argument("--numPosUnseen", type=int, default=10)
    ap.add_argument("--numNegHardUnseen", type=int, default=10)
    ap.add_argument("--numNegSoftUnseen", type=int, default=10)
    ap.add_argument("--numNegAntiTool", type=int, default=10)

    ap.add_argument("--shuffle", action="store_true", help="Shuffle before writing")

    args = ap.parse_args()

    random.seed(args.seed)

    rows: List[Dict[str, Any]] = []
    rows += generate_pos_seen(args.numPosSeen, with_context=args.withContext)
    if args.numPosUnseen > 0:
        rows += generate_pos_unseen(args.numPosUnseen, with_context=args.withContext)
    rows += generate_neg_hard_unseen(args.numNegHardUnseen, with_context=args.withContext)
    rows += generate_neg_soft_unseen(args.numNegSoftUnseen, with_context=args.withContext)
    rows += generate_neg_anti_tool(args.numNegAntiTool, with_context=args.withContext)

    if args.shuffle:
        random.shuffle(rows)

    write_jsonl(args.out, rows)

    # Small summary for quick sanity.
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    print("Wrote", len(rows), "rows to", args.out)
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()

'''
python3 scripts/gen_circuit_eval_prompts.py --out scripts/prompts_circuit_eval_v3.jsonl --withContext --numPosSeen 3 --numPosUnseen 3 --numNegHardUnseen 5 --numNegSoftUnseen 5 --numNegAntiTool 2 --shuffle
'''