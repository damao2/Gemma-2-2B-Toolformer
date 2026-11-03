"""Generate near-boundary OOD prompt candidates for manual screening.

This script is intentionally different from gen_circuit_eval_prompts.py:
- It starts from seen-style positive anchors.
- It applies *small* perturbations that are likely to move the gate margin
  without changing the prompt too much.
- It emits variant-group metadata so you can compare anchor vs variants after
  running screen_circuit_eval_prompts.ts.

The goal is not to define the final benchmark yet. The goal is to cheaply build
an OOD candidate bank and then use forward-pass screening to find prompts whose
probability margin is close to the decision boundary.

Usage (from neuronpedia/apps/webapp):
  python3 scripts/gen_circuit_eval_boundary_candidates.py \
    --out scripts/prompts_boundary_candidates_v1.jsonl \
    --seed 0 \
    --numAnchors 24
"""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import Any, Dict, List

from gen_circuit_eval_prompts import _ensure_trailing_newline, _op_name, _sample_arith_base


DISCOURSE_PREFIXES = [
    "Please, ",
    "So, ",
]

SMALL_NUMBER_PAIRS = [
    (3, 5),
    (7, 8),
    (4, 9),
]

ROUND_NUMBER_PAIRS = [
    (100, 200),
    (120, 360),
    (250, 400),
]


def _replace_first(text: str, old: str, new: str) -> str:
    idx = text.find(old)
    if idx == -1:
        return text
    return text[:idx] + new + text[idx + len(old) :]


def _replace_numbers(prompt: str, old_a: int, old_b: int, new_a: int, new_b: int) -> str:
    out = _replace_first(prompt, str(old_a), str(new_a))
    out = _replace_first(out, str(old_b), str(new_b))
    return out


def _lowercase_first_alpha(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.lower()
            break
    return "".join(chars)


def _strip_terminal_punctuation(text: str) -> str:
    return re.sub(r"[?.!]$", "", text.strip())


def _compact_operator_spacing(text: str) -> str:
    return re.sub(r"\s*([+\-*/])\s*", r"\1", text)


def _append_clause(prompt: str, clause: str) -> str:
    stripped = prompt.strip()
    return f"{stripped} {clause}"


def _row(*, label: str, prompt: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": label,
        "prompt": _ensure_trailing_newline(prompt),
        "meta": meta,
    }


def _base_meta(
    *,
    family: str,
    distribution: str,
    intent: str,
    expected_behavior: str,
    prompt_class: str,
    perturbation_type: str,
    variant_group: str,
    anchor_prompt: str,
    op: str,
    expr: str,
    a: int,
    b: int,
    anchor_expr: str,
    anchor_a: int,
    anchor_b: int,
    template_family: str,
    template_id: int,
) -> Dict[str, Any]:
    return {
        "schemaVersion": "gate_eval_v1",
        "family": family,
        "intent": intent,
        "distribution": distribution,
        "promptClass": prompt_class,
        "expectedMode": "tool_call" if expected_behavior == "tool_call" else "echo",
        "expectedBehavior": expected_behavior,
        "variantGroup": variant_group,
        "perturbationType": perturbation_type,
        "anchorPrompt": anchor_prompt,
        "anchorExpr": anchor_expr,
        "anchorA": anchor_a,
        "anchorB": anchor_b,
        "op": op,
        "expr": expr,
        "a": a,
        "b": b,
        "templateFamily": template_family,
        "templateId": template_id,
    }


def generate_boundary_candidates(num_anchors: int, *, with_context: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for anchor_idx in range(num_anchors):
        sample = _sample_arith_base(with_context=with_context, use_seen_templates=True)
        anchor_label = f"anchor_pos_seen_{anchor_idx:03d}_{_op_name(sample.op)}"
        variant_group = f"boundary_anchor|{anchor_idx:03d}|{sample.template_family}|{_op_name(sample.op)}"

        rows.append(
            _row(
                label=anchor_label,
                prompt=sample.prompt,
                meta=_base_meta(
                    family="pos_seen_anchor",
                    distribution="id",
                    intent="pos",
                    expected_behavior="tool_call",
                    prompt_class="anchor_base",
                    perturbation_type="base_anchor",
                    variant_group=variant_group,
                    anchor_prompt=sample.prompt,
                    op=sample.op,
                    expr=sample.expr,
                    a=sample.a,
                    b=sample.b,
                    anchor_expr=sample.expr,
                    anchor_a=sample.a,
                    anchor_b=sample.b,
                    template_family=sample.template_family,
                    template_id=sample.template_id,
                ),
            )
        )

        # Positive near-OOD: tiny surface perturbations.
        pos_variants = [
            (
                "pos_near_case_lower_first",
                _lowercase_first_alpha(sample.prompt),
                "surface_case_lower_first",
            ),
            (
                "pos_near_prefix_please",
                random.choice(DISCOURSE_PREFIXES) + _lowercase_first_alpha(sample.prompt),
                "surface_discourse_prefix",
            ),
            (
                "pos_near_no_terminal_punct",
                _strip_terminal_punctuation(sample.prompt),
                "surface_punctuation_drop",
            ),
            (
                "pos_near_compact_operator_spacing",
                _compact_operator_spacing(sample.prompt),
                "surface_operator_spacing",
            ),
        ]

        small_a, small_b = random.choice(SMALL_NUMBER_PAIRS)
        round_a, round_b = random.choice(ROUND_NUMBER_PAIRS)
        pos_variants.extend(
            [
                (
                    "pos_near_small_numbers",
                    _replace_numbers(sample.prompt, sample.a, sample.b, small_a, small_b),
                    "numeric_small_numbers",
                ),
                (
                    "pos_near_round_numbers",
                    _replace_numbers(sample.prompt, sample.a, sample.b, round_a, round_b),
                    "numeric_round_numbers",
                ),
            ]
        )

        for label_stem, prompt, perturbation_type in pos_variants:
            rows.append(
                _row(
                    label=f"{label_stem}_{anchor_idx:03d}_{_op_name(sample.op)}",
                    prompt=prompt,
                    meta=_base_meta(
                        family="pos_near_ood",
                        distribution="ood",
                        intent="pos",
                        expected_behavior="tool_call",
                        prompt_class="near_boundary_positive",
                        perturbation_type=perturbation_type,
                        variant_group=variant_group,
                        anchor_prompt=sample.prompt,
                        op=sample.op,
                        expr=_extract_expr(prompt, sample.op, fallback=sample.expr),
                        a=_extract_first_int(prompt, fallback=sample.a),
                        b=_extract_second_int(prompt, fallback=sample.b),
                        anchor_expr=sample.expr,
                        anchor_a=sample.a,
                        anchor_b=sample.b,
                        template_family=sample.template_family,
                        template_id=sample.template_id,
                    ),
                )
            )

        # Negative near-OOD: minimal intent flips from the same positive anchor.
        neg_variants = [
            (
                "neg_near_append_no_tool",
                _append_clause(sample.prompt, "Do not use any tool."),
                "anti_tool_append",
            ),
            (
                "neg_near_prefix_without_tools",
                f"Without using tools, {_lowercase_first_alpha(sample.prompt)}",
                "anti_tool_prefix",
            ),
            (
                "neg_near_explain_only",
                _append_clause(sample.prompt, "Explain the method only and do not compute it."),
                "explain_only_append",
            ),
        ]

        for label_stem, prompt, perturbation_type in neg_variants:
            rows.append(
                _row(
                    label=f"{label_stem}_{anchor_idx:03d}_{_op_name(sample.op)}",
                    prompt=prompt,
                    meta=_base_meta(
                        family="neg_near_ood",
                        distribution="ood",
                        intent="neg",
                        expected_behavior="echo_first",
                        prompt_class="near_boundary_negative",
                        perturbation_type=perturbation_type,
                        variant_group=variant_group,
                        anchor_prompt=sample.prompt,
                        op=sample.op,
                        expr=sample.expr,
                        a=sample.a,
                        b=sample.b,
                        anchor_expr=sample.expr,
                        anchor_a=sample.a,
                        anchor_b=sample.b,
                        template_family=sample.template_family,
                        template_id=sample.template_id,
                    ),
                )
            )

    return rows


def _extract_all_ints(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def _extract_first_int(text: str, *, fallback: int) -> int:
    ints = _extract_all_ints(text)
    return ints[0] if len(ints) >= 1 else fallback


def _extract_second_int(text: str, *, fallback: int) -> int:
    ints = _extract_all_ints(text)
    return ints[1] if len(ints) >= 2 else fallback


def _extract_expr(text: str, op: str, *, fallback: str) -> str:
    ints = _extract_all_ints(text)
    if len(ints) >= 2:
        return f"{ints[0]}{op}{ints[1]}"
    return fallback


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--numAnchors", type=int, default=24)
    ap.add_argument("--withContext", action="store_true", help="Allow natural-language context in anchors")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    rows = generate_boundary_candidates(args.numAnchors, with_context=args.withContext)
    if args.shuffle:
        random.shuffle(rows)

    write_jsonl(args.out, rows)

    print("Wrote", len(rows), "rows to", args.out)
    print("Each anchor yields:")
    print("  1 base anchor")
    print("  6 positive near-OOD variants")
    print("  3 negative near-OOD variants")


if __name__ == "__main__":
    main()
