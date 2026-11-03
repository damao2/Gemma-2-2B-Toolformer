"""Analyze batch circuit-eval outputs (JSON) and summarize metrics by label/family.

This is a lightweight, dependency-free aggregator for the Neuronpedia batch-eval
workflow.

Input:
  scripts/out/circuit_eval_*.json

Output (default next to input):
  - <stem>.summary_by_label.csv
  - <stem>.summary_by_family.csv
  - <stem>.summary_overall.json

Notes:
- Current batch-eval output includes `targetToken` and `targetProb`. In some runs
  `targetToken` may vary across prompts (e.g. "<" vs "Please"), so a simple
  tool-call "hit" proxy is:

    isAngleBracket = (targetToken == "<")

  and/or thresholds on `targetProb`.

- Replacement/Completeness values are still useful to compare circuit influence,
  but interpret them carefully when `targetToken != "<"`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_mean(xs: Sequence[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.mean(xs2) if xs2 else None


def _safe_median(xs: Sequence[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.median(xs2) if xs2 else None


def _safe_pctl(xs: Sequence[float], p: float) -> Optional[float]:
    xs2 = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs2:
        return None
    if p <= 0:
        return xs2[0]
    if p >= 1:
        return xs2[-1]
    i = (len(xs2) - 1) * p
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return xs2[lo]
    w = i - lo
    return xs2[lo] * (1 - w) + xs2[hi] * w


def _label_family(label: str) -> str:
    # Conservative grouping that matches your generator label patterns.
    if label.startswith("pos_seen_"):
        return "pos_seen"
    if label.startswith("pos_unseen_"):
        return "pos_unseen"
    if label.startswith("neg_hard_unseen_"):
        return "neg_hard_unseen"
    if label.startswith("neg_soft_unseen_"):
        return "neg_soft_unseen"
    if label.startswith("neg_anti_tool_"):
        return "neg_anti_tool"
    return "other"


def _expected_mode_for_family(family: str) -> Optional[str]:
        """Return the expected behavior mode for a prompt family.

        In this project setup:
            - POS families are expected to enter tool-call mode (next token "<").
            - NEG families are expected to stay in echo mode (next token not "<").

        Returns:
            - "tool_call" | "echo" | None (unknown)
        """

        if family in ("pos_seen", "pos_unseen"):
                return "tool_call"
        if family.startswith("neg_"):
                return "echo"
        return None


@dataclass(frozen=True)
class Row:
    label: str
    family: str
    target_token: str
    target_prob: float
    prob_angle_bracket: Optional[float]
    replacement_full: float
    completeness_full: float
    replacement_subgraph: float
    completeness_subgraph: float
    replacement_full_gate: Optional[float]
    completeness_full_gate: Optional[float]
    replacement_subgraph_gate: Optional[float]
    completeness_subgraph_gate: Optional[float]
    coverage: float
    matched: int
    template_count: int

    @property
    def is_angle_bracket(self) -> bool:
        return self.target_token == "<"

    @property
    def bracket_prob(self) -> float:
        # Proxy with older batch-eval output.
        return self.target_prob if self.is_angle_bracket else 0.0

    @property
    def prob_angle_bracket_best_effort(self) -> Optional[float]:
        # Prefer strict gate output if present.
        if self.prob_angle_bracket is not None and math.isfinite(self.prob_angle_bracket):
            return float(self.prob_angle_bracket)
        # Fall back to proxy.
        return float(self.bracket_prob)

    @property
    def replacement_subgraph_gate_best_effort(self) -> Optional[float]:
        return self.replacement_subgraph_gate

    @property
    def completeness_subgraph_gate_best_effort(self) -> Optional[float]:
        return self.completeness_subgraph_gate

    @property
    def matched_frac(self) -> Optional[float]:
        if self.template_count <= 0:
            return None
        return self.matched / self.template_count


def _load_rows(json_path: str) -> Tuple[Dict[str, Any], List[Row]]:
    with open(json_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    meta = blob.get("meta", {})
    results = blob.get("results", [])
    rows: List[Row] = []

    for r in results:
        label = str(r.get("label", ""))
        rows.append(
            Row(
                label=label,
                family=_label_family(label),
                target_token=str(r.get("targetToken", "")),
                target_prob=float(r.get("targetProb", float("nan"))),
                prob_angle_bracket=(None if r.get("probAngleBracket", None) is None else float(r.get("probAngleBracket"))),
                replacement_full=float(r.get("replacement_full", float("nan"))),
                completeness_full=float(r.get("completeness_full", float("nan"))),
                replacement_subgraph=float(r.get("replacement_subgraph", float("nan"))),
                completeness_subgraph=float(r.get("completeness_subgraph", float("nan"))),
                replacement_full_gate=(None if r.get("replacement_full_gate", None) is None else float(r.get("replacement_full_gate"))),
                completeness_full_gate=(None if r.get("completeness_full_gate", None) is None else float(r.get("completeness_full_gate"))),
                replacement_subgraph_gate=(None if r.get("replacement_subgraph_gate", None) is None else float(r.get("replacement_subgraph_gate"))),
                completeness_subgraph_gate=(None if r.get("completeness_subgraph_gate", None) is None else float(r.get("completeness_subgraph_gate"))),
                coverage=float(r.get("coverage", float("nan"))),
                matched=int(r.get("matched", 0)),
                template_count=int(r.get("templateCount", 0)),
            )
        )

    return meta, rows


def _summarize_group(rows: Sequence[Row]) -> Dict[str, Any]:
    target_probs = [x.target_prob for x in rows]
    bracket_probs = [x.bracket_prob for x in rows]
    prob_angles = [x.prob_angle_bracket_best_effort for x in rows]
    is_brackets = [1.0 if x.is_angle_bracket else 0.0 for x in rows]

    # Behavior metrics: does the model follow the intended POS/NEG mode?
    expected_modes = [_expected_mode_for_family(x.family) for x in rows]
    mode_correct_flags: List[float] = []
    tool_call_expected_flags: List[float] = []
    echo_expected_flags: List[float] = []
    for x, exp in zip(rows, expected_modes):
        if exp is None:
            continue
        pred_tool_call = x.is_angle_bracket
        exp_tool_call = exp == "tool_call"
        mode_correct_flags.append(1.0 if pred_tool_call == exp_tool_call else 0.0)
        if exp_tool_call:
            tool_call_expected_flags.append(1.0 if pred_tool_call else 0.0)
        else:
            echo_expected_flags.append(1.0 if pred_tool_call else 0.0)

    repl_s = [x.replacement_subgraph for x in rows]
    comp_s = [x.completeness_subgraph for x in rows]
    repl_gate_all = [x.replacement_subgraph_gate_best_effort for x in rows]
    comp_gate_all = [x.completeness_subgraph_gate_best_effort for x in rows]
    repl_s_gate = [v for v in repl_gate_all if v is not None]
    comp_s_gate = [v for v in comp_gate_all if v is not None]
    cov = [x.coverage for x in rows]
    matched_frac = [x.matched_frac for x in rows if x.matched_frac is not None]

    # Also summarize conditional on angle bracket to reduce mixing.
    bracket_rows = [x for x in rows if x.is_angle_bracket]
    repl_s_br = [x.replacement_subgraph for x in bracket_rows]
    comp_s_br = [x.completeness_subgraph for x in bracket_rows]
    cov_br = [x.coverage for x in bracket_rows]

    bp_p25 = _safe_pctl(bracket_probs, 0.25)
    bp_p75 = _safe_pctl(bracket_probs, 0.75)
    bp_iqr = (bp_p75 - bp_p25) if (bp_p25 is not None and bp_p75 is not None) else None

    cov_p25 = _safe_pctl(cov, 0.25)
    cov_p75 = _safe_pctl(cov, 0.75)
    cov_iqr = (cov_p75 - cov_p25) if (cov_p25 is not None and cov_p75 is not None) else None

    return {
        "n": len(rows),
        # Proxy for "entered tool-call mode" under current batch-eval output.
        "angle_bracket_rate": _safe_mean(is_brackets),
        # Model behavior correctness (independent of circuit quality).
        "expected_tool_call_n": sum(1 for m in expected_modes if m == "tool_call"),
        "expected_echo_n": sum(1 for m in expected_modes if m == "echo"),
        "mode_accuracy": _safe_mean(mode_correct_flags),
        # For POS families this is the tool-call hit rate; for NEG families this is not used.
        "tool_call_expected_hit_rate": _safe_mean(tool_call_expected_flags),
        # For NEG families this is the false-positive tool-call rate (leakage).
        "echo_expected_tool_call_fp_rate": _safe_mean(echo_expected_flags),
        "targetProb_mean": _safe_mean(target_probs),
        "targetProb_median": _safe_median(target_probs),
        # Strict gate metric (best-effort): P('<') if available, else proxy based on top-1 token.
        "probAngleBracket_mean": _safe_mean([p for p in prob_angles if p is not None]),
        "probAngleBracket_p10": _safe_pctl([p for p in prob_angles if p is not None], 0.10),
        "probAngleBracket_p50": _safe_pctl([p for p in prob_angles if p is not None], 0.50),
        "probAngleBracket_p90": _safe_pctl([p for p in prob_angles if p is not None], 0.90),
        "bracketProb_mean": _safe_mean(bracket_probs),
        "bracketProb_p10": _safe_pctl(bracket_probs, 0.10),
        "bracketProb_p25": bp_p25,
        "bracketProb_p50": _safe_pctl(bracket_probs, 0.50),
        "bracketProb_p75": bp_p75,
        "bracketProb_p90": _safe_pctl(bracket_probs, 0.90),
        "bracketProb_iqr": bp_iqr,
        "replacement_subgraph_mean": _safe_mean(repl_s),
        "replacement_subgraph_median": _safe_median(repl_s),
        "completeness_subgraph_mean": _safe_mean(comp_s),
        "completeness_subgraph_median": _safe_median(comp_s),
        "replacement_subgraph_gate_mean": _safe_mean(repl_s_gate),
        "completeness_subgraph_gate_mean": _safe_mean(comp_s_gate),
        "coverage_mean": _safe_mean(cov),
        "coverage_median": _safe_median(cov),
        "coverage_p25": cov_p25,
        "coverage_p75": cov_p75,
        "coverage_iqr": cov_iqr,
        "matchedFrac_mean": _safe_mean(matched_frac),
        "n_angle_bracket": len(bracket_rows),
        "replacement_subgraph_mean_angle_bracket": _safe_mean(repl_s_br),
        "completeness_subgraph_mean_angle_bracket": _safe_mean(comp_s_br),
        "coverage_mean_angle_bracket": _safe_mean(cov_br),
    }


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input circuit_eval JSON")
    ap.add_argument(
        "--outStem",
        default=None,
        help="Output path stem; default is input path without .json suffix",
    )
    args = ap.parse_args()

    in_path = args.in_path
    out_stem = args.outStem
    if out_stem is None:
        out_stem = in_path
        if out_stem.endswith(".json"):
            out_stem = out_stem[: -len(".json")]

    meta, rows = _load_rows(in_path)

    # Per-label
    by_label: Dict[str, List[Row]] = {}
    for r in rows:
        by_label.setdefault(r.label, []).append(r)

    label_summaries: List[Dict[str, Any]] = []
    for label, grp in sorted(by_label.items(), key=lambda kv: kv[0]):
        s = _summarize_group(grp)
        s.update({"label": label, "family": grp[0].family})
        label_summaries.append(s)

    # Per-family
    by_family: Dict[str, List[Row]] = {}
    for r in rows:
        by_family.setdefault(r.family, []).append(r)

    family_summaries: List[Dict[str, Any]] = []
    for fam, grp in sorted(by_family.items(), key=lambda kv: kv[0]):
        s = _summarize_group(grp)
        s.update({"family": fam})
        family_summaries.append(s)

    # Overall
    overall = {
        "meta": meta,
        "overall": _summarize_group(rows),
        "families": family_summaries,
        "labels": label_summaries,
    }

    overall_json_path = out_stem + ".summary_overall.json"
    with open(overall_json_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "family",
        "label",
        "n",
        "angle_bracket_rate",
        "expected_tool_call_n",
        "expected_echo_n",
        "mode_accuracy",
        "tool_call_expected_hit_rate",
        "echo_expected_tool_call_fp_rate",
        "targetProb_mean",
        "targetProb_median",
        "probAngleBracket_mean",
        "probAngleBracket_p10",
        "probAngleBracket_p50",
        "probAngleBracket_p90",
        "bracketProb_mean",
        "bracketProb_p10",
        "bracketProb_p25",
        "bracketProb_p50",
        "bracketProb_p75",
        "bracketProb_p90",
        "bracketProb_iqr",
        "replacement_subgraph_mean",
        "replacement_subgraph_median",
        "completeness_subgraph_mean",
        "completeness_subgraph_median",
        "replacement_subgraph_gate_mean",
        "completeness_subgraph_gate_mean",
        "coverage_mean",
        "coverage_median",
        "coverage_p25",
        "coverage_p75",
        "coverage_iqr",
        "matchedFrac_mean",
        "n_angle_bracket",
        "replacement_subgraph_mean_angle_bracket",
        "completeness_subgraph_mean_angle_bracket",
        "coverage_mean_angle_bracket",
    ]

    label_csv_path = out_stem + ".summary_by_label.csv"
    _write_csv(label_csv_path, label_summaries, fieldnames)

    family_fieldnames = [fn for fn in fieldnames if fn != "label"]
    family_csv_path = out_stem + ".summary_by_family.csv"
    _write_csv(family_csv_path, family_summaries, family_fieldnames)

    print("Wrote:")
    print(" ", overall_json_path)
    print(" ", label_csv_path)
    print(" ", family_csv_path)


if __name__ == "__main__":
    main()
