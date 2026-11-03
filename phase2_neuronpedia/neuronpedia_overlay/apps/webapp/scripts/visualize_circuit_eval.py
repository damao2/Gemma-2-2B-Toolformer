"""Create lightweight visualizations for circuit-eval outputs.

Goal: make the eval pipeline "complete":
  JSONL prompts -> batch-eval JSON -> analysis summaries -> visualizations.

This script is dependency-free (no matplotlib/pandas). It produces an output
folder containing SVG plots.

Input:
  - original batch-eval JSON: scripts/out/circuit_eval_*.json

Important limitation:
    Older batch-eval output only reported (targetToken, targetProb) for the model's
    top-1 next token. For tool-call gating you likely care about P("<"), not just
    whether the top-1 token is "<".

    Newer batch-eval output may include strict-gating fields:
        - probAngleBracket: P("<")
        - replacement_subgraph_gate / completeness_subgraph_gate: R/C computed with
            logit weights overridden to target the gate token ("<")

So we visualize two things:
  1) the proxy `isAngleBracket = (targetToken == "<")`
  2) circuit influence metrics (replacement/completeness) with a separate view
     for the angle-bracket subset.

Usage:
    /bin/python3 scripts/visualize_circuit_eval.py \
        --in scripts/out/circuit_eval_v2.json \
        --outDir scripts/out/circuit_eval_v2.viz

Optional (off by default):
    --writeIndex   also write an index.html to browse the SVGs
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _label_family(label: str) -> str:
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


def _expected_tool_call_for_family(family: str) -> Optional[bool]:
    if family in ("pos_seen", "pos_unseen"):
        return True
    if family.startswith("neg_"):
        return False
    return None


FAMILY_COLORS: Dict[str, str] = {
    "pos_seen": "#1f77b4",
    "pos_unseen": "#17becf",
    "neg_hard_unseen": "#ff7f0e",
    "neg_soft_unseen": "#d62728",
    "neg_anti_tool": "#9467bd",
    "other": "#7f7f7f",
}


@dataclass(frozen=True)
class Sample:
    label: str
    family: str
    url: str
    target_token: str
    target_prob: float
    prob_angle_bracket: Optional[float]
    replacement_subgraph: float
    completeness_subgraph: float
    replacement_subgraph_gate: Optional[float]
    completeness_subgraph_gate: Optional[float]
    coverage: float

    @property
    def is_angle_bracket(self) -> bool:
        return self.target_token == "<"

    @property
    def bracket_prob_proxy(self) -> float:
        return self.target_prob if self.is_angle_bracket else 0.0

    @property
    def prob_angle_bracket_best_effort(self) -> float:
        """Best-effort P('<').

        Prefer strict `probAngleBracket` if present; else fall back to a
        conservative proxy that's only nonzero when '<' is top-1.
        """

        if self.prob_angle_bracket is not None and math.isfinite(self.prob_angle_bracket):
            return float(self.prob_angle_bracket)
        return float(self.bracket_prob_proxy)

    @property
    def replacement_gate_best_effort(self) -> float:
        if self.replacement_subgraph_gate is not None and math.isfinite(self.replacement_subgraph_gate):
            return float(self.replacement_subgraph_gate)
        return float(self.replacement_subgraph)

    @property
    def completeness_gate_best_effort(self) -> float:
        if self.completeness_subgraph_gate is not None and math.isfinite(self.completeness_subgraph_gate):
            return float(self.completeness_subgraph_gate)
        return float(self.completeness_subgraph)


def _load_samples(path: str) -> Tuple[Dict[str, Any], List[Sample]]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    meta = blob.get("meta", {})
    out: List[Sample] = []
    for r in blob.get("results", []):
        label = str(r.get("label", ""))
        out.append(
            Sample(
                label=label,
                family=_label_family(label),
                url=str(r.get("url", "")),
                target_token=str(r.get("targetToken", "")),
                target_prob=float(r.get("targetProb", float("nan"))),
                prob_angle_bracket=(None if r.get("probAngleBracket", None) is None else float(r.get("probAngleBracket"))),
                replacement_subgraph=float(r.get("replacement_subgraph", float("nan"))),
                completeness_subgraph=float(r.get("completeness_subgraph", float("nan"))),
                replacement_subgraph_gate=(
                    None
                    if r.get("replacement_subgraph_gate", None) is None
                    else float(r.get("replacement_subgraph_gate"))
                ),
                completeness_subgraph_gate=(
                    None
                    if r.get("completeness_subgraph_gate", None) is None
                    else float(r.get("completeness_subgraph_gate"))
                ),
                coverage=float(r.get("coverage", float("nan"))),
            )
        )
    return meta, out


def _finite(xs: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if x is None:
            continue
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            continue
        out.append(float(x))
    return out


def _pctl(xs: Sequence[float], p: float) -> Optional[float]:
    xs2 = sorted(_finite(xs))
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


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs2 = _finite(xs)
    return statistics.mean(xs2) if xs2 else None


def _svg_header(w: int, h: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img">\n'
        '<style>\n'
        '  .axis { stroke: #333; stroke-width: 1; }\n'
        '  .grid { stroke: #ddd; stroke-width: 1; }\n'
        '  .label { font: 12px sans-serif; fill: #111; }\n'
        '  .title { font: 16px sans-serif; font-weight: 600; fill: #111; }\n'
        '  .small { font: 11px sans-serif; fill: #333; }\n'
        '</style>\n'
    )


def _svg_footer() -> str:
    return "</svg>\n"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def bar_chart(
    *,
    title: str,
    items: List[Tuple[str, float, str]],
    width: int = 900,
    height: int = 420,
    y_min: float = 0.0,
    y_max: float = 1.0,
    note: Optional[str] = None,
) -> str:
    """Simple bar chart.

    items: (name, value, color)
    """
    margin_l, margin_r, margin_t, margin_b = 60, 20, 50, 90
    w = width
    h = height
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    n = max(1, len(items))
    gap = 6
    bar_w = max(6, int((plot_w - gap * (n - 1)) / n))

    def x_of(i: int) -> int:
        return margin_l + i * (bar_w + gap)

    def y_of(v: float) -> float:
        v2 = min(max(v, y_min), y_max)
        t = (v2 - y_min) / (y_max - y_min) if y_max > y_min else 0.0
        return margin_t + (1 - t) * plot_h

    parts: List[str] = [_svg_header(w, h)]
    parts.append(f'<text class="title" x="{margin_l}" y="28">{html.escape(title)}</text>\n')

    # grid + y labels
    for frac, lab in [(0.0, str(y_min)), (0.5, str((y_min + y_max) / 2)), (1.0, str(y_max))]:
        y = margin_t + (1 - frac) * plot_h
        parts.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{w-margin_r}" y2="{y:.1f}"/>\n')
        parts.append(f'<text class="small" x="{margin_l-8}" y="{y+4:.1f}" text-anchor="end">{html.escape(lab)}</text>\n')

    # axes
    parts.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t+plot_h}" x2="{w-margin_r}" y2="{margin_t+plot_h}"/>\n')
    parts.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}"/>\n')

    for i, (name, value, color) in enumerate(items):
        x = x_of(i)
        y = y_of(value)
        y0 = margin_t + plot_h
        bh = max(0.0, y0 - y)
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{color}">'
            f'<title>{html.escape(name)}: {value:.4f}</title></rect>\n'
        )
        # x labels rotated
        tx = x + bar_w / 2
        ty = margin_t + plot_h + 12
        parts.append(
            f'<text class="small" x="{tx:.1f}" y="{ty}" text-anchor="end" transform="rotate(-45 {tx:.1f},{ty})">{html.escape(name)}</text>\n'
        )

    if note:
        parts.append(f'<text class="small" x="{margin_l}" y="{h-12}">{html.escape(note)}</text>\n')

    parts.append(_svg_footer())
    return "".join(parts)


def scatter_chart(
    *,
    title: str,
    points: List[Tuple[float, float, str, str]],
    x_label: str,
    y_label: str,
    width: int = 900,
    height: int = 520,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    note: Optional[str] = None,
) -> str:
    """Simple scatter plot.

    points: (x, y, color, tooltip)
    """
    margin_l, margin_r, margin_t, margin_b = 70, 25, 55, 70
    w, h = width, height
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    def x_of(v: float) -> float:
        v2 = min(max(v, x_min), x_max)
        t = (v2 - x_min) / (x_max - x_min) if x_max > x_min else 0.0
        return margin_l + t * plot_w

    def y_of(v: float) -> float:
        v2 = min(max(v, y_min), y_max)
        t = (v2 - y_min) / (y_max - y_min) if y_max > y_min else 0.0
        return margin_t + (1 - t) * plot_h

    parts: List[str] = [_svg_header(w, h)]
    parts.append(f'<text class="title" x="{margin_l}" y="30">{html.escape(title)}</text>\n')

    # grid lines
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = margin_l + frac * plot_w
        y = margin_t + (1 - frac) * plot_h
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{margin_t+plot_h}"/>\n')
        parts.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{margin_l+plot_w}" y2="{y:.1f}"/>\n')

    # axes
    parts.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}"/>\n')
    parts.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}"/>\n')

    # axis labels
    parts.append(f'<text class="label" x="{margin_l+plot_w/2:.1f}" y="{h-18}" text-anchor="middle">{html.escape(x_label)}</text>\n')
    parts.append(
        f'<text class="label" x="18" y="{margin_t+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 18,{margin_t+plot_h/2:.1f})">{html.escape(y_label)}</text>\n'
    )

    for x, y, color, tooltip in points:
        cx = x_of(x)
        cy = y_of(y)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" fill-opacity="0.85" stroke="#111" stroke-width="0.4">'
            f'<title>{html.escape(tooltip)}</title></circle>\n'
        )

    if note:
        parts.append(f'<text class="small" x="{margin_l}" y="{h-10}">{html.escape(note)}</text>\n')

    parts.append(_svg_footer())
    return "".join(parts)


def _summaries_by_family(samples: Sequence[Sample]) -> List[Dict[str, Any]]:
    by: Dict[str, List[Sample]] = {}
    for s in samples:
        by.setdefault(s.family, []).append(s)

    out: List[Dict[str, Any]] = []
    for fam, grp in sorted(by.items(), key=lambda kv: kv[0]):
        out.append(
            {
                "family": fam,
                "n": len(grp),
                "angle_bracket_rate": _mean([1.0 if g.is_angle_bracket else 0.0 for g in grp]) or 0.0,
                "bracket_prob_mean": _mean([g.bracket_prob_proxy for g in grp]) or 0.0,
                "prob_angle_bracket_mean": _mean([g.prob_angle_bracket_best_effort for g in grp]) or 0.0,
                "replacement_mean": _mean([g.replacement_subgraph for g in grp]) or 0.0,
                "completeness_mean": _mean([g.completeness_subgraph for g in grp]) or 0.0,
                "replacement_gate_mean": _mean([g.replacement_gate_best_effort for g in grp]) or 0.0,
                "completeness_gate_mean": _mean([g.completeness_gate_best_effort for g in grp]) or 0.0,
                "coverage_mean": _mean([g.coverage for g in grp]) or 0.0,
            }
        )
    return out


def _summaries_by_label(samples: Sequence[Sample]) -> List[Dict[str, Any]]:
    by: Dict[str, List[Sample]] = {}
    for s in samples:
        by.setdefault(s.label, []).append(s)

    out: List[Dict[str, Any]] = []
    for label, grp in sorted(by.items(), key=lambda kv: kv[0]):
        fam = grp[0].family
        out.append(
            {
                "label": label,
                "family": fam,
                "n": len(grp),
                "angle_bracket_rate": _mean([1.0 if g.is_angle_bracket else 0.0 for g in grp]) or 0.0,
                "bracket_prob_mean": _mean([g.bracket_prob_proxy for g in grp]) or 0.0,
                "prob_angle_bracket_mean": _mean([g.prob_angle_bracket_best_effort for g in grp]) or 0.0,
                "replacement_mean": _mean([g.replacement_subgraph for g in grp]) or 0.0,
                "completeness_mean": _mean([g.completeness_subgraph for g in grp]) or 0.0,
                "replacement_gate_mean": _mean([g.replacement_gate_best_effort for g in grp]) or 0.0,
                "completeness_gate_mean": _mean([g.completeness_gate_best_effort for g in grp]) or 0.0,
                "coverage_mean": _mean([g.coverage for g in grp]) or 0.0,
            }
        )
    return out


def _html_index(*, out_dir: str, meta: Dict[str, Any], plots: List[Tuple[str, str]]) -> str:
    title = "Circuit Eval Visualization"
    model_id = str(meta.get("modelId", ""))
    subgraph = str(meta.get("subgraphName", ""))
    semantics = str(meta.get("scoringSemantics", ""))

    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "  body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: #111; }",
        "  .meta { margin: 8px 0 18px; color: #333; }",
        "  .plot { margin: 22px 0; padding: 14px; border: 1px solid #ddd; border-radius: 10px; }",
        "  .plot h2 { margin: 0 0 10px; font-size: 16px; }",
        "  .note { color: #444; font-size: 13px; line-height: 1.35; }",
        "  img { max-width: 100%; height: auto; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        "<div class='meta'>",
        f"<div><b>modelId</b>: {html.escape(model_id)}</div>",
        f"<div><b>subgraph</b>: {html.escape(subgraph)}</div>",
        f"<div><b>semantics</b>: {html.escape(semantics)}</div>",
        "</div>",
        "<div class='note'>",
        "<b>Target warning</b>: for tool-call gating you want P('&lt;'). ",
        "If strict-gating fields are present we use them (probAngleBracket + *_gate scores). ",
        "Otherwise we fall back to a conservative proxy based on the top-1 token: ",
        "<code>isAngleBracket = (targetToken == '&lt;')</code> and <code>bracketProbProxy = targetProb if isAngleBracket else 0</code>.",
        "</div>",
    ]

    for filename, caption in plots:
        parts.append("<div class='plot'>")
        parts.append(f"<h2>{html.escape(caption)}</h2>")
        parts.append(f"<img src='{html.escape(filename)}' alt='{html.escape(caption)}' />")
        parts.append("</div>")

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input circuit_eval JSON")
    ap.add_argument("--outDir", required=True, help="Output directory for SVG + index.html")
    ap.add_argument("--topLabels", type=int, default=30, help="Show top-N labels in label bar charts")
    ap.add_argument("--writeIndex", action="store_true", help="Also write index.html for browsing")
    args = ap.parse_args()

    meta, samples = _load_samples(args.in_path)
    out_dir = args.outDir
    os.makedirs(out_dir, exist_ok=True)

    fam = _summaries_by_family(samples)
    lab = _summaries_by_label(samples)

    plots: List[Tuple[str, str]] = []

    # Family bar charts
    items = [(d["family"], float(d["angle_bracket_rate"]), FAMILY_COLORS.get(d["family"], "#777")) for d in fam]
    svg = bar_chart(title="Angle-bracket rate by family (proxy tool-call hit)", items=items, y_min=0.0, y_max=1.0)
    _write(os.path.join(out_dir, "family_angle_bracket_rate.svg"), svg)
    plots.append(("family_angle_bracket_rate.svg", "Angle-bracket rate by family"))

    # Mode accuracy: POS expects '<' (tool-call), NEG expects not '<' (echo)
    acc_items: List[Tuple[str, float, str]] = []
    for d in fam:
        family = str(d["family"])
        exp_tool = _expected_tool_call_for_family(family)
        rate = float(d["angle_bracket_rate"])
        if exp_tool is None:
            continue
        acc = rate if exp_tool else (1.0 - rate)
        acc_items.append((family, acc, FAMILY_COLORS.get(family, "#777")))

    if acc_items:
        svg = bar_chart(
            title="Mode accuracy by family (POS expects '<', NEG expects echo)",
            items=acc_items,
            y_min=0.0,
            y_max=1.0,
        )
        _write(os.path.join(out_dir, "family_mode_accuracy.svg"), svg)
        plots.append(("family_mode_accuracy.svg", "Mode accuracy by family"))

    items = [(d["family"], float(d["prob_angle_bracket_mean"]), FAMILY_COLORS.get(d["family"], "#777")) for d in fam]
    svg = bar_chart(title="P('<') mean by family (strict if available; else proxy)", items=items, y_min=0.0, y_max=1.0)
    _write(os.path.join(out_dir, "family_bracket_prob_mean.svg"), svg)
    plots.append(("family_bracket_prob_mean.svg", "P('<') mean by family"))

    items = [(d["family"], float(d["replacement_gate_mean"]), FAMILY_COLORS.get(d["family"], "#777")) for d in fam]
    svg = bar_chart(title="Replacement (gate token) mean by family", items=items, y_min=0.0, y_max=1.0)
    _write(os.path.join(out_dir, "family_replacement_mean.svg"), svg)
    plots.append(("family_replacement_mean.svg", "Replacement (gate token) mean by family"))

    items = [(d["family"], float(d["completeness_gate_mean"]), FAMILY_COLORS.get(d["family"], "#777")) for d in fam]
    svg = bar_chart(title="Completeness (gate token) mean by family", items=items, y_min=0.0, y_max=1.0)
    _write(os.path.join(out_dir, "family_completeness_mean.svg"), svg)
    plots.append(("family_completeness_mean.svg", "Completeness (gate token) mean by family"))

    items = [(d["family"], float(d["coverage_mean"]), FAMILY_COLORS.get(d["family"], "#777")) for d in fam]
    svg = bar_chart(title="Coverage mean by family", items=items, y_min=0.0, y_max=1.0)
    _write(os.path.join(out_dir, "family_coverage_mean.svg"), svg)
    plots.append(("family_coverage_mean.svg", "Coverage mean by family"))

    # Label: top-N by P('<') mean (strict if available; else proxy)
    lab_sorted = sorted(lab, key=lambda d: float(d["prob_angle_bracket_mean"]), reverse=True)
    top = lab_sorted[: max(1, args.topLabels)]
    items = [(d["label"], float(d["prob_angle_bracket_mean"]), FAMILY_COLORS.get(d["family"], "#777")) for d in top]
    svg = bar_chart(
        title=f"Top-{len(top)} labels by P('<') mean (strict if available; else proxy)",
        items=items,
        y_min=0.0,
        y_max=1.0,
        note="Bars colored by family",
    )
    _write(os.path.join(out_dir, "label_top_bracket_prob.svg"), svg)
    plots.append(("label_top_bracket_prob.svg", f"Top-{len(top)} labels by P('<')"))

    # Scatter: replacement vs completeness (per-sample; gate-aligned if available)
    pts: List[Tuple[float, float, str, str]] = []
    for s in samples:
        color = FAMILY_COLORS.get(s.family, "#777")
        tooltip = (
            f"{s.label} | top1={s.target_token} top1p={s.target_prob:.4f} "
            f"P('<')={s.prob_angle_bracket_best_effort:.4f} cov={s.coverage:.3f} | {s.url}"
        )
        pts.append((s.replacement_gate_best_effort, s.completeness_gate_best_effort, color, tooltip))

    svg = scatter_chart(
        title="Per-sample: replacement vs completeness (gate token if available)",
        points=pts,
        x_label="replacement_subgraph_gate (fallback: replacement_subgraph)",
        y_label="completeness_subgraph_gate (fallback: completeness_subgraph)",
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        note="Tooltip includes label + top-1 token + P('<') + URL",
    )
    _write(os.path.join(out_dir, "scatter_repl_vs_comp.svg"), svg)
    plots.append(("scatter_repl_vs_comp.svg", "Per-sample scatter: replacement vs completeness"))

    # Scatter: coverage vs P('<') (strict if available; else proxy)
    pts2: List[Tuple[float, float, str, str]] = []
    for s in samples:
        color = FAMILY_COLORS.get(s.family, "#777")
        tooltip = (
            f"{s.label} | top1={s.target_token} top1p={s.target_prob:.4f} "
            f"P('<')={s.prob_angle_bracket_best_effort:.4f} "
            f"repl={s.replacement_gate_best_effort:.3f} comp={s.completeness_gate_best_effort:.3f} | {s.url}"
        )
        pts2.append((s.coverage, s.prob_angle_bracket_best_effort, color, tooltip))

    svg = scatter_chart(
        title="Per-sample: coverage vs P('<') (strict if available; else proxy)",
        points=pts2,
        x_label="coverage",
        y_label="P('<')",
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        note="Low coverage often means transfer-key mismatch, not circuit failure",
    )
    _write(os.path.join(out_dir, "scatter_cov_vs_bracketprob.svg"), svg)
    plots.append(("scatter_cov_vs_bracketprob.svg", "Per-sample scatter: coverage vs BracketProbProxy"))

    if args.writeIndex:
        index_html = _html_index(out_dir=out_dir, meta=meta, plots=plots)
        _write(os.path.join(out_dir, "index.html"), index_html)

    print("Wrote SVGs to:")
    print(" ", out_dir)
    for filename, _caption in plots:
        print("  -", filename)
    if args.writeIndex:
        print("Wrote:")
        print("  - index.html")


if __name__ == "__main__":
    main()
