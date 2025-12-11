import os
import csv
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Exp1 results over (d_new, percentile) combos")
    p.add_argument("--root_dir", type=str, required=True,
                   help="Root dir containing exp1_outputs_* subdirs")
    p.add_argument("--d_news", type=str, default="32,64,128",
                   help="Comma-separated d_new list, e.g. '32,64,128'")
    p.add_argument("--percentiles", type=str, default="0.25,0.50,0.75",
                   help="Comma-separated percentile list, e.g. '0.25,0.50,0.75'")
    p.add_argument("--out_dir", type=str, default="./exp1/aggregate",
                   help="Where to save aggregated CSV/plots")
    return p.parse_args()


def load_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(x):
    return float(x)


def aggregate_combo(rows: List[Dict]) -> Dict:
    """Make an overall summary for the per-layer rows of a single (d_new, pctl) combination"""
    # arrays over layers
    dce_full = np.array([to_float(r["dce_full"]) for r in rows], dtype=float)
    dce_base = np.array([to_float(r["dce_base"]) for r in rows], dtype=float)
    cos_full = np.array([to_float(r["full_mean_cos"]) for r in rows], dtype=float)
    cos_base = np.array([to_float(r["base_mean_cos"]) for r in rows], dtype=float)
    l2_full = np.array([to_float(r["full_mean_l2"]) for r in rows], dtype=float)
    l2_base = np.array([to_float(r["base_mean_l2"]) for r in rows], dtype=float)

    # Simple summary: average / maximum by layer, etc.
    summary = dict(
        mean_dce_full=float(dce_full.mean()),
        max_dce_full=float(dce_full.max()),
        mean_dce_base=float(dce_base.mean()),
        mean_cos_full=float(cos_full.mean()),
        mean_cos_base=float(cos_base.mean()),
        mean_l2_full=float(l2_full.mean()),
        mean_l2_base=float(l2_base.mean()),
    )
    return summary


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d_news = [int(x) for x in args.d_news.split(",") if x.strip()]
    pcts = [float(x) for x in args.percentiles.split(",") if x.strip()]

    all_summaries: List[Dict] = []

    for d_new in d_news:
        for p in pcts:
            tag = f"d{d_new}_p{p}"
            p_str = f"{p:.2f}"
            # Assume your directory names are like: exp1_outputs_d32_p0.50
            subdir = os.path.join(args.root_dir, f"exp1_outputs_d{d_new}_p{p_str}")
            csv_path = os.path.join(subdir, f"exp1_metrics_d{d_new}_p{p}.csv")
            if not os.path.exists(csv_path):
                print(f"[WARN] Missing CSV for d_new={d_new}, p={p} at {csv_path}, skip.")
                continue
            print(f"[Load] {csv_path}")
            rows = load_csv(csv_path)
            summ = aggregate_combo(rows)
            summ["d_new"] = d_new
            summ["percentile"] = p
            all_summaries.append(summ)

    if not all_summaries:
        print("[ERROR] No summaries collected, check paths/root_dir/d_news/percentiles.")
        return

    # Save summary CSV / JSON
    summary_csv = os.path.join(args.out_dir, "exp1_combo_summary.csv")
    summary_json = os.path.join(args.out_dir, "exp1_combo_summary.json")

    fieldnames = list(all_summaries[0].keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in all_summaries:
            wr.writerow(r)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[Save] Summary saved to:\n  {summary_csv}\n  {summary_json}")

    # ---- Plotting: heatmap: x=d_new, y=pctl, value=mean_dce_full / mean_cos_full ----
    # First do pivot
    d_new_sorted = sorted({r["d_new"] for r in all_summaries})
    p_sorted = sorted({r["percentile"] for r in all_summaries})

    def build_matrix(key: str):
        mat = np.full((len(p_sorted), len(d_new_sorted)), np.nan, dtype=float)
        for r in all_summaries:
            i = p_sorted.index(r["percentile"])
            j = d_new_sorted.index(r["d_new"])
            mat[i, j] = r[key]
        return mat

    mat_dce = build_matrix("mean_dce_full")
    mat_cos = build_matrix("mean_cos_full")

    # ΔCE heatmap
    plt.figure(figsize=(6, 4))
    im = plt.imshow(mat_dce, origin="lower", aspect="auto",
                    extent=[min(d_new_sorted)-0.5, max(d_new_sorted)+0.5,
                            min(p_sorted)-0.01, max(p_sorted)+0.01],
                    cmap="viridis")
    plt.colorbar(im, label="mean ΔCE_full (vs RAW)")
    plt.xticks(d_new_sorted)
    plt.yticks(p_sorted)
    plt.xlabel("d_new")
    plt.ylabel("percentile")
    plt.title("Exp1 mean ΔCE_full over (d_new, percentile)")
    plt.tight_layout()
    fig_dce = os.path.join(args.out_dir, "exp1_mean_dce_full_heatmap.png")
    plt.savefig(fig_dce)
    plt.close()
    print(f"[Save] Heatmap (ΔCE_full) -> {fig_dce}")

    # cosine heatmap
    plt.figure(figsize=(6, 4))
    im = plt.imshow(mat_cos, origin="lower", aspect="auto",
                    extent=[min(d_new_sorted)-0.5, max(d_new_sorted)+0.5,
                            min(p_sorted)-0.01, max(p_sorted)+0.01],
                    cmap="viridis")
    plt.colorbar(im, label="mean cos_full")
    plt.xticks(d_new_sorted)
    plt.yticks(p_sorted)
    plt.xlabel("d_new")
    plt.ylabel("percentile")
    plt.title("Exp1 mean cos_full over (d_new, percentile)")
    plt.tight_layout()
    fig_cos = os.path.join(args.out_dir, "exp1_mean_cos_full_heatmap.png")
    plt.savefig(fig_cos)
    plt.close()
    print(f"[Save] Heatmap (cos_full) -> {fig_cos}")


if __name__ == "__main__":
    main()
'''
python3 exp1_aggregate_results.py \
  --root_dir ./exp1 \
  --d_news 32,64,128 \
  --percentiles 0.25,0.50,0.75 \
  --out_dir ./exp1/aggregate
'''