"""Build the small JSON payload used by the static GitHub Pages viewer (docs/circuit-viewer/).

Inputs:
  --export   directory written by export_neuronpedia_db.py (features.parquet, activations/<snapshot>/layer_XX.parquet)
  --circuit  results/test1_circuit/test1_circuit_annotated.json
  --graph    results/test1_circuit/source_graph_calculate35.json

Output (--out, a single JSON file):
  circuit:   the annotated manual circuit (nodes, supernodes, labels, base/delta)
  graph:     pruned edge list restricted to the circuit's nodes (for drawing)
  features:  {"<layer>_<feature>": {is_delta, label, examples: {snapshot: [...]}}}
             for every circuit feature and every delta feature of the fused transcoder.
Each example keeps the tokens and per-token activation values (rounded) so the page can
highlight which token fired.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE_WIDTH = 16384


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-examples", type=int, default=5)
    args = ap.parse_args()

    circuit = json.load(open(args.circuit))
    graph = json.load(open(args.graph))

    wanted: dict[int, set[int]] = {}
    labels: dict[str, str] = {}
    for n in circuit["nodes"]:
        if n["type"] == "feature":
            wanted.setdefault(n["layer"], set()).add(n["feature_index"])
            if n.get("label"):
                labels[f'{n["layer"]}_{n["feature_index"]}'] = n["label"]

    features: dict[str, dict] = {}
    snap_dirs = sorted(glob.glob(os.path.join(args.export, "activations", "*")))
    for snap_dir in snap_dirs:
        snap = os.path.basename(snap_dir)
        for path in sorted(glob.glob(os.path.join(snap_dir, "layer_*.parquet"))):
            layer = int(os.path.basename(path)[6:8])
            t = pq.read_table(path)
            mask = pc.or_(t["is_delta"], pc.is_in(t["feature"], value_set=pa.array(sorted(wanted.get(layer, set())), "int32")))
            t = t.filter(mask)
            for r in t.to_pylist():
                key = f'{layer}_{r["feature"]}'
                f = features.setdefault(
                    key,
                    {
                        "layer": layer,
                        "feature": r["feature"],
                        "is_delta": r["feature"] >= BASE_WIDTH,
                        "in_circuit": r["feature"] in wanted.get(layer, set()),
                        "label": labels.get(key),
                        "examples": {},
                    },
                )
                ex = f["examples"].setdefault(snap, [])
                if len(ex) < args.max_examples:
                    ex.append(
                        {
                            "max": round(r["max_value"], 3),
                            "max_idx": r["max_token_index"],
                            "tokens": r["tokens"],
                            "values": [round(v, 2) for v in r["values"]],
                        }
                    )
    for f in features.values():
        for ex in f["examples"].values():
            ex.sort(key=lambda e: -e["max"])

    keep = {n["node_id"] for n in circuit["nodes"]}
    edges = [
        {"source": e["source"], "target": e["target"], "weight": round(e["weight"], 4)}
        for e in graph["links"]
        if e["source"] in keep and e["target"] in keep
    ]

    payload = {
        "circuit": circuit,
        "graph": {"prompt_tokens": graph["metadata"]["prompt_tokens"], "edges": edges},
        "snapshots": [os.path.basename(d) for d in snap_dirs],
        "features": features,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    n_delta = sum(f["is_delta"] for f in features.values())
    print(f"features={len(features)} (delta={n_delta}) edges={len(edges)} -> {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
