"""Export the local Neuronpedia Postgres DB (feature top-activations, graphs, subgraphs) to portable files.

The local Neuronpedia instance stored, for every transcoder feature of the fine-tuned
Gemma-2-2B Toolformer model, its top-k activating text snippets (table "Activation")
plus per-feature metadata (table "Neuron"). This script dumps them to Parquet so they can
be published (e.g. as a Hugging Face dataset) without running Postgres.

Outputs (under --out):
  features.parquet                      one row per feature: layer, feature, is_delta, max_act_approx, frac_nonzero, n_examples_general_8k
  activations/<snapshot>/layer_XX.parquet one row per stored top-activation example (see SNAPSHOTS)
  graphs/graph_metadata.jsonl           rows of "GraphMetadata"
  graphs/subgraphs.jsonl                rows of "GraphMetadataSubgraph" (manually pinned circuits)

Feature indexing: the fused transcoder keeps the 16384 Gemma-Scope base features at indices
[0, 16384) and appends the delta features after them, so is_delta = feature >= 16384.

Usage:
  python export_neuronpedia_db.py --dsn "host=<socket dir> port=5439 user=qc62 dbname=neuronpedia_local" --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import psycopg2
import psycopg2.extras
import pyarrow as pa
import pyarrow.parquet as pq

MODEL_ID = "local-gemma-2b-toolformer"
SOURCE_SET = "toolformer-fused"
N_LAYERS = 26
BASE_WIDTH = 16384

ACT_SCHEMA = pa.schema(
    [
        ("layer", pa.int8()),
        ("feature", pa.int32()),
        ("is_delta", pa.bool_()),
        ("max_value", pa.float32()),
        ("max_token_index", pa.int16()),
        ("tokens", pa.list_(pa.string())),
        ("values", pa.list_(pa.float32())),
        ("created_at", pa.timestamp("ms")),
    ]
)


def export_features(conn, out_dir: str) -> None:
    sql = f"""
        SELECT n.layer, n.index, n."maxActApprox", n.frac_nonzero, COALESCE(a.cnt, 0)
        FROM "Neuron" n
        LEFT JOIN (
            SELECT layer, index, COUNT(*) cnt FROM "Activation"
            WHERE "modelId" = %s AND "createdAt" >= '2026-01-27 11:00' AND "createdAt" < '2026-01-27 12:00'
            GROUP BY 1, 2
        ) a USING (layer, index)
        WHERE n."modelId" = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (MODEL_ID, MODEL_ID))
        rows = cur.fetchall()
    layers, feats = [], []
    for r in rows:
        layers.append(int(r[0].split("-", 1)[0]))
        feats.append(int(r[1]))
    table = pa.table(
        {
            "layer": pa.array(layers, pa.int8()),
            "feature": pa.array(feats, pa.int32()),
            "is_delta": pa.array([f >= BASE_WIDTH for f in feats]),
            "max_act_approx": pa.array([r[2] for r in rows], pa.float32()),
            "frac_nonzero": pa.array([r[3] for r in rows], pa.float32()),
            "n_examples_general_8k": pa.array([r[4] for r in rows], pa.int32()),
        }
    )
    table = table.sort_by([("layer", "ascending"), ("feature", "ascending")])
    pq.write_table(table, os.path.join(out_dir, "features.parquet"), compression="zstd")
    print(f"features.parquet: {table.num_rows} rows", flush=True)


# The collector (collect_top_activations.py) periodically flushes its *cumulative* top-k table,
# so the DB holds several full snapshots per run. Only the last committed snapshot of each run
# is exported. Windows are [start, end) on "createdAt" and were checked against the slurm logs.
SNAPSHOTS = {
    # corpus_20k (mixed toolcall + web text), job cancelled by time limit after the 8000-prompt flush
    "general_corpus_8k": ("2026-01-27 11:00", "2026-01-27 12:00"),
    # toolcall_activations_corpus_500, the 500-prompt flush was killed mid-transaction -> 400 is last
    "toolcall_corpus_400": ("2026-01-28 15:00", "2026-01-28 16:00"),
}


def export_activations(conn, out_dir: str, batch: int = 50_000) -> None:
    for snap, (start, end) in SNAPSHOTS.items():
        act_dir = os.path.join(out_dir, "activations", snap)
        os.makedirs(act_dir, exist_ok=True)
        total = 0
        for layer in range(N_LAYERS):
            sql = """
                SELECT index, "maxValue", "maxValueTokenIndex", tokens, values, "createdAt"
                FROM "Activation"
                WHERE "modelId" = %s AND layer = %s AND "createdAt" >= %s AND "createdAt" < %s
                ORDER BY index::int, "maxValue" DESC
            """
            path = os.path.join(act_dir, f"layer_{layer:02d}.parquet")
            writer = pq.ParquetWriter(path, ACT_SCHEMA, compression="zstd")
            n = 0
            with conn.cursor(name=f"act_{snap}_{layer}") as cur:
                cur.itersize = batch
                cur.execute(sql, (MODEL_ID, f"{layer}-{SOURCE_SET}", start, end))
                while True:
                    rows = cur.fetchmany(batch)
                    if not rows:
                        break
                    feats = [int(r[0]) for r in rows]
                    writer.write_table(
                        pa.table(
                            {
                                "layer": pa.array([layer] * len(rows), pa.int8()),
                                "feature": pa.array(feats, pa.int32()),
                                "is_delta": pa.array([f >= BASE_WIDTH for f in feats]),
                                "max_value": pa.array([r[1] for r in rows], pa.float32()),
                                "max_token_index": pa.array([r[2] for r in rows], pa.int16()),
                                "tokens": pa.array([r[3] for r in rows], pa.list_(pa.string())),
                                "values": pa.array([r[4] for r in rows], pa.list_(pa.float32())),
                                "created_at": pa.array([r[5] for r in rows], pa.timestamp("ms")),
                            },
                            schema=ACT_SCHEMA,
                        )
                    )
                    n += len(rows)
            writer.close()
            total += n
            print(f"[{snap}] layer {layer:02d}: {n} rows", flush=True)
        print(f"[{snap}] total {total} rows", flush=True)


def export_jsonl(conn, out_dir: str) -> None:
    gdir = os.path.join(out_dir, "graphs")
    os.makedirs(gdir, exist_ok=True)
    for table, fname in [("GraphMetadata", "graph_metadata.jsonl"), ("GraphMetadataSubgraph", "subgraphs.jsonl")]:
        with conn.cursor() as cur:
            cur.execute(f'SELECT row_to_json(t) FROM "{table}" t')
            rows = [r[0] for r in cur.fetchall()]
        with open(os.path.join(gdir, fname), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{fname}: {len(rows)} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-activations", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    conn = psycopg2.connect(args.dsn)
    conn.set_session(readonly=True)
    export_jsonl(conn, args.out)
    export_features(conn, args.out)
    if not args.skip_activations:
        export_activations(conn, args.out)
    conn.close()


if __name__ == "__main__":
    main()
