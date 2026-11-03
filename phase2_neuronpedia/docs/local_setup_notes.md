# Running the local Neuronpedia stack

Working notes for running the self-hosted Neuronpedia used in Phase 2 on a GPU cluster
without root access. Paths are placeholders.

| Component | Port | Notes |
|---|---|---|
| Postgres 15 (built from source, with pgvector) | 5433 | holds models, sources, graphs, saved subgraphs and feature activations |
| Graph server (`apps/graph`, circuit-tracer backend) | 5004 | the webapp hard-codes `http://127.0.0.1:5004` as its localhost graph host |
| Webapp (`apps/webapp`, Next.js) | 3000 | UI and `/api/graph/generate` |

## 1. Postgres

The database files persist on disk, so after reconnecting to a compute node it only needs
to be started again:

```bash
cd <db_root>
./local/pgsql/bin/pg_ctl -D local/pgdata -l local/pg.log start -o "-p 5433"
./local/pgsql/bin/pg_isready -h localhost -p 5433    # check
./local/pgsql/bin/pg_ctl -D local/pgdata stop        # stop
```

## 2. Graph server

```bash
cd <neuronpedia>/apps/graph
poetry run python start.py \
  --model_id <path>/gemma_2b_toolformer_merged_v4 \
  --transcoder_set damaoo/gemma2b-toolformer-fused-transcoders \
  --model_dtype float32 \
  --port 5004
```

`--transcoder_set` also accepts a local directory of fused transcoders, e.g.
`<path>/fused_delta_32_pctl_0.5`. On Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

**Authentication.** The graph server reads `SECRET` from `apps/graph/.env`, and the webapp
sends `GRAPH_SERVER_SECRET` from its env file. The two must be equal, otherwise requests fail
with `x-secret-key header missing` or `Invalid x-secret-key`.

Test the server directly:

```bash
curl -sS -X POST http://localhost:5004/generate-graph \
  -H "x-secret-key: <SECRET>" -H "Content-Type: application/json" \
  -d '{
    "prompt": "Calculate 3+5.",
    "model_id": "<path>/gemma_2b_toolformer_merged_v4",
    "slug_identifier": "local-gemma2b-demo",
    "batch_size": 24,
    "max_n_logits": 10,
    "desired_logit_prob": 0.95,
    "node_threshold": 0.8,
    "edge_threshold": 0.98,
    "max_feature_nodes": 5000
  }' > graph.json
```

## 3. Webapp

```bash
cd <neuronpedia>/apps/webapp
npm install                                                        # first time only
npx env-cmd -f .env.localdb prisma db push                         # first time, or on missing tables/columns
npx env-cmd -f .env.localdb --use-shell "prisma generate && next dev"
```

`.env.localdb` points `DATABASE_URL` at the Postgres above and sets `GRAPH_SERVER_SECRET`.
The local model and source set (`local-gemma-2b-toolformer`, `toolformer-fused`) are
registered with `scripts/seed-local-graph-model.ts`.

To use the UI from a laptop, forward both ports through the login node:

```bash
ssh -L 3000:<compute-node>:3000 -L 5004:<compute-node>:5004 <user>@<login-node>
```

## 4. Feature dashboards (top activations)

`corpus/collect_top_activations.sh` runs `apps/graph/collect_top_activations.py`, which runs
the model and fused transcoders over a text corpus, keeps the top-k activating snippets per
feature, and writes them to the `Neuron` and `Activation` tables. Features are stored with
`layer = "{layer}-{SOURCE_SET}"` (e.g. `20-toolformer-fused`) and `index = str(feature)`,
which is the encoding the webapp's feature API expects.

Check the result:

```sql
-- psql postgresql://<user>@localhost:5433/neuronpedia_local
SELECT "modelId", "layer", "index", "maxActApprox", "frac_nonzero"
FROM "Neuron" WHERE "modelId" = 'local-gemma-2b-toolformer' LIMIT 5;

SELECT "tokens", "values", "maxValue", "maxValueTokenIndex"
FROM "Activation" WHERE "modelId" = 'local-gemma-2b-toolformer' LIMIT 5;
```

Then open a feature page in the UI and check that its activation panel loads.

## 5. Building the evaluation set

```bash
cd <neuronpedia>/apps/webapp
python3 scripts/gen_circuit_eval_prompts.py --out scripts/prompts_circuit_eval_v3.jsonl --seed 0

npx env-cmd -f .env.localdb --use-shell "npx tsx scripts/screen_circuit_eval_prompts.ts \
  --in scripts/prompts_circuit_eval_v3.jsonl \
  --outStem scripts/out/prompts_circuit_eval_v3 \
  --modelId local-gemma-2b-toolformer \
  --sourceSetName toolformer-fused \
  --maxNLogits 15 \
  --desiredLogitProb 0.99 \
  --boundaryThreshold 0.10"
```

Screening runs one forward pass per prompt and writes the capability / boundary / stress
slices. `--boundaryThreshold` defaults to 0.40.

## 6. Batch circuit evaluation

Save a subgraph in the UI first. The script loads it from Postgres by name:

```bash
cd <neuronpedia>/apps/webapp
npx env-cmd -f .env.localdb --use-shell "npx tsx scripts/batch-eval-subgraph.ts \
  --baseUrl http://localhost:3000 \
  --modelId local-gemma-2b-toolformer \
  --subgraphName test1 \
  --prompts scripts/prompts_circuit_eval_v3.jsonl \
  --out scripts/out/circuit_eval_v3.json \
  --sourceSetName toolformer-fused"
```

`--sourceSlug` is optional. Without it, the most recently updated subgraph with that
`(modelId, displayName)` is used, together with its source graph. Passing `--sourceSetName`
avoids depending on the model's default source set. See
[`batch_eval_explained.md`](batch_eval_explained.md) for what the outputs mean.

## Troubleshooting

* **Graph generation fails:** check the error from `/api/graph/generate` in the webapp
  terminal first, then the graph server terminal (usually authentication, model inference or
  transcoder loading).
* **Subgraph not found:** the saved `displayName` must match `--subgraphName` exactly (case
  sensitive).
