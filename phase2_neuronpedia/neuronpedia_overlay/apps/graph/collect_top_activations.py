import os
import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable
import uuid
import resource

import psycopg2
from psycopg2.extras import execute_values
import torch

# NOTE: This script assumes you have `circuit_tracer` and its dependencies
# installed in the same environment where you run it.
# It intentionally does NOT depend on Neuronpedia webapp code – it only writes
# directly to the Postgres DB that Neuronpedia uses.


@dataclass
class TopActivation:
  """Single top activation example for one feature.

  This mirrors the important fields needed for the `Activation` table.
  """

  tokens: List[str]
  values: List[float]
  max_value: float
  max_index: int
  data_source: str | None = None


FeatureKey = Tuple[int, int]  # (layer, feature_index)


class TopKCollector:
  """Maintain top-K activations per feature across a corpus."""

  def __init__(self, k: int) -> None:
    self.k = k
    self._store: Dict[FeatureKey, List[TopActivation]] = defaultdict(list)

  def add(self, layer: int, feature_index: int, act: TopActivation) -> None:
    key = (layer, feature_index)
    lst = self._store[key]
    lst.append(act)
    # keep only top-k by max_value
    lst.sort(key=lambda a: a.max_value, reverse=True)
    if len(lst) > self.k:
      del lst[self.k :]

  def items(self) -> Iterable[Tuple[FeatureKey, List[TopActivation]]]:
    return self._store.items()

  def stats(self) -> Tuple[int, int]:
    """Return (num_features, num_top_activations) currently held in memory."""

    num_features = len(self._store)
    num_top_acts = sum(len(v) for v in self._store.values())
    return num_features, num_top_acts


def get_memory_usage_mb() -> float:
  """Return current max RSS (resident set size) in MB.

  On Linux, resource.getrusage(RUSAGE_SELF).ru_maxrss is in kilobytes.
  """

  usage = resource.getrusage(resource.RUSAGE_SELF)
  return float(usage.ru_maxrss) / 1024.0


def iter_texts(path: str) -> Iterable[Tuple[int, str]]:
  """Yield (line_idx, text) from a UTF-8 text file (one sample per line)."""

  with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
      line = line.strip()
      if line:
        yield i, line


# --------------------------
# PLACEHOLDER: model-specific logic
# --------------------------

"""We keep this as a thin abstraction so you can plug in your
own circuit_tracer / ReplacementModel code.

In practice, for this repo we just reuse the already-configured
model from `neuronpedia.apps.graph.neuronpedia_graph.server`.
"""


def load_model():
  """Reuse the graph server's loaded ReplacementModel.

  This respects all the existing env-var based config:
  - MODEL_ID, TRANSCODER_SET, OFFICIAL_TLENS_MODEL_ID
  - DEVICE, MODEL_DTYPE, etc.
  """

  # Import inside the function so that just importing this script
  # does not immediately load the large model.
  from neuronpedia_graph import server as graph_server

  return graph_server.model


def _decode_tokens(model, input_ids: List[int]) -> List[str]:
  """Decode a list of token ids to string tokens, one by one.

  This matches how the steer endpoint decodes tokens.
  """

  return [model.tokenizer.decode([tid]) for tid in input_ids]


def get_feature_activations(model, text: str) -> Dict[FeatureKey, TopActivation]:
  """Run the model on `text` and return per-feature activations.

  Implementation details:
  - Uses `model.get_activations(text, sparse=True)` (same API used by the
    steer endpoint) to obtain feature activations keyed by (layer, pos, idx).
  - Reconstructs, for each (layer, feature_index), a length-T activation
    vector over tokens, then takes the max over tokens.

  Return value:
    dict[(layer:int, feature_index:int)] = TopActivation
  """

  # Tokenize the prompt similar to the graph server
  token_ids = model.tokenizer.encode(text, add_special_tokens=False)
  seq_len = len(token_ids)
  if seq_len == 0:
    return {}

  str_tokens = _decode_tokens(model, token_ids)

  # Get sparse activations. This is the same API used in the steer handler:
  #   _, activations = model.get_activations(prompt, sparse=True)
  _, activations = model.get_activations(text, sparse=True)

  feature_to_values: Dict[FeatureKey, List[float]] = defaultdict(
    lambda: [0.0] * seq_len
  )

  # Case 1: activations is mapping-like (sparse dict keyed by (layer, pos, idx))
  if hasattr(activations, "items"):
    for key, value in activations.items():
      try:
        layer, pos, feat_idx = key
      except ValueError:
        # Unexpected key shape; skip
        continue

      if not isinstance(pos, int) or pos < 0 or pos >= seq_len:
        continue

      if hasattr(value, "item"):
        v = float(value.item())
      else:
        v = float(value)

      feature_to_values[(int(layer), int(feat_idx))][pos] = v

  # Case 2: activations is a tensor [layers, seq_len, n_features]
  elif isinstance(activations, torch.Tensor):
    # `sparse=True` may return a sparse tensor. Densify in that case,
    # since fused transcoders usually have small feature dims.
    if activations.layout is not torch.strided:
      acts = activations.to_dense().cpu()
    else:
      acts = activations.detach().cpu()
    if acts.ndim != 3:
      raise RuntimeError(
        f"Unexpected activations tensor shape {acts.shape}; expected [layers, seq_len, n_features]"
      )
    n_layers, t_len, n_features = acts.shape
    eff_len = min(seq_len, t_len)

    for layer in range(n_layers):
      for feat_idx in range(n_features):
        vec = acts[layer, :eff_len, feat_idx].tolist()
        # Zero-pad up to seq_len if needed
        if eff_len < seq_len:
          vec = vec + [0.0] * (seq_len - eff_len)
        feature_to_values[(layer, feat_idx)] = vec

  else:
    raise RuntimeError(
      f"Unsupported activations type: {type(activations)}; expected mapping or torch.Tensor"
    )

  result: Dict[FeatureKey, TopActivation] = {}
  for (layer, feat_idx), values in feature_to_values.items():
    if not values:
      continue
    # Ignore features that are identically zero across the sequence
    max_val = max(values)
    if max_val == 0.0:
      continue
    max_index = max(range(len(values)), key=lambda i: values[i])
    result[(layer, feat_idx)] = TopActivation(
      tokens=str_tokens,
      values=values,
      max_value=float(max_val),
      max_index=int(max_index),
    )

  return result


# --------------------------
# Postgres helpers
# --------------------------


def get_pg_conn():
  """Create a psycopg2 connection from DATABASE_URL or individual env vars."""

  dsn = os.environ.get("DATABASE_URL")
  if not dsn:
    # fallback to individual PG* env vars
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD", "")
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    db = os.environ.get("PGDATABASE", "postgres")
    dsn = f"dbname={db} user={user} password={password} host={host} port={port}"
  return psycopg2.connect(dsn)


def upsert_neurons_and_activations(
  collector: TopKCollector,
  np_model_id: str,
  source_set: str,
  creator_id: str,
  data_source: str | None = None,
) -> None:
  """Write Neuron + Activation rows into Postgres.

  - np_model_id: should match `Model.id` and `graph.metadata.scan`
  - source_set: used to build Neuron.layer as f"{layer}-{source_set}"
  - creator_id: some existing User.id that is public (or your own id that you
    add to PUBLIC_ACTIVATIONS_USER_IDS)
  """

  conn = get_pg_conn()
  conn.autocommit = False
  cur = conn.cursor()

  try:
    # 1) Upsert Neuron rows (one per feature)
    neuron_rows = []
    for (layer, feat_idx), acts in collector.items():
      if not acts:
        continue
      # Approximate statistics
      max_act = max(a.max_value for a in acts)
      nonzero_fracs = []
      for a in acts:
        if a.values:
          nz = sum(1 for v in a.values if abs(v) > 1e-6)
          nonzero_fracs.append(nz / len(a.values))
      frac_nonzero = float(sum(nonzero_fracs) / len(nonzero_fracs)) if nonzero_fracs else 0.0

      layer_str = f"{layer}-{source_set}"
      index_str = str(feat_idx)

      neuron_rows.append(
        (
          np_model_id,
          layer_str,
          index_str,
          max_act,
          frac_nonzero,
        )
      )

    if neuron_rows:
      # Use ON CONFLICT to avoid duplicates
      execute_values(
        cur,
        """
        INSERT INTO "Neuron" ("modelId", "layer", "index", "maxActApprox", "frac_nonzero")
        VALUES %s
        ON CONFLICT ("modelId", "layer", "index") DO UPDATE
        SET "maxActApprox" = EXCLUDED."maxActApprox",
            "frac_nonzero" = EXCLUDED."frac_nonzero";
        """,
        neuron_rows,
      )

    # 2) Insert Activation rows (one per top activation instance)
    activation_rows = []
    for (layer, feat_idx), acts in collector.items():
      layer_str = f"{layer}-{source_set}"
      index_str = str(feat_idx)
      for a in acts:
        tokens = a.tokens
        values = a.values
        if not values:
          continue
        max_val = float(a.max_value)
        max_idx = int(a.max_index)
        min_val = float(min(values))

        activation_rows.append(
          (
            str(uuid.uuid4()),
            tokens,
            index_str,
            layer_str,
            np_model_id,
            data_source,
            max_val,
            max_idx,
            min_val,
            values,
            [],  # dfaValues
            None,  # dfaTargetIndex
            None,  # dfaMaxValue
            creator_id,
          )
        )
    print(f"[DEBUG] neurons: {len(neuron_rows)}, activations: {len(activation_rows)}")
    if activation_rows:
      execute_values(
        cur,
        """
        INSERT INTO "Activation" (
          "id",
          "tokens",
          "index",
          "layer",
          "modelId",
          "dataSource",
          "maxValue",
          "maxValueTokenIndex",
          "minValue",
          "values",
          "dfaValues",
          "dfaTargetIndex",
          "dfaMaxValue",
          "creatorId"
        )
        VALUES %s;
        """,
        activation_rows,
      )

    conn.commit()

    # After a successful commit, log approximate table sizes for monitoring.
    try:
      cur.execute(
        """
        SELECT
          pg_total_relation_size('"Neuron"'),
          pg_total_relation_size('"Activation"');
        """
      )
      neuron_size, activation_size = cur.fetchone()
      neuron_mb = neuron_size / (1024 * 1024)
      activation_mb = activation_size / (1024 * 1024)
      print(
        f"[PG] approx table sizes -> Neuron: {neuron_mb:.2f} MB, Activation: {activation_mb:.2f} MB"
      )
    except Exception as e:
      # This logging is best-effort only; don't fail the main job if it breaks.
      print(f"[WARN] failed to fetch table sizes: {e}")
  except Exception:
    conn.rollback()
    raise
  finally:
    cur.close()
    conn.close()


# --------------------------
# Main
# --------------------------


def main() -> None:
  text_path = os.environ.get("TEXT_PATH")
  if not text_path:
    raise SystemExit("TEXT_PATH env var must point to a UTF-8 text file (one sample per line)")

  np_model_id = os.environ.get("NP_MODEL_ID", "local-gemma-2b-toolformer")
  source_set = os.environ.get("SOURCE_SET", "toolformer-fused")
  top_k = int(os.environ.get("TOP_K", "50"))
  creator_id = os.environ.get("CREATOR_ID")
  if not creator_id:
    raise SystemExit("CREATOR_ID env var must be set to some existing User.id in your Postgres DB")

  data_source = os.environ.get("DATA_SOURCE")  # optional tag for where text came from

  # Optional: flush partial results to Postgres every N prompts to
  #  - bound memory usage of the in-memory TopKCollector, and
  #  - reduce work lost if the run is interrupted.
  # If 0 or unset, we only write once at the end (original behavior).
  flush_every_n = int(os.environ.get("FLUSH_EVERY_N_PROMPTS", "0"))

  # Optional: log in-memory stats (collector size + process RSS) every M prompts.
  # This does not touch Postgres and is cheap; default to 100.
  log_every_n = int(os.environ.get("LOG_EVERY_N_PROMPTS", "100"))

  print("==== collect_top_activations configuration ====")
  print(f"TEXT_PATH={text_path}")
  print(f"NP_MODEL_ID={np_model_id}, SOURCE_SET={source_set}")
  print(f"TOP_K={top_k}, FLUSH_EVERY_N_PROMPTS={flush_every_n}, LOG_EVERY_N_PROMPTS={log_every_n}")
  print(f"DATA_SOURCE={data_source}")
  print("==============================================")

  print(f"Loading model for NP modelId={np_model_id} ...")
  model = load_model()

  collector = TopKCollector(k=top_k)
  processed = 0

  for line_idx, text in iter_texts(text_path):
    print(f"[sample {line_idx}] processing text of length {len(text)}")
    feats = get_feature_activations(model, text)
    for (layer, feat_idx), act in feats.items():
      collector.add(layer, feat_idx, act)

    processed += 1

    # Periodic in-memory stats (no DB access).
    if log_every_n > 0 and processed % log_every_n == 0:
      num_features, num_acts = collector.stats()
      mem_mb = get_memory_usage_mb()
      print(
        f"[STATS] processed={processed}, collector_features={num_features}, "
        f"collector_top_activations={num_acts}, rss={mem_mb:.1f} MB"
      )

    # Periodically flush to Postgres to cap memory usage and avoid losing all
    # progress on interruption. Note this means you may end up with more than
    # top_k activations per feature overall (top_k per batch), which is usually
    # acceptable for inspection purposes.
    if flush_every_n > 0 and processed % flush_every_n == 0:
      num_features, num_acts = collector.stats()
      mem_mb = get_memory_usage_mb()
      print(
        f"[FLUSH] after {processed} prompts: features={num_features}, "
        f"top_activations={num_acts}, rss={mem_mb:.1f} MB -- flushing to Postgres"
      )
      upsert_neurons_and_activations(collector, np_model_id, source_set, creator_id, data_source=data_source)
      collector = TopKCollector(k=top_k)

  # Final flush for any remaining activations in the collector.
  num_features, num_acts = collector.stats()
  mem_mb = get_memory_usage_mb()
  print(
    f"[FINAL FLUSH] remaining features={num_features}, top_activations={num_acts}, "
    f"rss={mem_mb:.1f} MB -- writing into Postgres ..."
  )
  upsert_neurons_and_activations(collector, np_model_id, source_set, creator_id, data_source=data_source)
  print("Done.")


if __name__ == "__main__":
  main()
'''
cd /usr/project/xtmp/qc62
./local/pgsql/bin/pg_ctl -D local/pgdata -l local/pg.log start -o "-p 5433"

export TEXT_PATH=collect_top_activations_corpus.txt
export NP_MODEL_ID=local-gemma-2b-toolformer
export SOURCE_SET=toolformer-fused
export TOP_K=5
export CREATOR_ID=local-user-id
export DATABASE_URL=postgresql://qc62@localhost:5433/neuronpedia_local
export MODEL_ID=/home/users/qc62/MI/gemma_2b_toolformer_merged_v4
export TRANSCODER_SET=damaoo/gemma2b-toolformer-fused-transcoders
export OFFICIAL_TLENS_MODEL_ID=google/gemma-2-2b
export FLUSH_EVERY_N_PROMPTS=500
export LOG_EVERY_N_PROMPTS=100

python3 collect_top_activations.py
poetry run python collect_top_activations.py
'''