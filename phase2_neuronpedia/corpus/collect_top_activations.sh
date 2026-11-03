

cd ..path/to/database
./local/pgsql/bin/pg_ctl -D local/pgdata -l local/pg.log start -o "-p 5433"

cd ../neuronpedia/apps/graph
export TEXT_PATH=../neuronpedia/apps/graph/toolcall_activations_corpus_500.txt
export NP_MODEL_ID=local-gemma-2b-toolformer
export SOURCE_SET=toolformer-fused
export TOP_K=5
export CREATOR_ID=local-user-id
export DATABASE_URL=postgresql://qc62@localhost:5433/neuronpedia_local
export MODEL_ID=path/to/gemma_2b_toolformer_merged_v4
export TRANSCODER_SET=damaoo/gemma2b-toolformer-fused-transcoders
export OFFICIAL_TLENS_MODEL_ID=google/gemma-2-2b
export FLUSH_EVERY_N_PROMPTS=100
export LOG_EVERY_N_PROMPTS=10
export PYTHONUNBUFFERED=1

poetry run python collect_top_activations.py