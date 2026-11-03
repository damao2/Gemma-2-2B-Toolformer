#!/bin/bash

#SBATCH --job-name=collect_top_activations # 作业名称，方便识别
#SBATCH --partition=compsci-gpu                   # 提交到的分区，请根据您的集群修改
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8                 # 每个任务使用的CPU核心数
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=3:00:00                    # 每个任务的预计运行时间
#SBATCH --output=slurm_logs/collect_top_activations_2.out # 标准输出日志
#SBATCH --error=slurm_logs/collect_top_activations_2.err  # 错误输出日志

cd /usr/project/xtmp/qc62
./local/pgsql/bin/pg_ctl -D local/pgdata -l local/pg.log start -o "-p 5433"

cd /usr/project/xtmp/qc62/neuronpedia/apps/graph
export TEXT_PATH=/usr/project/xtmp/qc62/neuronpedia/apps/graph/toolcall_activations_corpus_500.txt
export NP_MODEL_ID=local-gemma-2b-toolformer
export SOURCE_SET=toolformer-fused
export TOP_K=5
export CREATOR_ID=local-user-id
export DATABASE_URL=postgresql://qc62@localhost:5433/neuronpedia_local
export MODEL_ID=/home/users/qc62/MI/gemma_2b_toolformer_merged_v4
export TRANSCODER_SET=damaoo/gemma2b-toolformer-fused-transcoders
export OFFICIAL_TLENS_MODEL_ID=google/gemma-2-2b
export FLUSH_EVERY_N_PROMPTS=100
export LOG_EVERY_N_PROMPTS=10
export PYTHONUNBUFFERED=1

poetry run python collect_top_activations.py