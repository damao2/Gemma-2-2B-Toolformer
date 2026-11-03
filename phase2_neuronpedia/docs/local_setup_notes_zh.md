> 本地自建 Neuronpedia 的启动与排错笔记（集群路径和主机名为当时环境，仅供参考）。

poetry run python start.py \
  --model_id /Volumes/Untitled/Downloads/gemma_2b_toolformer_merged_v4 \
  --transcoder_set /Volumes/Untitled/Downloads/fused_delta_32_pctl_0.5 \
  --model_dtype float32 \
  --port 5004

cd ./neuronpedia/apps/graph
poetry run python start.py \
  --model_id /home/users/qc62/MI/gemma_2b_toolformer_merged_v4 \
  --transcoder_set damaoo/gemma2b-toolformer-fused-transcoders \
  --model_dtype float32 \
  --port 5004

export PYTORCH_ENABLE_MPS_FALLBACK=1

curl -sS -X POST http://localhost:5004/generate-graph \
  -H "x-secret-key: MY_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Calculate 3+5.",                  
    "model_id": "/home/users/qc62/MI/gemma_2b_toolformer_merged_v4", 
    "slug_identifier": "local-gemma2b-demo",
    "batch_size": 24,
    "max_n_logits": 10,
    "desired_logit_prob": 0.95,
    "node_threshold": 0.8,
    "edge_threshold": 0.98,
    "max_feature_nodes": 5000
  }' > graph_delta_32_pctl_0.5.json

cd /usr/project/xtmp/qc62/neuronpedia/apps/webapp

# 第一次需要安装依赖
npm install          # 或 yarn install

# 使用 demo 环境运行（也可以直接用 .env.local 手动加载）
npm run dev:demo     # 它会用 .env.demo，如果你想强制用 .env.local 则直接 npm run dev
# 或者如果你直接想用 .env.local:
npm run dev -- --port 3000
npx env-cmd -f .env.localdb --use-shell "prisma generate && next dev"

Postgres 数据库本身（pgsql + pgdata）已经建好并持久化在磁盘上，重连计算节点不会丢。
你只需要在「新开的 shell 上」重新启动数据库服务一次，然后照常启动 webapp。
# 1. 启动本地 Postgres（只要这一条）
cd /usr/project/xtmp/qc62
./local/pgsql/bin/pg_ctl -D local/pgdata -l local/pg.log start -o "-p 5433"

./local/pgsql/bin/pg_ctl -D local/pgdata stop
# 2. 启动 webapp（照旧用 .env.localdb）
cd neuronpedia/apps/webapp
npx env-cmd -f .env.localdb --use-shell "prisma generate && next dev"

local$ ssh -L 3000:compsci-cluster-fitz-07:3000 \
    -L 5004:compsci-cluster-fitz-07:5004 \
    qc62@login.cs.duke.edu

collect_top_activation
跑完后，打开 Postgres 查一下某个 feature：
psql postgresql://qc62@localhost:5433/neuronpedia_local

-- 看几个用户
SELECT "id","email","name"
FROM "User"
LIMIT 10;
SELECT "modelId","layer","index","maxActApprox","frac_nonzero"
FROM "Neuron"
WHERE "modelId" = 'local-gemma-2b-toolformer'
LIMIT 5;

SELECT "tokens","values","maxValue","maxValueTokenIndex"
FROM "Activation"
WHERE "modelId" = 'local-gemma-2b-toolformer'
LIMIT 5;
然后在前端 /api/features 或某个 feature 页面看右侧面板是否能拉到你刚写入的 activations。
如果你愿意，我可以帮你再对照一下前端 /api/features 的请求体（特别是 layer 和 index 的编码方式），确认和现在脚本里的 layer_str = f"{layer}-{SOURCE_SET}" / index = str(feature_idx) 100% 对齐。

# circuit eval 的用法：
启动 Postgres（你给的命令就对）并确认端口：
cd /usr/project/xtmp/qc62
./local/pgsql/bin/pg_ctl -D local/pgdata -l pg.log start -o "-p 5433"
快速确认：./local/pgsql/bin/pg_isready -h localhost -p 5433

启动 graph server（端口必须是 5004，因为 webapp 的 localhost graph host 是写死 127.0.0.1:5004）
cd /usr/project/xtmp/qc62/neuronpedia/apps/graph
poetry run python start.py --model_id /home/users/qc62/MI/gemma_2b_toolformer_merged_v4 --transcoder_set damaoo/gemma2b-toolformer-fused-transcoders --model_dtype float32 --port 5004

鉴权要点：graph server 读的是 SECRET，webapp 传的是 GRAPH_SERVER_SECRET（你 .env.localdb 和 graph 的 .env 都是 MY_SECRET，这块已对齐）。如果不对齐会看到 x-secret-key header missing 或 Invalid x-secret-key。
启动 webapp（你给的命令 OK）
cd /usr/project/xtmp/qc62/neuronpedia/apps/webapp
npx env-cmd -f .env.localdb --use-shell "prisma generate && next dev"

如果第一次跑 DB schema 报错（缺表/缺列）：先跑一次 npx env-cmd -f .env.localdb prisma db push

跑 batch eval（从 Postgres 自动取你前端保存的 subgraph）
脚本在这里：batch-eval-subgraph.ts
命令（推荐带上 sourceSetName，避免依赖 model 默认 source set）：
cd /usr/project/xtmp/qc62/neuronpedia/apps/webapp
npx env-cmd -f .env.localdb --use-shell "npx tsx scripts/batch-eval-subgraph.ts --baseUrl http://localhost:3000 --modelId local-gemma-2b-toolformer --subgraphName test1 --prompts scripts/prompts_circuit_eval_v3.jsonl --out scripts/out/circuit_eval_v3.json --sourceSetName toolformer-fused"
说明：我已经把 --sourceSlug 做成“可选”。你不传时会按 (modelId, displayName) 自动挑最近更新的那条 subgraph，并自动用它所属的源 graph 来建模板（即你只要在前端保存 subgraph 就能继续实验）。

最常见的定位入口
生成图失败：先看 webapp 终端里 /api/graph/generate 的报错；再看 graph server 终端（通常是鉴权/模型推理/转码器加载问题）。
找不到 subgraph：确认你在前端保存时的 displayName 精确是 test1（大小写敏感）；或者直接不传 --sourceSlug 让它自动选最新那条。

进行prompts分类筛选，构建验证集
python3 scripts/gen_circuit_eval_prompts.py --out scripts/prompts_circuit_eval_v3.jsonl --seed 0

npx env-cmd -f .env.localdb --use-shell "npx tsx scripts/screen_circuit_eval_prompts.ts \
  --in scripts/prompts_circuit_eval_v3.jsonl \
  --outStem scripts/out/prompts_circuit_eval_v3 \
  --modelId local-gemma-2b-toolformer \
  --sourceSetName toolformer-fused \
  --maxNLogits 15 \
  --desiredLogitProb 0.99 \
  --boundaryThreshold 0.10"