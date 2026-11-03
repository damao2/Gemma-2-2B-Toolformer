# batch-eval-subgraph.ts 逻辑详解

本文解释 `scripts/batch-eval-subgraph.ts` 这个脚本在做什么、如何把一个人工选出的 circuit 用到一批新的 eval prompts 上，以及它最终产出的各项指标分别代表什么。

脚本的核心目的不是“看新 prompt 里有没有出现同一批 node id”，而是更接近下面这个问题：

> 在一个源 prompt 上由人工 pin 出来的 circuit，如果迁移到其他 prompts 上，是否还能解释新图中的关键输出行为？

这里的“解释”不是靠肉眼看节点重不重合，而是通过一套基于 attribution graph 的打分函数来衡量。

## 一句话概括整个流程

1. 从数据库里找出某个源 graph 上保存过的一个 subgraph，也就是人工选中的 circuit。
2. 从这个源 graph 中提取出一个“相对模板”：保留每个被选中特征的 `(layer, feature)`，以及它相对于目标 logit 的位置偏移。
3. 对每个 eval prompt 重新生成一张 attribution graph。
4. 把源 circuit 模板映射到新 graph 上，得到这张新图里“对应的” pinned feature 集合。
5. 从新 graph 的 logit 节点中提取一个行为摘要，刻画它更像是朝着 tool-call 还是 echo-first 方向走。
6. 对新 graph 计算三组分数：
   - 默认 logit 概率加权下的 `replacement_full` / `completeness_full` 与 `replacement_subgraph` / `completeness_subgraph`
   - 只盯着 gate token `<` 的 `*_gate` 分数
   - 只盯着 `P("<") - P(echoFirst)` 这个 signed margin 的 `*_signed_margin` 分数
7. 再输出一组“映射质量”指标，例如 `matched`、`exactCtxMatched`、`coverage`，以及行为摘要字段。

最终，这个脚本回答的是：

> 这个在源 prompt 上人工选出的 circuit，在别的 prompts 上还能不能找到对应结构？找到了多少？而且找到之后，它对新 prompt 的默认输出、tool-call gate、以及 tool-call 与 echo-first 的区分能力，还剩多少？

## 1. 输入是什么

脚本从命令行读取参数，关键参数包括：

- `--baseUrl`：本地 webapp 地址，脚本通过它调用 `/api/graph/generate`
- `--modelId`：生成 graph 用的模型
- `--subgraphName`：人工保存的 circuit 名称
- `--prompts`：eval prompt 文件
- `--out`：输出结果 JSON 路径
- `--sourceSlug`：可选，指定源 graph；不传时会找这个 `subgraphName` 的最新版本

还可以传一组图生成超参数：

- `--sourceSetName`
- `--maxNLogits`
- `--desiredLogitProb`
- `--nodeThreshold`
- `--edgeThreshold`
- `--maxFeatureNodes`
- `--maxRetries`
- `--initialBackoffMs`

这些参数不会直接改变打分公式，但会改变“生成出来的 graph 长什么样”，所以会间接影响最终指标。

## 2. eval prompts 是怎么读的

脚本允许三种输入形式：

1. 严格 JSONL，对象里至少有 `prompt`
2. JSONL，但每行本身是一个字符串
3. 纯文本，每行一个 prompt

如果一行不是合法 JSON，就会被当作纯文本 prompt。

像 `scripts/prompts_circuit_eval_v2.jsonl` 这种文件里，每行通常长这样：

```json
{"label": "pos_seen_arith_div", "prompt": "I'm stuck on this calculation: 865 / 395.\n", "meta": {"family": "pos_seen", "expectedMode": "tool_call"}}
```

其中：

- `prompt` 是实际送去生成 graph 的文本
- `label` 只用于结果分组和分析
- `meta` 现在会参与 `expectedBehavior` 的推断，但仍然不会原样写回输出结果

### `meta` 如何参与行为判定

新版脚本会用 `meta` 来推断该 prompt 的期望行为 `expectedBehavior`。

优先级是：

1. `meta.expectedBehavior`
2. `meta.expectedMode`
3. `label`

具体规则是：

- 如果 `expectedBehavior` 已经明确是 `tool_call` 或 `echo_first`，直接使用
- 否则如果 `expectedMode === 'tool_call'`，则期望行为是 `tool_call`
- 否则如果 `expectedMode === 'echo'`，则期望行为是 `echo_first`
- 否则退化为：`label` 以 `pos_` 开头就当作 `tool_call`，其余当作 `echo_first`

这意味着新版脚本不再只关心“graph 解释力”，也会显式跟踪：

> 这个 prompt 本来期待模型朝 tool-call 还是 echo-first 方向走？

### 自动补换行

脚本会对每个 prompt 调用 `ensureTrailingNewline`：如果 prompt 末尾没有换行，就补一个 `\n`。

原因很直接：很多 chat/instruction 模型对末尾换行比较敏感，作者希望 batch eval 时输入格式尽量统一。结果里会记录 `promptNewlineAdded`，方便排查输入格式导致的差异。

## 3. “人工选中的 circuit” 到底是什么

这个脚本里的 circuit，不是单独保存在某个 JSON 文件里，而是保存在数据库表 `GraphMetadataSubgraph` 里。

这个表里最关键的字段是：

- `graphMetadataId`：它属于哪张 graph
- `displayName`：用户给 subgraph 起的名字
- `pinnedIds`：用户在图界面里 pin 下来的节点 ID 列表

所以，这里的“人工选中的 circuit”其实就是：

> 某张源 graph 上，被人手工 pin 住的一组节点。

脚本会先确定源 graph：

- 如果传了 `--sourceSlug`，就精确找 `modelId + sourceSlug`
- 否则就找这个 `subgraphName` 在该模型下最新的一次保存

然后再加载对应 subgraph 的 `pinnedIds`。

## 4. 源 graph 是怎么加载的

脚本不是重新生成源 graph，而是从已有 graph JSON 读取。

流程是：

1. 从 `GraphMetadata` 拿到该 graph 的 `url`
2. 如果 `url` 以 `/` 开头，就从 `public/` 下的本地文件读取
3. 否则退回到 `public/local-graphs/${modelId}-${slug}.json`

这意味着：

- 源 circuit 是固定在某个已经生成好的 graph 上的
- batch eval 时不会重新计算源 graph，只会重用它

## 5. 脚本怎么把源 circuit 抽象成一个“模板”

这是整个脚本最关键的部分。

### 5.1 先找源 graph 的目标 logit

函数 `getTargetLogitNode` 会在 graph 的 logit 节点里：

1. 优先找 `is_target_logit === true` 的节点
2. 如果没有，就退化为选择 `token_prob` 最大的 logit

因此，后续所有位置对齐都是围绕“这个 graph 的目标输出 token”展开的。

### 5.2 从 clerp 里抽目标 token

函数 `extractTargetTokenFromClerp` 会从类似下面的字符串里取 token：

```text
Output "<" (p=0.997)
```

它只做一个正则匹配，目的是把这个 graph 当前在追踪哪个 token 记下来，例如结果文件里的 `sourceTargetToken` 或每个样本的 `targetToken`。

### 5.3 真正进入模板构造

函数 `buildTemplateFromSourceGraph` 会遍历 `pinnedIds`，但只保留满足下面条件的节点：

- 节点存在于源 graph 中
- `feature_type === 'cross layer transcoder'`
- `feature` 是数值

也就是说，这个脚本定义的“circuit”只由 CLT feature 节点构成，不把 embedding、logit、error node 当作模板主体。

对每个被保留的 pinned feature，会记录一个三元组：

- `layer`
- `feature`
- `ctxDeltaFromLogit = sourceLogitCtxIdx - node.ctx_idx`

这个 `ctxDeltaFromLogit` 很重要。它表示的不是绝对 token 位置，而是：

> 该 feature 相对于目标 logit 所在位置，往前或往后偏移了多少。

例如：

- 如果源目标 logit 的 `ctx_idx = 12`
- 某个 feature 在 `ctx_idx = 10`

那么 `ctxDeltaFromLogit = 2`。

这意味着模板记住的不是“它在第 10 个 token 上”，而是“它位于目标 logit 左边两个位置”。

### 5.4 去重

模板最后按下面这个键去重：

```text
layer | feature | ctxDeltaFromLogit
```

所以模板本质上是一个“相对位置版”的 feature 列表，而不是源 graph 上具体 node id 的拷贝。

## 6. eval prompt 上的新 graph 是怎么生成的

对每个 eval prompt，脚本都会调用：

```text
POST {baseUrl}/api/graph/generate
```

请求体里包含：

- `prompt`
- `modelId`
- `sourceSetName`（可选）
- `maxNLogits`
- `desiredLogitProb`
- `nodeThreshold`
- `edgeThreshold`
- `maxFeatureNodes`

### 6.1 `/api/graph/generate` 实际做了什么

这个 API route 又会做几件事：

1. 校验参数是否合法
2. 用 `getGraphTokenize(...)` 先做一次 tokenization 和 salient logit 选择，并检查 prompt 不超过 64 tokens
3. 如果没有提供 slug，就自动生成一个
4. 调用外部 graph server 的 `generate-graph` 接口
5. 把返回的 graph JSON 存到本地 `public/local-graphs/`
6. 把 graph metadata 写回数据库
7. 最终把整张 graph 一并返回给脚本

新版脚本会额外依赖 graph metadata 中的：

- `prompt`
- `prompt_tokens`

因为它要从 `prompt_tokens` 里恢复 `echoFirstToken`，用于后续行为摘要和 signed-margin 分数。

### 6.2 这些生成参数的作用

脚本本身不实现 graph tracing；它只是把参数传给 graph generator。

可以安全地理解为：

- `maxNLogits` 和 `desiredLogitProb` 控制“追踪哪些输出 logits”
- `nodeThreshold` 和 `edgeThreshold` 控制图的剪枝强度
- `maxFeatureNodes` 控制图大小上限

因此本脚本评估的其实不是“完整模型上的原始机制”，而是：

> 在这些生成与剪枝超参数之下，得到的 attribution graph 里，这个 circuit 还能解释多少。

### 6.3 重试逻辑

如果 `/api/graph/generate` 返回 `RUNPOD_BUSY` 或 HTTP 503，脚本会指数退避重试。

所以 batch 运行时，单个 prompt 的失败不一定是模型问题，也可能只是生成服务暂时忙。

## 7. 怎么把源 circuit 映射到新 graph 上

这是另一个核心问题。

脚本并不会要求新 graph 里出现和源 graph 完全相同的 node id，因为不同 prompt 生成出来的 node id 通常不一样。

它做的是“模板对齐”。

### 7.1 先找新 graph 的目标 logit

和源 graph 一样，新 graph 也先通过 `getTargetLogitNode` 找目标 logit：

- 优先 `is_target_logit`
- 否则退化为 `token_prob` 最大的 logit

这一步很关键，因为后续位置都是相对这个新 logit 计算。

### 7.2 建一个 `(layer, feature)` 索引

脚本遍历新 graph 的所有 CLT feature 节点，按：

```text
key = layer | feature
```

分桶，然后把每个桶里的候选节点按 `ctx_idx` 排序。

### 7.3 用“相对于 logit 的偏移”来找最接近节点

对于模板中的每一项：

- 模板给出 `layer`
- 模板给出 `feature`
- 模板给出 `ctxDeltaFromLogit`

在新 graph 里，脚本先算出这个模板理想上应该落在什么位置：

$$
desiredCtxIdx = targetLogitCtxIdx - ctxDeltaFromLogit
$$

然后在相同 `(layer, feature)` 的候选节点里，选 `ctx_idx` 最接近 `desiredCtxIdx` 的那个。

这意味着这个脚本在假设：

> 如果一个 circuit 真有可迁移性，那么它在新 prompt 上至少应该还能表现为“同一层、同一 feature，并且位于相对目标输出差不多的位置”。

### 7.4 三个映射质量指标

映射完成后，会得到：

- `templateCount`：模板总条目数
- `matched`：模板里有多少项在新 graph 中至少找到了同 `(layer, feature)` 的候选节点
- `exactCtxMatched`：其中有多少项不仅 feature 对上了，而且 `ctx_idx` 也正好等于理想位置
- `coverage = matched / templateCount`

这三个指标的含义分别是：

#### `templateCount`

源 circuit 被抽象后有多少个独立模板项。它近似代表“这个电路有多大”。

#### `matched`

说明新 graph 中能找到多少“同 feature identity”的候选。它回答的是：

> 源 circuit 里的 feature，在新 prompt 上还出没出现在图里？

#### `exactCtxMatched`

这是更严格的版本。它回答的是：

> 不仅 feature 还在，而且它是否仍然出现在和目标输出同样的相对位置上？

#### `coverage`

就是模板覆盖率。越高说明越多模板项在新 graph 里有对应 feature；越低说明大量 feature 在新图里根本找不到。

注意：`matched` 高并不等于“这个电路真的解释了新 prompt”。它只说明模板可以映射。真正的解释能力还要看后面的 replacement/completeness 分数。

## 8. 新版脚本新增了什么

和旧版相比，新版脚本增加了两层内容：

1. 它会从生成出来的 graph 中重新总结“行为方向”，即这张图更像是在支持 tool-call，还是更像在支持 echo-first。
2. 它不再只算默认分数，还会额外算 gate token `<` 分数，以及 signed margin 分数。

这使得脚本不只是回答“这个 subgraph 一般性地解释图到什么程度”，还会回答：

> 这个 subgraph 是否特别在解释 tool-call gate？
>
> 它是否特别在解释 `<` 相对于 echo-first 备选输出的优势？

## 9. 行为摘要是怎么提取的

新版脚本对每个生成好的 graph 都会执行 `summarizeBehaviorFromGraph(...)`。

### 9.1 先从 graph 本身恢复 logit token

脚本会遍历所有 `feature_type === 'logit'` 的节点，并优先从：

- `node.logitToken`
- 否则 `node.clerp`

中提取每个 logit 对应的 token 文本。

然后按 `token_prob` 从高到低排序，得到：

- `topToken`
- `topProb`
- `topNonAngleToken`
- `topNonAngleProb`

这里的 `topToken` 表示：

> 在这张 graph 里被追踪到的 logit 节点中，概率最高的那个 token。

它和 `targetToken` 很接近，但不完全是同一个概念：

- `targetToken` 是 `getTargetLogitNode(...)` 选中的“目标 logit”
- `topToken` 是把图中现有 logit 节点按 `token_prob` 排序后拿到的 top-1

在大多数情况下二者一致，但脚本语义上把这两件事区分开了。

### 9.2 `echoFirstToken` 怎么来的

脚本不再像 screening 脚本那样用 forward-pass 返回值，而是直接使用 graph metadata 中的 `prompt_tokens`。

它会跳过 special tokens，例如：

- `<bos>`
- `<eos>`
- `<pad>`
- `<unk>`
- `<start_of_turn>`
- `<end_of_turn>`

然后取第一个非 special token，记为 `echoFirstToken`。

它的含义是：

> 如果模型沿着 prompt 本身继续自然文本，这个 prompt 最有代表性的“echo-first 竞争目标”是什么？

### 9.3 `probMargin` 和 `signedProbMargin`

脚本随后会计算：

$$
probMargin = P("<") - P(echoFirst)
$$

这里的 `<` 被硬编码成 gate token，也就是 tool-call 起手符号。

所以：

- `probMargin > 0`：图中的 logit 更偏向 tool-call 起手
- `probMargin < 0`：图中的 logit 更偏向 echo-first

接着，脚本再结合 prompt 的 `expectedBehavior`，定义：

$$
signedProbMargin = \begin{cases}
probMargin & \text{if expectedBehavior = tool_call} \\
-probMargin & \text{if expectedBehavior = echo_first}
\end{cases}
$$

这意味着：

- `signedProbMargin` 越大，越朝着该 prompt 的预期行为方向走
- `signedProbMargin` 越小甚至为负，越说明图中的 logit 走反了方向

### 9.4 `decisionByTop1`

脚本还会给出一个非常硬的 top-1 判定：

- 如果 `topToken === '<'`，则 `decisionByTop1 = tool_call`
- 否则如果 `topToken === echoFirstToken`，则 `decisionByTop1 = echo_first`
- 否则为 `other`

这不是一个复杂分类器，而只是一个便于离线分析的粗判定。

## 10. 这个脚本现在到底算了哪些分数

对每个新 graph，脚本现在会计算三组分数，而且每组都区分 full graph 与 mapped subgraph。

### 10.1 默认分数

这是原来就有的那一组：

1. `full = computeGraphScoresFromGraphData(graph, [])`
2. `sub = computeGraphScoresFromGraphData(graph, mapped.pinnedIds)`

输出字段是：

- `replacement_full`
- `completeness_full`
- `replacement_subgraph`
- `completeness_subgraph`

它们的 logit 权重就是 graph 中各 logit 节点自己的 `token_prob`。

### 10.2 gate token 分数

脚本还会尝试只针对 gate token `<` 重新打分。

做法是调用：

- `computeGraphScoresFromGraphDataWithLogitWeightOverride(graph, [], ...)`
- `computeGraphScoresFromGraphDataWithLogitWeightOverride(graph, mapped.pinnedIds, ...)`

并使用下面的 logit 权重函数：

- 对 token 是 `<` 的 logit 节点赋权重 1
- 其它 logit 节点赋权重 0

因此这组分数回答的是：

> 如果我只关心 tool-call gate token `<` 这一条输出方向，那么整张图/这个 subgraph 的解释能力有多强？

输出字段是：

- `replacement_full_gate`
- `completeness_full_gate`
- `replacement_subgraph_gate`
- `completeness_subgraph_gate`

这组分数只有在图里真的包含 `<` 这个 logit 时才有意义；否则脚本会返回 `null`。

### 10.3 signed margin 分数

脚本还会针对 `<` 和 `echoFirstToken` 之间的 signed margin 计算分数。

它给 logit 节点的权重规则是：

- `<` 对应的 logit 权重为 `+1`
- `echoFirstToken` 对应的 logit 权重为 `-1`
- 其它 logit 权重为 `0`

然后调用：

- `computeGraphScoresFromGraphDataWithSignedLogitWeightOverride(...)`

这和普通 override 版本的区别在于：

> 它会把 influence 和 output influence 在聚合时取绝对值。

这样做的原因是：

> 对 signed margin 来说，支持 `<` 和抑制 `echoFirst` 都是在帮助模型朝正确方向走；二者都应计入“有用解释”，而不是在总量里互相抵消。

输出字段是：

- `replacement_full_signed_margin`
- `completeness_full_signed_margin`
- `replacement_subgraph_signed_margin`
- `completeness_subgraph_signed_margin`

只有在图里同时包含：

- `<`
- `echoFirstToken`

这两个目标 logit 时，这组分数才会有意义；否则脚本写出 `null`。

## 11. 分数计算背后的图论逻辑

### 11.1 节点类型顺序

打分函数先把节点重排为固定顺序：

1. `cross layer transcoder`
2. `mlp reconstruction error`
3. `embedding`
4. `logit`

这样后面就能按区间切出：

- feature nodes
- error nodes
- token embedding nodes
- logit nodes

### 11.2 邻接矩阵怎么建

对每条边：

```text
adjacencyMatrix[dst][src] = edge.weight
```

所以矩阵的每一行代表某个节点“从哪些上游节点接收输入”。

### 11.3 为什么要按绝对值归一化

`normalizeMatrix` 会对每一行做：

1. 先取绝对值
2. 再除以这一行绝对值总和

于是每个节点的输入边会被转成一个分布，表示：

> 这个节点的输入中，有多少比例来自各个上游来源。

这一步会丢掉符号信息，只保留“影响份额”。所以这些分数更像解释比例，而不是净正向贡献。

### 11.4 influence 是怎么从 logit 反传回去的

`computeInfluence(A, logitWeights)` 的思路是从 logit 开始，把影响沿着归一化后的图一层层向上游传播。

其中：

- `logitWeights` 只放在 logit 节点上
- 权重值就是每个 logit 节点的 `token_prob`

换句话说，多个输出 logit 会按各自概率加权，一起决定这张图里哪些上游节点更重要。

迭代过程可以理解为：

$$
I^{(1)} = wA
$$

$$
I^{(2)} = I^{(1)}A
$$

$$
I^{(3)} = I^{(2)}A
$$

最后把所有轮次的影响加总，得到每个节点对输出的总影响强度。

直观上：

- 如果某个 token / feature 处在很多重要路径上，它的 influence 会更大
- 如果路径大多经过 error node，那么 error influence 会更大

## 12. subgraph 分数和 full graph 分数的关键差异

这是最容易误解的一点。

### 12.1 full graph

当 `pinnedIds = []` 时，函数直接对整张 graph 打分。

这反映的是：

> 这张生成出来的 attribution graph 整体上解释输出解释得好不好。

### 12.2 subgraph

当传入 `mapped.pinnedIds` 时，函数并不是简单“裁掉其余节点”，而是：

> 把所有不在 subgraph 里的 CLT feature，当作 unexplained error 来处理。

具体实现是：

1. 找到每个 `(layer, ctx_idx)` 对应的 `mlp reconstruction error` 节点
2. 遍历所有 CLT feature
3. 如果这个 feature 不在 pinned set 里：
   - 把它对下游节点的边权并入同位置的 error node
   - 然后把这个 feature 的行和列清零

如果某个 feature 所在位置根本没有对应 error node，就直接把这个 feature 清零。

这套语义很关键，因为它表示：

> subgraph 没覆盖到的 feature，不被当成“无所谓的背景”，而被当成“没有被当前 circuit 解释掉的剩余误差”。

所以 `replacement_subgraph` / `completeness_subgraph` 实际上是在问：

> 如果只允许这组映射后的 feature 作为解释主体，那么新 graph 里还有多少解释能力可以保住？

## 13. Replacement Score 的含义

代码里先把 influence 按节点类型切开：

- `tokenInfluence`：落在 embedding/token 节点上的总 influence
- `errorInfluence`：落在 error node 上的总 influence

然后计算：

$$
replacementScore = \frac{tokenInfluence}{tokenInfluence + errorInfluence}
$$

它的直观含义是：

> 从输出 logit 反向追溯回输入时，最终有多少“端到端解释份额”真正落在输入 token 上，而不是落在 error node 上。

因此：

- 分数高：说明输出大部分可以通过 feature 路径一直解释回输入 token
- 分数低：说明很多影响最终掉进 error node，也就是还有较大一部分机制没有被 feature-based circuit 解释掉

### 为什么说它是“严格”的

因为它只看最终端点：

- 到了 token，就算解释成功
- 到了 error，就算解释失败

中间某个节点“部分被解释”并不会得到很多额外奖励。所以这是一个偏严格的指标。

## 14. Completeness Score 的含义

对每个节点，代码会先算一个：

$$
nonErrorFraction_i = 1 - \sum_{e \in error} A_{i,e}
$$

也就是该节点所有输入里，有多大比例不是来自 error nodes。

然后再用每个节点对输出的重要性做加权平均：

$$
outputInfluence_i = nodeInfluence_i + logitWeight_i
$$

$$
completenessScore = \frac{\sum_i nonErrorFraction_i \cdot outputInfluence_i}{\sum_i outputInfluence_i}
$$

它的直观含义是：

> 对所有会影响输出的重要节点来说，它们自身又有多大程度是被 feature/token 输入解释的，而不是依赖 error 输入。

因此：

- 分数高：说明图中重要节点的大部分输入都来自可解释部分
- 分数低：说明很多重要节点的输入仍然严重依赖 error

### 为什么它比 replacement 更“宽松”

因为它给“部分解释”记分。

举例说，一个重要节点如果：

- 80% 输入来自 feature/token
- 20% 输入来自 error

那么它在 completeness 里仍然拿到 0.8 的局部得分；但在 replacement 的端到端意义上，这部分 error 可能还是会显著拉低整体表现。

所以通常可以把二者理解为：

- `replacement`：严格看最终有没有把解释真正接回输入 token
- `completeness`：宽松看图中重要部分有多大比例已经被 feature-based explanation 覆盖

## 15. 新版脚本如何用这些指标评估 generalizability

如果你把这个脚本放回它的研究目标里看，它其实是在同时测三件事：

### 15.1 结构可迁移性

由下面这些指标反映：

- `matched`
- `exactCtxMatched`
- `coverage`

它们回答的是：

> 源 circuit 的 feature 身份和相对位置，在新 prompt 上还能不能找到？

### 15.2 新 prompt 本身是否形成了“好图”

由下面这些指标反映：

- `replacement_full`
- `completeness_full`

它们回答的是：

> 不考虑你挑的 circuit，只看新 prompt 生成出来的整张 attribution graph，本身解释输出解释得好不好？

这一步很重要，因为如果 full graph 本来就很差，那么 subgraph 差也不一定是 circuit 没有泛化。

### 15.3 你挑的 circuit 能保留多少解释能力

由下面这些指标反映：

- `replacement_subgraph`
- `completeness_subgraph`

它们回答的是：

> 把新 graph 中不属于这个 mapped circuit 的 feature 都视作 error 后，这个 circuit 单独还能解释多少？

如果一个 circuit 真有 generalizability，通常你会希望看到：

- `coverage` 不低
- `replacement_subgraph` 和 `completeness_subgraph` 不至于比 full 掉得太夸张

### 15.4 它是否保住了 tool-call gate

新版脚本新增的 `*_gate` 分数在回答：

> 这个 subgraph 是否特别还能解释 `<` 这个 gate token？

这对于区分下面两种情况特别有用：

- subgraph 一般性解释力下降
- subgraph 对整体图解释力一般，但对 tool-call gate 仍然高度专门化

### 15.5 它是否保住了“向正确行为方向偏移”的能力

新版脚本新增的 `*_signed_margin` 分数在回答：

> 这个 subgraph 是否还能解释 `<` 相对于 echo-first 备选的优势？

这比只看 `*_gate` 更细，因为它不只是看 `<` 本身，还看它是否压过了最自然的文本延续方向。

反过来说，下面这些现象都可能意味着泛化较差：

- `coverage` 很低：新 prompt 根本找不到大部分 feature
- `coverage` 还行，但 `replacement_subgraph` 很低：feature 虽然还在，但已经不是主解释路径
- `targetToken` 改掉了：新图追踪的目标输出 token 已经不是源任务里的那个 token
- `replacement_subgraph_gate` 很低：subgraph 还在，但对 `<` 这一 gate token 已经没有解释力
- `replacement_subgraph_signed_margin` 很低：subgraph 还在，但已经不再支持“tool-call 胜过 echo-first”的方向性差异

## 16. 输出文件里每个字段是什么意思

输出会写两个文件：

- 一个 JSON
- 一个同名 CSV

### 16.1 JSON 的 `meta`

- `modelId`：评估所用模型
- `sourceSlug`：源 graph 的 slug
- `subgraphName`：使用的 subgraph 名称
- `subgraphId`：数据库里的 subgraph id
- `sourceTargetToken`：源 graph 的目标 token
- `gateToken`：当前脚本固定使用的 gate token，默认是 `<`
- `templateSize`：模板大小，也就是 `template.length`
- `generatedAt`：输出生成时间
- `scoringSemantics`：提醒你 subgraph 打分采用的是“把未选 feature 合并进 error nodes”的语义
- `behaviorSemantics`：提醒你 `probMargin` / `signedProbMargin` 的定义，以及 signed-margin 分数要求两个目标 logit 都在图里出现

### 16.2 每条结果的字段

- `index`：prompt 序号
- `label`：输入 JSONL 中的标签
- `promptNewlineAdded`：是否自动补了换行
- `slug`：新 graph 的 slug
- `url`：新 graph 的查看链接
- `targetToken`：新 graph 实际追踪的目标 token
- `targetProb`：该目标 token 的概率
- `topToken` / `topProb`：图中现有 logit 节点里概率最高的 token 及其概率
- `gateToken`：当前比较的 gate token，固定为 `<`
- `echoFirstToken`：从 `prompt_tokens` 中恢复出的第一个非 special token
- `expectedBehavior`：这个 prompt 预期应该朝 `tool_call` 还是 `echo_first` 走
- `decisionByTop1`：根据 `topToken` 做的粗 top-1 行为判定
- `probAngleBracket` / `rankAngleBracket`：`<` 在图中 logit 集合里的概率和 rank
- `probEchoFirst` / `rankEchoFirst`：`echoFirstToken` 在图中 logit 集合里的概率和 rank
- `probMargin`：`P("<") - P(echoFirst)`
- `signedProbMargin`：按 `expectedBehavior` 翻转后的 margin
- `topNonAngleToken` / `topNonAngleProb`：除 `<` 之外最高概率的 token 及其概率
- `salientLogitCount`：图中被追踪到的 logit 数量
- `salientCumulativeProbability`：这些 logit 概率的总和
- `hasAngleBracketTarget`：图里是否真的出现了 `<`
- `hasEchoFirstTarget`：图里是否真的出现了 `echoFirstToken`
- `hasSignedMarginTarget`：图里是否同时出现了 `<` 和 `echoFirstToken`
- `replacement_full_gate` / `completeness_full_gate`：只针对 `<` 这一 gate token 的 full-graph 分数
- `replacement_subgraph_gate` / `completeness_subgraph_gate`：只针对 `<` 这一 gate token 的 subgraph 分数
- `replacement_full_signed_margin` / `completeness_full_signed_margin`：针对 `<` 相对 `echoFirstToken` 的 signed-margin full-graph 分数
- `replacement_subgraph_signed_margin` / `completeness_subgraph_signed_margin`：对应的 subgraph 分数
- `replacement_full`：整张 graph 的 replacement 分数
- `completeness_full`：整张 graph 的 completeness 分数
- `replacement_subgraph`：mapped circuit 作为 subgraph 的 replacement 分数
- `completeness_subgraph`：mapped circuit 作为 subgraph 的 completeness 分数
- `templateCount`：模板条目数
- `matched`：成功找到对应 `(layer, feature)` 的模板项数
- `exactCtxMatched`：连相对位置都完全对上的模板项数
- `coverage`：`matched / templateCount`
- `error`：这一条 prompt 失败时的错误信息

## 17. 读结果时的几个关键注意点

### 17.1 这不是因果验证，而是 graph-level proxy evaluation

这个脚本没有做 intervention，也没有做 feature ablation。它是在现有 attribution graph 上做结构映射和分数比较。

因此它衡量的是：

> 这个人工选出的 circuit，在 graph 表征里是否还能作为一个好的解释子图。

它不是严格意义上的 causal proof。

### 17.2 指标受 graph generation 超参数影响很大

因为新图本身是经过 `maxNLogits`、`desiredLogitProb`、`nodeThreshold`、`edgeThreshold`、`maxFeatureNodes` 剪出来的，所以：

- 不同剪枝阈值
- 不同 source set
- 不同 logit 选择策略

都可能显著改变 `coverage`、`replacement_*` 和 `completeness_*`。

### 17.3 `targetToken`、`topToken` 和 `gateToken` 不是一回事

新版脚本里至少要区分三类 token：

- `targetToken`：图生成器选中的目标 logit
- `topToken`：图中现有 logit 节点里概率最高的 token
- `gateToken`：脚本人为指定、用来分析 tool-call gate 的 `<`

这三者有时相同，有时不同。

因此在读结果时，你不能只看某一个 token 字段，而要先搞清楚：

> 当前这张图是在追踪哪个 token？概率最高的是哪个 token？而我又正在用哪个 token 去做 gate 分数？

### 17.4 `targetToken` 可能漂移

结果样例里可以看到：有些 prompt 的 `targetToken` 是 `<`，有些变成了 `Return`、`Please`、`Consider`、`In`。

这说明在这些 prompt 上，graph generator 选中的“主要输出目标”已经变了。此时即使模板还能映射，解释的也可能不是原来那种 tool-call 行为本身，而是别的输出倾向。

这在解读 generalizability 时必须单独看。

### 17.5 `*_gate` 和 `*_signed_margin` 可能为空

如果图里没有追踪到 `<`，那么 `*_gate` 分数没有意义，会写成空值。

如果图里没有同时追踪到：

- `<`
- `echoFirstToken`

那么 `*_signed_margin` 分数也没有意义，会写成空值。

这不是脚本 bug，而是因为这些目标化分数在数学上没有足够的 logit 支撑。

## 18. 研究上该如何使用这些结果

如果你的目标是研究一个 circuit 的 generalizability，可以把结果按下面三个层面读：

### A. 先看图到底在朝哪个行为方向走

优先看：

- `expectedBehavior`
- `decisionByTop1`
- `topToken`
- `probMargin`
- `signedProbMargin`

这一步回答的是：

> 这张图整体上是在支持 tool-call，还是支持 echo-first？它和 prompt 的期望方向一致吗？

### B. 再看是否对到了同类目标输出

优先看：

- `targetToken`
- `targetProb`

如果目标 token 已经漂移到完全不同的词上，那么默认分数通常不能直接和源 prompt 相提并论。

### C. 再看模板能否迁移

优先看：

- `coverage`
- `matched`
- `exactCtxMatched`

这是结构层面的泛化。

### D. 最后看迁移后的 circuit 还能保住多少解释能力

优先比较：

- `replacement_subgraph` 相对于 `replacement_full`
- `completeness_subgraph` 相对于 `completeness_full`

如果 subgraph 分数接近 full graph，说明你挑出的 circuit 在新 prompt 上依然接近“主解释子图”。

如果 full 很高、sub 很低，则更像是：

> 新 prompt 确实有一张解释性不错的图，但你从源 prompt 搬过来的那组 feature 已经不是主要机制了。

### E. 单独看它是否保住了 gate

优先比较：

- `replacement_subgraph_gate` 相对于 `replacement_full_gate`
- `completeness_subgraph_gate` 相对于 `completeness_full_gate`

如果默认分数一般，但 gate 分数仍然高，说明：

> 这个 circuit 可能不是在解释整张图，却仍然对 tool-call gate 很关键。

### F. 单独看它是否保住了 signed margin

优先比较：

- `replacement_subgraph_signed_margin` 相对于 `replacement_full_signed_margin`
- `completeness_subgraph_signed_margin` 相对于 `completeness_full_signed_margin`

如果这组分数高，说明：

> 这个 circuit 仍然在帮助 `<` 压过 echo-first 竞争方向。

这比只看 gate token 更像是在测“行为选择边界”而不是单个输出 token。

## 19. 最后用一句话概括这个脚本

这个脚本做的事情可以概括为：

> 先把一个源 prompt 上人工挑出的 circuit 抽象成“feature identity + 相对 logit 位置”的模板，再把这个模板映射到一批新 prompts 生成的 attribution graphs 上，随后同时评估它对整张图、对 tool-call gate `<`、以及对 `<` 相对于 echo-first 竞争方向的解释能力。

如果你后面要继续分析这份结果，最值得优先看的通常不是单个绝对分数，而是以下组合：

1. `decisionByTop1` 和 `signedProbMargin` 是否仍然朝预期行为方向走
2. `targetToken` 是否仍然对应同类输出
3. `coverage` 是否足够高
4. `replacement_subgraph / replacement_full`
5. `replacement_subgraph_gate / replacement_full_gate`
6. `replacement_subgraph_signed_margin / replacement_full_signed_margin`

这些指标合起来，才比较接近“这个 circuit 是否泛化”的完整证据链。