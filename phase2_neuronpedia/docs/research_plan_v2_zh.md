用户的想法：
模型本身在OOD prompt上能力不足，那么仍然可以研究其heuristic shortcut / classifier / gate decision的机制。

1. 更改评估集。现在的prompts分类太杂。我不太确定：A. IN-DISTRIBUTION 包括POS_SEEN。B. OOD 包含模型会输出错误的prompts。模型能正确echo的NEG prompt应该放入那个集？模型能正确输出<的POS_UNSEEN应该放入哪个集？还是说应该视每一个prompt类的每一个具体prompt，模型能否正确输出，来具体情况具体分析？如果是这样，可以写程序自动筛选： webapp里应该有forward pass的api，网页前端用的就是这个，可以run 1 forward pass然后根据next token筛选prompt放入A集或者B集。不过同一个prompt可能改了首字母大小写，或者数字，就会进入不同的集，这在认知上可能有点混乱。还是说我们应该舍弃OOD prompts at all？
2. Circuit Eval Pipeline 完善。要对之前scripts目录下我们eval用的代码进行修改。可以考虑加入gate margin（logit(<) - logit(echo-first token)）作为核心指标。还需要的指标可能包括next logit 是否为 token "<"，P("<")，以及我们之前已经有的指标可以选择性保留。
3. Auto Circuit Selection + 稳定子电路发现。 让Circuit Eval Pipeline支持选出 TOK-K 个“好”的 subgraph（目前只能加载已经保存的subgraph），并与随机子图做 baseline 对比。然后在我们定义的评估集上系统性报告一些选定的metric，选那些metric也要想一下。一个担心是如果模型输出非常依赖浅层feature，可能无法在prompt集上找到能够泛化的电路（i.e.可能和随机选没有显著变好。不过我们得做了实验才能知道）

4. 讲story。我人工用webapp前端可视化选择的电路，对电路含的feature做分析。或者借鉴attribution-graph-probing对features做自动命名和supernode clustering。最后我人工做manual validation并讲故事。可以关注模型输出的gate到底有多brittle？主要依赖哪些浅层特征（例如：是否含运算符、是否出现 <tool_call> 字样、句首 token、数字形态等）？还可以具体对比base transcoder features和delta transcoder features的不同分工。

# 和copilot讨论后完善的后续research plan:
我先按“研究设计”而不是“代码实现”来整理：核心是把你现在的项目从“证明 semantic understanding”改成“刻画一个脆弱 gate/classifier 的机制”，然后把评估集、指标、选电路和讲 story 串成一条闭环。我再补一层“怎么组织实验单元”的建议：不要把 in-distribution / OOD / 正确与否混成一个轴，最好拆成多轴标签。这样同一个模板的大小写/数字变体落在不同 bucket，不是混乱，而是你要研究的 brittle boundary。

Read memory

**判断**
这版研究计划是可行的，而且比原来“证明 semantic understanding”更稳。你现在最强的信号不是“模型理解了 tool call 语义”，而是“微调后形成了一个很脆的 gate / shortcut classifier”。把项目改写成“刻画这个 gate 的机制、边界和子电路”是成立的。

真正需要调整的不是是否保留 OOD，而是不要再把“分布属性”和“模型是否做对”混成一个轴。

**1. 评估集怎么改**
不要只分 A 集 / B 集。建议每个 prompt 同时带 4 个标签：

- intent: POS 或 NEG
- distribution: ID 或 OOD
- behavior: correct、boundary、incorrect
- variant_group: 同一模板的轻微扰动组，比如首词大小写、数字替换、标点变化

其中 behavior 最好不要只看 top-1，对 gate 任务更合理的是看 margin。可以定义：

- gate margin: $m = \ell_{<} - \ell_{echoFirst}$
- signed margin: $m_s = y \cdot m$，其中 POS 取 $y=+1$，NEG 取 $y=-1$

然后按规则切片：

- capability slice: 行为正确且 $|m|$ 足够大
- boundary slice: 行为正确但 $|m|$ 很小，说明就在决策边界附近
- stress slice: 行为错误

这样就不会有“模型能正确 echo 的 NEG 应该放哪”这种混乱了：
- OOD 但正确 echo 的 NEG，属于 OOD-capability
- OOD 但正确输出 “<” 的 POS_UNSEEN，属于 OOD-capability
- 同一模板改了首字母大小写落到不同 slice，不是问题，恰好是你要研究的 brittle boundary

所以我的建议不是舍弃 OOD，而是保留 OOD，并把它分成：
- OOD-capability: 用来测可迁移电路
- OOD-boundary / OOD-stress: 用来测脆弱性与失败机制

如果你想让主结果更干净，可以把主 benchmark 定义为：
- ID-capability
- OOD-capability

然后把 boundary 和 stress 作为单独章节。

从这个结果看，最合理的组织方式是：

主 benchmark 用 main_id_capability 和 main_ood_capability
brittle boundary 分析用 boundary_analysis
“本来不是 stress probe，但模型失败了”的样本单独放 probe_failure
“本来就是 incorrect stress probe，而且确实失败了”的样本单独放 confirmed_incorrect_stress
“本来以为会稳定失败，但模型居然做对了”的样本单独放 unexpected_stress_probe_success
这比原来的 id_capability 和 ood_capability 干净很多。最关键的一个例子是：

旧的 ood_capability 有 23 条
新的 main_ood_capability 只有 21 条
少掉的 2 条不是普通 capability prompt，而是设计上属于 incorrect stress probe、但模型意外做对的样本
这正是我们之前计划里缺的那一层区分。否则你会把“stress probe 的意外成功”混进“真正想做主结果的 OOD capability”。


**2. Pipeline 应该怎么补**
你提的 gate margin 是对的，而且应该升成主指标。建议变成“三层指标”：

- 行为指标
  - next top-1 是否为 “<”
  - $P(<)$
  - $P(echoFirst)$
  - gate margin: $\ell_{<} - \ell_{echoFirst}$
  - signed margin: POS 用正号，NEG 用反号
- 电路指标
  - margin-targeted replacement / completeness
  - coverage
  - 跨 prompt 方差或 IQR
- 数据切片指标
  - capability / boundary / stress 的通过率
  - 各 family 的 mode accuracy
  - minimal-pair 翻转率

这里最关键的一点是：电路评分最好不要只对 “<” 做，而要直接对 margin 做。  
也就是在图打分里把 logit 权重设成：

- “<” 取 +1
- echoFirst 取 -1

这样 replacement / completeness 才真正是在测 gate，而不是只测单个 token。

你现有代码结构已经适合这么做，主要改这几处：
- [neuronpedia/apps/webapp/app/[modelId]/graph/score-core.ts](neuronpedia/apps/webapp/app/[modelId]/graph/score-core.ts)
- batch-eval-subgraph.ts
- analyze_circuit_eval.py
- visualize_circuit_eval.py

其中 echoFirst 我建议优先定义为“prompt 的首 token”，因为你的 echo 模式本来就是从那里开始竞争。  
可以顺手再存一个“top non-< competitor”作辅助诊断，但主定义最好固定。

**2.1修订后的pipeline**
**修订后的目标**

我认同现在应该把 batch circuit eval pipeline 的职责收窄成一件事：

- 给定一个已经选定的电路
- 在一组 prompt 上批量评估它是否泛化
- 主评估集就是 screening 之后的 capability slice
- boundary 和 stress 只作为可选诊断集，用来看为什么失效或不稳定

也就是说，这一步不再负责复杂的 prompt 角色再分类。screening 已经把 prompt 分成 capability、boundary、stress，batch eval 只需要消费这些结果。

**修改计划**

1. 输入约定只保留最必要的东西  
- 输入 prompt 文件优先使用 screening 产出的 capability.jsonl  
- 如果需要诊断，再额外跑 boundary.jsonl 或 stress.jsonl  
- 不再让 batch pipeline 理解 designRole 一类额外概念  
- prompt 自带的 meta 里保留 family、distribution、intent、promptClass、variantGroup 即可  
- 这样同一个 pipeline 可以直接回答两个问题：
  - 在 capability slice 上，这个电路有多 general
  - 在 boundary 或 stress slice 上，它为什么开始失效

2. batch evaluator 的每条 prompt 输出要分成两层  
- 行为层
  - top-1 是否是 <
  - $P(<)$
  - $P(echoFirst)$
  - probability margin：$m = P(<) - P(echoFirst)$
  - signed probability margin：$m_s = y \cdot m$，其中 POS 取 $y=+1$，NEG 取 $y=-1$
- 电路层
  - coverage
  - gate-targeted replacement / completeness：针对 <
  - signed-margin-targeted replacement / completeness：如果图里同时有 < 和 echoFirst 这两个 logit 节点，就用 +1 和 -1 做一组 margin score
- 这里我建议做一个现实一点的折中：
  - 行为指标始终报告 signed probability margin
  - 电路指标里，signed-margin score 作为首选机制指标
  - 但 gate-only score 不删，作为 fallback 和对照
- 原因很简单：graph 里有时不一定同时保留这两个 logit 节点。纯 margin-only 会丢掉一部分样本，尤其是很稳的 capability prompt

3. 对 selected circuit 的泛化评估，主结果只看 capability slice  
- 先在 capability.jsonl 上跑 batch eval
- 汇总时按 distribution 分成：
  - ID capability
  - OOD capability
- 这是最直接的 generalization 结果
- boundary 和 stress 不混进主分数，只做可选附录或诊断：
  - boundary：看电路在临界样本上是否不稳定
  - stress：看它在模型失败样本上是否完全失效，还是仍保留部分方向性

4. batch-eval-subgraph 的具体修改方向  
- 在 batch-eval-subgraph.ts 里补这些 per-prompt 字段：
  - echoFirstToken
  - probEchoFirst
  - probMargin
  - signedProbMargin
  - topNonAngleToken
  - topNonAngleProb
  - hasAngleBracketTarget
  - hasEchoFirstTarget
  - hasSignedMarginTarget
- 这里的 echoFirst 继续定义成 prompt 的首 token，不改
- 同时保留已有的：
  - targetToken
  - targetProb
  - probAngleBracket
  - coverage
  - replacement / completeness

5. 评分逻辑的修改  
- 在 [neuronpedia/apps/webapp/app/[modelId]/graph/score-core.ts](neuronpedia/apps/webapp/app/[modelId]/graph/score-core.ts) 不需要大改框架
- 主要是让 batch-eval 调两种 override：
  - gate-only：< 取 +1
  - signed-margin：< 取 +1，echoFirst 取 -1
- 然后在输出里同时存：
  - replacement_subgraph_gate
  - completeness_subgraph_gate
  - replacement_subgraph_signed_margin
  - completeness_subgraph_signed_margin
- 如果 signed-margin target 缺失，就写空，不硬造近似值

6. analysis 脚本的修改方向  
- 在 analyze_circuit_eval.py 里，把汇总重点改成：
  - capability slice overall
  - capability slice by distribution
  - capability slice by family
- 核心统计项：
  - mode accuracy
  - mean / median / IQR of signedProbMargin
  - mean / median / IQR of replacement_subgraph_signed_margin
  - mean / median / IQR of completeness_subgraph_signed_margin
  - signed-margin target availability rate
  - coverage mean / median / IQR
- boundary 和 stress 也可汇总，但明确标成 diagnostics，不和主 benchmark 混

7. visualization 的修改方向  
- 在 visualize_circuit_eval.py 里，优先做对 generalization 最有用的图：
  - capability slice 下 ID vs OOD 的 signedProbMargin
  - capability slice 下 ID vs OOD 的 signed-margin completeness
  - family 粒度的 mode accuracy
  - family 粒度的 coverage
- boundary / stress 的图先少做，除非你后面发现它们对讲 story 很关键

**我建议这一步不要做的事**

- 不把 auto circuit selection 一起塞进这次改动
- 不在 batch pipeline 里重新发明 prompt 分类逻辑
- 不强行只保留 margin score、完全删掉 gate-only score
- 不把 boundary 和 stress 混进主 generalization 分数

这样这一步会很清楚：

- screening 负责把 prompt 分成 capability、boundary、stress
- batch eval 负责在选定电路上测这些 prompt 的行为指标和电路指标
- 主结论先来自 capability slice
- boundary / stress 只做失效分析

如果你 approve，我会按这个范围改四个文件：

- batch-eval-subgraph.ts
- [neuronpedia/apps/webapp/app/[modelId]/graph/score-core.ts](neuronpedia/apps/webapp/app/[modelId]/graph/score-core.ts)
- analyze_circuit_eval.py
- visualize_circuit_eval.py


**3. Auto Circuit Selection 怎么做才像一个结果**
这个方向是有价值的，而且正好能把你和通用 attribution-graph-probing 区分开。关键不是“也能选 subgraph”，而是“你在做 gate 专用、带稳定性约束的子电路发现”。

建议目标从“找一个在所有 prompt 上都好”的电路，改成：

- 在 capability slice 上找到能稳定解释 signed margin 的最小子电路
- 在 boundary / stress slice 上检查它为什么失效或不稳定

一个可行的设计：

- 候选节点池
  - 从 source graph 全图来选，或从多个 anchor prompt 图取并集
- 选择目标
  - 最大化 capability slice 上的 signed-margin completeness
  - 同时惩罚节点数
  - 同时惩罚跨 prompt 不稳定
- 基线
  - 同尺寸随机子图
  - 同层分布 / 同节点类型分布的随机子图
  - 单 prompt top attribution 节点集合
- 稳定核心
  - 做 bootstrap 或 prompt 子集重采样
  - 统计每个节点被选中的频率
  - 高频节点组成 stable core，低频节点作为 fringe

如果最后发现“选出来的电路和随机差不多”，这不是失败，而是结果：
- 可能说明 gate 机制高度分散
- 或者过度依赖 prompt-specific 浅层特征，无法形成稳定稀疏电路

这两种结论都值得写。

**4. Story 怎么讲，才有贡献**
你现在最自然的论文故事不是“Gemma 学会语义工具调用”，而是：

- 微调产生了一个脆弱的工具调用 gate
- 这个 gate 在 ID 上有效，但在 OOD / 轻微扰动上非常不稳定
- attribution graph 揭示其更像浅层启发式分类器，而不是稳健的语义路由机制

你后面的人工解读和自动命名都可以围绕这个问题展开：

- 哪些特征最影响 gate margin
- 失败 prompt 和成功 prompt 的最小差异来自哪里
- delta transcoder 特征是否主要承担“新学来的格式 / 触发词 / 运算符检测”
- base transcoder 特征是否主要承担一般语言和上下文组织
- 是否存在专门的抑制回路，负责在 NEG 上压低 “<”

特别值得做的 3 个 case study：

- minimal pairs
  - 只改首词大小写、标点、数字形态
  - 看 margin 和子图怎么跳变
- POS 成功 vs POS 失败
  - 都是 OOD，但一个能触发 “<”，一个不能
- NEG 正确 echo vs NEG leakage
  - 看抑制路径是否缺失，还是被某些浅层 feature 覆盖掉

**原创性在哪里**
你不需要和 attribution-graph-probing 比“谁先有通用评分工具”。你的原创性可以放在这几件事上：

- gate-specific evaluation，而不是通用 token attribution
- 用 signed margin 把 POS/NEG 统一成一个机制目标
- capability / boundary / stress 的机制切片
- stable subcircuit discovery，而不是只看单 prompt 热点
- base vs delta 的分工分析
- 对 tool-call finetune 后“脆弱分类器”机制的系统实证

这已经不是“重复别人做过的库”，而是“在一个具体 gate task 上提出了新的机制问题和评测协议”。

**我建议你把项目目标改写成一句话**
不是“证明 Gemma 有 semantic understanding”，而是：

- 研究 tool-call finetune 是否主要引入了一个脆弱的 heuristic gate，以及这个 gate 的机制是否能被稳定、稀疏的子电路解释

这个问题是清楚的，而且无论结果是“找到稳定电路”还是“找不到、说明机制分散”，都有贡献。

**下一步**
1. 先把评估集重新标成多轴结构：intent、distribution、behavior、variant_group，不再硬分单一 A/B。
2. 把 pipeline 升级为 signed-margin 版本，把 margin 作为主指标，“<” top-1 和 $P(<)$ 作为辅助诊断。
3. 在 capability slice 上做第一版 auto circuit selection，加随机 baseline 和 bootstrap 稳定性。

如果你愿意，我下一步可以直接帮你把第 1 和第 2 步落到代码上：先给 prompts 加 variant / slice 所需元数据，并把 batch-eval 改成输出 echoFirst、$P(echoFirst)$、gate margin 和 signed-margin 的 replacement / completeness。