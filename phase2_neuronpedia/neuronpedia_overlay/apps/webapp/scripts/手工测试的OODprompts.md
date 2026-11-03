**对目前已测 OOD prompts 的分析**

**总判断**
目前这些结果最支持的结论，不是“模型对 tool-call 指令有稳定的 semantic understanding”，而是：模型学到了一个对模板、大小写、数字形态、句末标点、子句顺序和空格边界都高度敏感的 heuristic gate。  
这个 gate 在一部分 seen-style 模板上很强，但在 near-boundary 变体上会迅速翻转。

**主要发现**
1. OOD 失败不是均匀发生的，而是集中在少数非常具体的 surface-form interaction 上。很多轻微扰动其实并不会伤到模型，比如：
   - “What is 534 + 267” 仍然是 `<1.00`
   - “Give me the result of 930 - 224” 仍然是 `<0.99`
   - “I’m stuck on this calculation: 865 / 395” 仍然是 `<1.00`
   这说明模型不是“一碰 OOD 就崩”，而是存在一些非常窄但很真实的脆弱边界。

2. “What’s the difference between …?” 是目前最脆弱的 seen 模板之一。它和其他 seen 模板最不一样的地方，是对句末问号和后接子句特别敏感。
   - “What’s the difference between 598 and 914?” 是 `<0.98`
   - 去掉问号后，“What’s the difference between 598 and 914” 变成 `What0.68, <0.29`
   而同类的 “What is …” 和 “Give me the result …” 去掉句末标点并没有明显翻转。  
   这说明脆弱性是 template-specific 的，不是一般性的 punctuation sensitivity。

3. 小写首词和数字扰动之间存在很强的交互效应，这是目前最清楚的一条 near-boundary family。
   - “Compute the value of 112 * 309.” 是 `<1.00`
   - “compute the value of 112 * 309.” 是 `<0.80, compute0.19`
   - “compute the value of 11 * 309.” 是 `<0.63, compute0.36`
   - “compute the value of 1 * 309.” 变成 `compute0.69, <0.31`
   - “compute the value of 1 * 3.” 变成 `compute0.53, <0.47`
   但大写版本基本一直稳。  
   所以真正的问题不是“小数字本身”或“小写本身”，而是两者叠加后会把 gate 推向 echo-first。

4. 数字效应也不是简单的“数字越小越容易翻”。例如：
   - “compute the value of 1 * 309.” 已经翻到 echo
   - “compute the value of 1 * 39.” 又回到 `<0.77`
   - “compute the value of 1 * 3.” 再次压到边界附近
   这说明影响不只是数字规模，而更像是 expression 的具体 tokenization、局部 pattern 和记忆化模板一起在起作用。

5. anti-tool / explain-only 子句确实容易引起翻转，但它们看起来不是通过“语义理解”起作用，而是强烈依赖顺序和布局。
   - 后置 anti-tool：
     - “What’s the difference between 598 and 914? Do not use any tool.” 是 `<0.66, What0.33`
   - 后置 explain-only：
     - “What’s the difference between 598 and 914? Explain the method only and do not compute it.” 是 `<0.68, What0.32`
   也就是把 NEG 子句接在后面，模型往往还是偏向 tool-call。  
   但前置时，行为会明显变：
   - “Do not use any tool. What’s the difference between 598 and 914?” 是 `Do0.98`
   - “Explain the method only and do not compute it. What’s the difference between 598 and 914?” 是 `<0.82, Explain0.17`
   - “Explain the method only and do not compute it.What's the difference between 598 and 914?” 是 `<0.61, Explain0.39`
   这更像是句首 directive token 改写了竞争格局，而不是模型在全局整合“禁止工具/只解释”的语义。

6. 空格和排版边界会以非常不稳定、prompt-specific 的方式改写行为。
   - “What’s the difference between 598 and 914? Do not use any tool.” 是 `<0.66, What0.33`
   - 末尾多一个空格后，变成 `What0.60, <0.30`，反而正确 echo
   - 但 “What’s the difference between 598 and 914? Explain the method only and do not compute it.” 是 `<0.68, What0.32`
   - 末尾多一个空格后，反而升到 `<0.96`
   这说明“空格”本身没有统一方向的语义作用；它更像在扰动 token boundary，而这种扰动的影响完全依赖具体 prompt 的局部结构。

7. discourse prefix 不是统一破坏项，反而有时会增强 tool-call。
   - “Please, compute the value of 112 * 309.” 是 `<1.00`
   - “So, what’s the difference between 598 and 914?” 是 `<0.97`
   - “Now, give me the result of 930 - 224.” 是 `<1.00`
   所以“句子变长”或“加前缀”本身不是问题；问题在于前缀具体是什么词、它如何改变句首 token 竞争和模板匹配。

**机制解释**
这些结果最符合的工作假说是：tool-call gate 主要由几类浅层特征竞争决定，而不是由一个稳健的语义路由器决定。

更具体地说，至少有三类候选启发式在一起作用：
1. seen positive 模板特征  
   例如 “What is …?”、“Compute the value of ….”、“What’s the difference between …?” 这种高 prior 模板，会把模型往 `"<"` 推。
2. echo-first 竞争特征  
   prompt 首 token 及其词形本身就会参与竞争，例如 “compute”、“What”、“Do”、“Explain”。
3. token boundary 特征  
   包括 question mark、period-space、是否有 trailing space、数字位数、expression 的 tokenization 形态。

最终 next-token 决策更像是这些局部特征在最后一层共同推拉 `"<"`` 和 echo-first token 的结果，而不是“先理解全局意图，再统一决定工具调用”。

**对后续评估集的直接含义**
后面的正式 OOD benchmark 不应该再以“远 OOD” 为主，而应该围绕已经被验证有效的三条 near-boundary 主轴来建：

1. template fragility  
   重点保留 “What’s the difference between …?” 这类脆弱模板，同时把 “What is …?”、“Give me the result …”、“I’m stuck …” 保留为稳健对照组。

2. lowercase imperative × short expression  
   这是目前最清楚的一条 near-boundary family，值得系统扩展。

3. directive clause layout  
   同一句 anti-tool / explain-only，系统比较：
   - 前置 vs 后置
   - period+space vs period-no-space
   - 有无 trailing space

**目前最值得保留做后续电路评估的 near-boundary prompts**
这些已经非常适合作为 capability/boundary/stress 分析里的核心 case：

- “compute the value of 1 * 3.”  
  `compute0.53, <0.47`
- “compute the value of 1 * 309.”  
  `compute0.69, <0.31`
- “compute the value of 11 * 309.”  
  `<0.63, compute0.36`
- “What’s the difference between 598 and 914”  
  `What0.68, <0.29`
- “What’s the difference between 598 and 914? Do not use any tool.”  
  `<0.66, What0.33`
- “What’s the difference between 598 and 914? Explain the method only and do not compute it.”  
  `<0.68, What0.32`
- “Explain the method only and do not compute it.What's the difference between 598 and 914?”  
  `<0.61, Explain0.39`

**一句话总结**
目前这些 OOD 测试最有力地说明：模型学到的不是一个稳定的“理解工具调用意图”的机制，而是一个对模板、大小写、数字形态和空格边界都高度敏感的 heuristic gate；这个 gate 在某些 seen-style prompt 上很强，但在 near-boundary 变体上会迅速翻转。

-以下为seen 模板的句末标点扰动
13 TOKENS (缺少问号)
<bos>
What
 is
 
5
3
4
 +
 
2
6
7
↵

Next Token
<1.00
15 TOKENS
<bos>
Compute
 the
 value
 of
 
1
1
2
 *
 
3
0
9
↵

Next Token
<1.00
17 TOKENS
<bos>
What
’
s
 the
 difference
 between
 
5
9
8
 and
 
9
1
4
↵

Next Token
<0.76
What0.23
16 TOKENS
<bos>
I
’
m
 stuck
 on
 this
 calculation
:
 
8
6
 /
 
5
↵

Next Token
<0.80
Click0.10
<eos>0.05
-以下为seen 模板的首词大小写 / discourse prefix 扰动
14 TOKENS
<bos>
what
 is
 
5
3
4
 +
 
2
6
7
?
↵

Next Token
<1.00
17 TOKENS
<bos>
give
 me
 the
 result
 of
 
9
3
0
 -
 
2
2
4
.
↵

Next Token
<0.94
give0.06
16 TOKENS
<bos>
compute
 the
 value
 of
 
1
1
2
 *
 
3
0
9
.
↵

Next Token
<0.80
compute0.19
15 TOKENS
<bos>
compute
 the
 value
 of
 
1
1
 *
 
3
0
9
.
↵

Next Token
<0.63
compute0.36
14 TOKENS
<bos>
compute
 the
 value
 of
 
1
 *
 
3
0
9
.
↵

Next Token
compute0.69
<0.31
14 TOKENS
<bos>
compute
 the
 value
 of
 
2
 *
 
3
0
9
.
↵

Next Token
<0.66
compute0.33 (这几个同时出现小写首字母+数字扰动，有点意思)
12 TOKENS
<bos>
compute
 the
 value
 of
 
1
 *
 
3
.
↵

Next Token
compute0.53
<0.47
13 TOKENS
<bos>
compute
 the
 value
 of
 
1
 *
 
3
9
.
↵

Next Token
<0.77
compute0.23
18 TOKENS
<bos>
Please
,
 compute
 the
 value
 of
 
1
1
2
 *
 
3
0
9
.
↵

Next Token
<1.00
20 TOKENS
<bos>
So
,
 what
’
s
 the
 difference
 between
 
5
9
8
 and
 
9
1
4
?
↵

Next Token
<0.97
19 TOKENS
<bos>
Now
,
 give
 me
 the
 result
 of
 
9
3
0
 -
 
2
2
4
.
↵

Next Token
<1.00
-以下为同模板下的小数字 / round number 替换
10 TOKENS
<bos>
What
 is
 
3
 +
 
5
?
↵

Next Token
<1.00
14 TOKENS
<bos>
What
 is
 
3
0
0
 +
 
5
0
0
?
↵

Next Token
<1.00
13 TOKENS
<bos>
Give
 me
 the
 result
 of
 
7
 -
 
8
.
↵

Next Token
<1.00
17 TOKENS
<bos>
Give
 me
 the
 result
 of
 
7
0
0
 -
 
8
0
0
.
↵

Next Token
<1.00
12 TOKENS
<bos>
Compute
 the
 value
 of
 
4
 *
 
9
.
↵

Next Token
<0.99
17 TOKENS
<bos>
I
’
m
 stuck
 on
 this
 calculation
:
 
1
2
 /
 
3
.
↵

Next Token
<1.00
20 TOKENS
<bos>
I
’
m
 stuck
 on
 this
 calculation
:
 
1
2
0
 /
 
3
6
0
.
↵

Next Token
<1.00
-以下为seen 模板后接最小 anti-tool / explain-only 子句
20 TOKENS
<bos>
What
 is
 
5
3
4
 +
 
2
6
7
?
 Do
 not
 use
 any
 tool
.
↵

Next Token
<1.00
21 TOKENS
<bos>
Do
 not
 use
 any
 tool
.
 What
 is
 
5
3
4
 +
 
2
6
7
?
 
↵

Next Token
Do0.91
<0.08 (这个有意思，explainonly字句放在前后的差异)
20 TOKENS
<bos>
Do
 not
 use
 any
 tool
.
 What
 is
 
5
3
4
 +
 
2
6
7
?
↵

Next Token
<0.98 (这条是？和换行符之间少了一个空格，居然也翻转)
24 TOKENS
<bos>
What
 is
 
5
3
4
 +
 
2
6
7
?
 Explain
 the
 method
 only
 and
 do
 not
 compute
 it
.
↵

Next Token
<0.95
25 TOKENS
<bos>
Explain
 the
 method
 only
 and
 do
 not
 compute
 it
.
What
 is
 
5
3
4
 +
 
2
6
7
?
 
↵

Next Token
Explain0.64
<0.30 (.和What之间没有空格)
25 TOKENS
<bos>
Explain
 the
 method
 only
 and
 do
 not
 compute
 it
.
 What
 is
 
5
3
4
 +
 
2
6
7
?
 
↵

Next Token
<0.50
Explain0.45(.和What之间有空格)
23 TOKENS
<bos>
Give
 me
 the
 result
 of
 
9
3
0
 -
 
2
2
4
.
 Do
 not
 call
 a
 calculator
.
↵

Next Token
<0.99
23 TOKENS
<bos>
Do
 not
 call
 a
 calculator
.
 Give
 me
 the
 result
 of
 
9
3
0
 -
 
2
2
4
.
↵

Next Token
<0.96
27 TOKENS
<bos>
Do
 not
 use
 any
 tool
.
 I
’
m
 stuck
 on
 this
 calculation
:
 
8
6
5
 /
 
3
9
5
.
 
↵

Next Token
Do0.98
31 TOKENS
<bos>
Explain
 the
 method
 only
 and
 do
 not
 compute
 it
.
 I
’
m
 stuck
 on
 this
 calculation
:
 
8
6
5
 /
 
3
9
5
.
 
↵

Next Token
Explain0.88
<0.09
30 TOKENS
<bos>
I
’
m
 stuck
 on
 this
 calculation
:
 
8
6
5
 /
 
3
9
5
.
 Explain
 the
 method
 only
 and
 do
 not
 compute
 it
.
↵

Next Token
<0.89
I0.11
24 TOKENS
<bos>
What
’
s
 the
 difference
 between
 
5
9
8
 and
 
9
1
4
?
 Do
 not
 use
 any
 tool
.
↵

Next Token
What0.50
<0.50
24 TOKENS
<bos>
 Do
 not
 use
 any
 tool
.
 What
’
s
 the
 difference
 between
 
5
9
8
 and
 
9
1
4
?
↵

Next Token
 Do0.49
Do0.38
 *0.07
*0.02
28 TOKENS
<bos>
Explain
 the
 method
 only
 and
 do
 not
 compute
 it
.
 What
’
s
 the
 difference
 between
 
5
9
8
 and
 
9
1
4
?
↵

Next Token
Explain0.98
28 TOKENS
<bos>
What
’
s
 the
 difference
 between
 
5
9
8
 and
 
9
1
4
?
 Explain
 the
 method
 only
 and
 do
 not
 compute
 it
.
↵

Next Token
What0.97
-补测结果
"Compute the value of 112 * 309."<1.00
"compute the value of 112 * 309."<0.80 compute0.19
"Compute the value of 11 * 309."<1.00
"compute the value of 11 * 309."<0.63 compute0.36
"Compute the value of 1 * 309."<1.00
"compute the value of 1 * 309."compute0.69 <0.31
"Compute the value of 1 * 39."<1.00
"compute the value of 1 * 39."<0.77 compute0.23
"Compute the value of 1 * 3."<0.99
"compute the value of 1 * 3."compute0.53 <0.47

"What's the difference between 598 and 914? Do not use any tool." <0.66 What0.33
"Do not use any tool. What's the difference between 598 and 914?" Do0.98
"Do not use any tool.What's the difference between 598 and 914?"Do0.98
"What's the difference between 598 and 914? Explain the method only and do not compute it." <0.68 What0.32
"Explain the method only and do not compute it. What's the difference between 598 and 914?"<0.82 Explain0.17
"Explain the method only and do not compute it.What's the difference between 598 and 914?" <0.61 Explain0.39
多加一个空格
"What's the difference between 598 and 914? Do not use any tool. " What0.60 <0.30 Calculate0.01 <bos>0.01 Answer0.01 （末尾多一个空格，模型反而正确重复）
"Do not use any tool. What's the difference between 598 and 914? " Do0.96
"Do not use any tool.What's the difference between 598 and 914? " Do0.96
"What's the difference between 598 and 914? Explain the method only and do not compute it. " <0.96 （居然反而上升）
"Explain the method only and do not compute it. What's the difference between 598 and 914? "Explain0.66 <0.29
"Explain the method only and do not compute it.What's the difference between 598 and 914? "Explain0.76 <0.21

"What is 534 + 267?"<1.00
"What is 534 + 267"<1.00
"Give me the result of 930 - 224."<1.00
"Give me the result of 930 - 224"<0.99
"What's the difference between 598 and 914?"<0.98
"What's the difference between 598 and 914"What0.68 <0.29 (这条翻转了)
"I'm stuck on this calculation: 865 / 395."<1.00
"I'm stuck on this calculation: 865 / 395"<1.00