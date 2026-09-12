# V0.6.2.27 待实施设计：相同typed合同下的语义规则块对照

本设计在`.26`预检结束后提出，尚无`.27`请求、响应或成绩。
`.26`typed候选8/8首轮通过，flat对照因H0384同类mode/引用错误重试，整体依协议停止。
两次不完整schedule不提供语义基线，也不授权更换阈值、反复重启直到预检碰巧通过。

## 唯一计划干预：替换一个声明的语义规则块

两臂统一dedup-retained-coverage-v3、同一编译器、typed binder、scope后处理与ownership路由。
以`.26`候选的system公共框架、pair模板和YAML rubric为共同任务，保留其完整命题重构与typed输出要求。
仅替换system的semantic_policy插槽：

- control：`.23`system从NONEMPTY RECORD ANCHORS开始，到SCOPE AND LOSS TYPES之前的原文规则块，
  包含非空记录、非正文保留意义与原简短成对边界。
- coverage：`.25`system从ESTABLISH THE ACTUAL RETAINED SUBJECT开始，到DIRECTIONS AND STRICT OUTPUT之前的
  原文规则块，即`.26`已使用的完整命题/实际主体/具体方法边界块。

这是**在共同typed任务上比较两个语义块**，不是纯`.23`版本对`.26`版本：共同pair/YAML本来就含简短
consent、Stories、identity、method提示，必须披露，不能声称control完全看不到这些概念。
system共同前后缀、生成的字段指南、schema顺序、pair/rubric、完整payload、实际反馈、温度和模型完全相同。
coverage的初始渲染消息应与`.26`typed候选逐条相同；只使用fresh响应，不复用该轮预检或缓存。
同一header保留在公共框架中，不另以不同版本角色说明制造第二个prompt变量。

运行前需保存所选块全文/来源摘要与两臂消息摘要，自动验证移除声明块后两臂system逐字一致、user消息一致。
禁止从row读取policy/标签/抽样原因作为模板变量；块由冻结配置静态指定。编译器不读取语义policy或reference。

## 实现与测试边界

新增不可变policy选择接口及实验配置，不修改`.26`已冻结runtime、typed合同、tests或旧实验。
优先复用已有typed builder、runtime、单臂执行/重放与公共schedule；只在native公开builder接口替换声明的system块。
两臂都是typed响应，不能把control误接回flat binder，或继续按旧critic列漏计其native纠错。

新增一对一测试需验证：

1. 每个policy块唯一、精确、静态嵌入；除该块外消息相同，candidate与`.26`渲染相同；
   未知块/边界歧义、row注入及schema变化均拒绝。
2. 两臂first/retry各一次完整Ray→本地HTTP→native三分支→writer→strict raw bind链路，验证真实请求与预渲染一致。
3. 原始非法分支/证据仍失败；全harmless context不冒充语义证据；真单向扩展、翻译、冲突、LOW U与owned行为不变。
4. 每个实际attempt、原始响应、反馈、native+outer纠错及终态失败均可重放，不把重复结果当首轮。

CPU与4个完整边界通过后，冻结实现/资源/测试/边界报告、两个schema、原人口/权重/main、call sets及token预算。
新运行根计划为`/raid/hfang/ihb/runs/v0.6.2.27-policy-diagnostic`，不复用旧judge cache。

## 调度、门槛与解读不变

继承原258对、230种payload、两个repeat顺序、147实际called/111 owned、原reference及分层权重。
两臂均27B、temperature0/top_p1、输出4096、thinking关闭、timeout600、max_retries2、并发16；
逐对实际input+4096+2048≤32768，不裁剪、不删除超预算pair。
先两臂各8条技术预检，须各8/8首轮有效、零native/outer纠错且HTTP200恰好8；失败则停止，不重启。
通过后R1 control→coverage、R2 coverage→control，1032公共输出、588基础请求，另16预检请求。
终态cell或确定性边界失败停止后续schedule。预检不发布语义排名，未完成schedule不算正式版本比较。

两轮全部保留。原local_gates、38/22负例及20负例/3benign底线、4真实containment、12exact、54translation、
H0347/H0453/H0748三目标、同步weighted primary不降/over-group不增、重复primary翻转不多于control均保留。
两臂分别100%最终严格完成，native+outer重试pair并集/实际called≤1%；HTTP服务失败另计。
合同completion不是蕴含正确率，不用合法引用证明harmless、identity或完整翻译判断正确。

只有全部原门槛通过，才支持fresh完整Judge局部验证；完整1000仍须75/75/79、over-group≤66和分类/重试保护。
本实验不是重新抽取的holdout。reference含预测可见AI修订的局限继续披露，不改H0177/H0634等未获批准口径。
之后的未见400条必须独立人工标注，不能由本agent替代独立人类；通过后才运行20000。
不自动release、commit、push或扩大模型权限。若typed两臂仍暴露新结构问题，按原失败策略保留并定位，而非宽松解析。
