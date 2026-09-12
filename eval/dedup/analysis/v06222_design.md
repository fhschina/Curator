# V0.6.2.22 下一步：按决策归属路由与单段 witness

## 状态及依据

尚未建立 `.22` 在线 run 或发出外部模型请求。
`.21` 已按其预声明零重试要求停止，不重写旧完成状态，不挑选其某次响应。
本次下一步选择依据是：唯一重试 H0742 的 main 已 no/no，critic 对其最终结果没有决策权；
对全部258输入的 [离线路由审计](v06221_coverage_routing_audit.json) 找到111个同类确定分支，
147个非exact正例仍需要新 critic 判断。主任务依然是完整开发集与holdout达标，不以省调用替代语义改进。

## 实现与验证顺序

1. 使用新的可配置实验入口承载版本资源和显式 binding/renderer，不继续复制每版完整执行器，
   不改 `.20/.21` 冻结文件，不用 monkeypatch 偷换旧函数。
2. coverage 路由只依据新 main 与完整原 payload、且发生于 critic 请求之前：
   incomplete→LOW unresolved；main NO/U→原主结果；source-positive且实际exact→原exact结果；
   其余正例一律新调用。不得按标签、模型响应、是否报错或哪个版本更好决定请求集合。
3. 所有258个 pair 的最终公共结果都要输出、校验、计分。没有 critic 调用的行标记 `NOT_REQUESTED_OWNED_BRANCH`，
   不伪造 critic、空壳 proof、response digest 或“schema成功”。
   分开报告全pair公共完成率、实际调用集合的原始schema/quote完成率、native+outer并集重试率及实际HTTP数。
4. 目前证明只针对 coverage 的仲裁分支。对 control 的 v8 单独比较实际公共字段与 evidence，
   不能假设两者路由相同；若不能证明，保留 control 的完整调用集合并明确不同调用数，仍按全258相同分母比较语义。
5. 真正需要调用的 critic 增加通用单段 quote 约束和完整侧字段示例，同时修改 system、pair、YAML rubric。
   一条 source_quote 必须完整来自一个 cited unique span；其他材料放说明，不拼接两段或跳过原文中的共享文本。
   空字段仍显式给出空字符串，不省略 coverage_counterpart_ids 等必需字段。
   不改变原严格parser来接受错误引文，不根据H0742的标签选择新的语义结论。
6. 用 mixed shared/unique、分隔的两个unique片段、缺字段、原文quote及真正单向扩展做合成回归。
   两臂首轮和重试再过完整本地HTTP执行链。常规测试和报告生成不得并发改变source snapshot。
7. 冻结全部资源、全pair与call-set清单、权重、main、原生请求摘要、预算和准入条件后，才启动新在线预检。
   原有已知技术预检不能继续用于“未见”声明；新预检顺序在看到新响应前确定，不挑简单样本。

## 实验界限

保持相同258开发对、原权重、固定.14 main、同27B、temperature0/top_p1/输出4096、相同重试预算。
对所有实际调用做完整token预检，不因长度或格式风险删样本。两轮顺序、全258保护集与primary/over-group对照不变。
需冻结新的调用分母，实际called-pair重试率也必须≤1%；不能用111个未请求的行稀释重试率。
技术预检仍要求实际提交的每个请求首轮通过，不能因为H0742被确定路由就免除其他需要critic的样本检查。

这不是为 `.21` 失败另选评分口径，而是一个新的前置执行策略，必须独立验证并保留旧失败记录。
路由不修复main的语义漏判，也不能自动保证剩余147个正例判断正确。
局部通过后仍需fresh完整Judge局部验证，再进入完整1000的75% precision/75% recall/79% primary、over-group≤66与所有保护门槛，
然后未见400条独立人工holdout，再到20000。没有新增较大模型或reference修订授权，不自动release/commit/push。
