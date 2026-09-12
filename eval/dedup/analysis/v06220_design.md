# V0.6.2.20：双向保留意义与精确 witness 合同

## 状态与研究假设

当前完成合同、仲裁、原生 NDD builder/renderer/解析集成及离线预算检查，**尚未进行在线评价**。
不是正式 release，不改旧标签、旧分数、v8 仲裁或任何旧冻结文件。
上一轮 .19 是有效进展：3,612 份离线重放否定了“有分歧就未决”的方案。
本轮要测试的不是同一方案的更多提示文字，而是把 critic 的“无矛盾”评分替换为明确的双向覆盖与未覆盖命题。

这会同时改变 critic 输出合同和仲裁，不能把将来的收益声称为只来自提示或只来自代码。
固定 .14 main 的配对实验可以隔离这整个组件，但不代表完整在线 Judge 的能力。

## 新合同

`dedup-retained-coverage-v1` 使用 NDD 原生 LLMStructuredColumnConfig，独立列 `qwen_dedup_coverage_witness`。
公共 Judge 输出仍为既有 `dedup-judge-output-v3`；新内部合同不是一次 v3 破坏性迁移。

- record_scope 与双侧独立 anchor：同一实质记录、非正文消息、不同/未绑定记录、exact 或未决。
- `a_meaning_in_b`：A 的全部保留意义是否被 B 覆盖；`b_meaning_in_a` 是反方向。
- 每侧明确 COVERED / UNCOVERED / UNRESOLVED，列出本侧全部 unique span ID，可使用完整同侧范围。
- COVERED 区分语义对应、无害控件、混合与无 unique span；语义对应有独立对侧引用与解释。
- UNCOVERED 给出一条最强的原文 witness：来源 IDs、精确 unique quote、差异类型、
  最接近的对侧 IDs/quote 或确实无对应、未找到等价还是值矛盾、实际保留后果。
  其他差异仍须在覆盖解释中交代；存在双侧实质损失时两侧都报 UNCOVERED。
- quote 不超过 240 个字符；逐字符验证它在所引的正确侧 span 中。只引用共同上下文不能充当 unique witness。
- 缺字段、旧 critic、未知/反向引用、漏 unique ID、虚构 quote、无对侧值却称矛盾等都校验失败，不能补造证明。
- 不完整/截断文本不能报完整可决 review；未决不当正确，不把缺失变成冲突值。

结构验证能证明引用与字段一致性，**不能确定性证明语义蕴含或“全局没有对应”**。
审计必须检查实际模型理由与原文，最终仍依赖独立人工校准，而不是将 witness 的存在本身称作语义正确证明。

## 仲裁变化与保护

[coverage_witness.py](../judging/coverage_witness.py) 复用原主 span ledger 及 v3 公共验证，不更改旧 adapter：

1. 实际输入不完整则未决；主 ledger 的已定 no/no 或未决不被 critic 重新打开。
2. 完整非空原文 exact 受保护，不捏造 unique evidence。
3. 正例 main 的 critic 不再拥有旧三字段的隐含等价规则。完整双向覆盖可纠正旧 critic 的误 veto，
   也可纠正 main 中误判为内容扩展的冗余内容；这两项能力都必须通过真实正负保护评估。
4. 非正文有真实未覆盖意义则 no/no，不能以 atomic extension 构造空集合 containment。
5. 同一实质记录、恰好一侧 MAIN_CONTENT 未覆盖、另一侧全覆盖、无矛盾才形成 containment。
   若 A 的内容被 B 覆盖而 B 有新增，则 A→B=NO、B→A=YES。不得把字段名误当替代方向。
6. 实际身份、页面功能、状态、成员、政策意义变化，或独立记录附着、双侧损失均 no/no；
   普通按钮/导航/语言差异本身不支持这些类型。记录作用不清则未决，不猜布局或身份。
7. 完整翻译语义 material difference=NONE；有 harmless chrome 可为 MINOR，不迁移历史 taxonomy。

新合同允许有独立双向覆盖依据的等价/方向纠正，因此不是保留所有旧正例方向的 v8 分支补丁。
不能声称仅通过现有四条包含单测就已保护真实包含；必须看新模型完整局部样本和重复评估。

## 原生集成与原始响应

[coverage_runtime.py](../judging/coverage_runtime.py) 复用现有模型/provider builder、Ray 生命周期、
DataDesignerStage、输入输出和 executor；只添加新 structured column，绝不在同一请求中调用旧 critic。
重用 `.16 control` YAML 的模型/运行参数，不复用其 judge 列或缓存。
新 system、pair、YAML rubric 与生成的 JSON schema 描述同步；rubric 是可信配置，在 builder 中静态注入，
原模型输入只包含相同可见 span packet、truncated 状态及安全 retry feedback。

本地已验证 NDD 原生 parser 默认会 pruning 额外字段。因此启用完整原始会话 trace，
再使用原生 StructuredResponseRecipe(pruning=False) 校验原始最终 assistant 文本，
要求它与 NDD 解析列完全相同。自动删字段不能冒充原始 schema 完成。
trace 仅供审计，绝不反馈主 Judge、标签、旧模型预测或抽样原因。

## 已完成的离线技术检查

- 34 个新的一对一测试，包括双侧方向、权限/身份/角色/状态/成员、UI、翻译、真实扩展、双侧损失、
  exact、主 no/no/未决保护、截断、引用篡改、原生 schema/renderer、raw response 与 pruning 检查。
- 用 `.18` 已冻结的 258 对和本地已固定 tokenizer，按实际原生 renderer（含 schema/rubric）计算预算。
- control 最大输入 17,067 tokens，coverage 最大输入 16,734 tokens；输出 4,096、安全余量 2,048，
  每一对都在 32,768 的客户端预算内。没有截断证据、删样本或新增模型调用。
- 这不是服务端上下文上限声明，也不是模型输出一定完整的保证。
- 全部 token/请求摘要留在
  `/raid/hfang/ihb/runs/v0.6.2.20-coverage-contract-preparation/token_preflight.json`。

## 下一阶段：先冻结再运行

尚需实现新合同专用的 prepare/run/replay/summarize harness，不能直接使用只接受旧三字段 critic 的 bind_fixed_main。
先完成不访问外部模型的运行边界检查，再冻结以下协议和所有依赖：

- 相同 258 对/原权重/固定 .14 main；同一 27B 模型、temperature=0、top_p=1、4096 输出上限、原并发/重试预算。
- control 是真实 `.14` critic + v8，candidate 是新 coverage critic + 新仲裁。两者都用全新 run root/缓存。
- 两臂各 8 对独立技术预检；正式 R1 control→coverage、R2 coverage→control，每 cell 全部 258 对。
  不能把预检回灌正式样本，不能按观测结果挑一次主判断或某次 retry。
- 每个 candidate 都必须校验原始 trace、内部 schema/引用、公共 v3 和双侧精确 evidence；记录 native correction 与 outer retry。
- 复用现有全部 38/22 负例、.12 20 负例/3 benign 底线、4 containment、12 exact、54 translation；
  H0347/H0453/H0748 三项均须正确，且检查理由。不得把只有 quote 命中算作修复。
- 两次配对的 weighted primary 均不得低于 control，over-group 不得增加，所有原局部门槛均通过；
  另报告同响应方向纠正、全部 cohort/置信档和两次重复波动，禁止按最好的一次晋级。
- 完成率 100%、outer retry ≤1%；若技术预检暴露合同不稳定，停止正式提交，保留失败记录，另开版本修订。

即使这项固定-main 诊断通过，也要先做独立 fresh 局部在线完整 Judge 验证，再决定完整 1,000 条运行。
最终原 75% precision / 75% recall / 79% primary、误合并 ≤66、全部分类保护条件与未见 400 人工 holdout 均不变。
本阶段不涉及较大模型调用、待确认 reference 修订、holdout 读取、release、commit 或 push。
