# V0.6.2.19：非正文等价／扩展冲突的离线诊断

## 范围与假设

.18 的 H0363 已暴露一个假设：main 声称完整非正文消息双向等价，critic 同时给出有引用的
同一记录实质扩展及 retained_conflict=NONE；现有 v8 最终保留 yes/yes。
不存在相互矛盾不等于不存在未覆盖内容。反过来，critic 声称扩展也不自动证明它是对的。

本轮是看过开发错误后提出的探索性代码反事实，不是新的盲测、在线 Judge 或已通过候选。
重放 .16/.17/.18 的全部 14 个正式 cell，每 cell 保留同一 258 对和原权重。
3,612 是历史响应数，不是独立样本数；不能池化为扩大后的人工集。
新模型调用为零，不使用预检结果、选择某次重试或生成新 critic 分数。
历史冻结代码、提示、main、critic、标签、缓存和结果均不修改。

## 唯一确定性反事实

从实际 v8 adapter 输出开始，只有同时满足以下条件才标记合同分歧：

1. 原完整文本可用且未截断，semantic diff COMPLETE，原有 span 坐标与原文精确对齐。
2. 非空原文本不完全相同；完全相同的文本仍受 exact 保护。
3. main 的完整 span ledger 引用有效，两侧均 NON_MAIN_ONLY，shared basis 为
   VERIFIED_EQUIVALENT_NON_MAIN_MESSAGE；实际 v8 最终为 yes/yes。
4. critic binding 为 ATOMIC_SAME_RECORD_EXTENSION，原引用满足双侧依据及 unique span 要求。
5. record_scope 为 SAME_SPECIFIC_RECORD 或 SAME_ORGANIZATION_DESCRIPTION，原引用有效。
6. retained_conflict 为 NONE 或 NOT_APPLICABLE，且原字段没有语法/packet 问题。

只使用结构化分数与已有引用检查，不靠理由中的关键词、语言、review ID 或旧标签触发。
不把“引用了 unique span”解释成已证明 opposite-side 不蕴含，也不据单侧 diff 数量猜方向。
不重新打开已有 no/no、未决、containment、substantive-main 或 evidence-invalid 路由。
完整翻译不凭语言差异触发；如果其 critic 也满足上述实质扩展冲突，仍如实报告可能回退，不另加样本豁免。

触发时仅在离线反事实中输出标准 v3 UNRESOLVED/LOW，绝不直接改为 no/no 或 containment。
复用 v3 unresolved schema（包括其历史 extraction-risk/no-evidence 约束），在独立诊断字段明确这是
合同冲突而非输入真的截断。原输出、原精确 evidence 和原始 critic 保存在来源及变化记录中。
这个现有 schema 的表达限制也是不能直接注册成生产版本的原因。

## 分析与决定

- 按每个历史 cell 分别报告原/反事实 weighted 与 unweighted P、R、primary、over-group、under-group、未决及 cohort。
- 未决保留在完整分母中，负例从 over-group 变为未决不计为主决策修复；正例变未决照常计漏判。
- 复用完整 .12 底线、22 个额外负例、38 个负例、5 benign、4 containment、12 exact 和 translation 检查。
- 保留原 cell 的运行质量描述，但不将旧在线 completion/retry 称为新反事实的在线质量。
- 与同一 raw response 的 v8 配对比较，不与不同运行的最好分数比较；不投票，不声称显著性。
- 输出变化的全部 review ID、原始三字段 critic、原 ledger 和引用问题，供区分模型误判与仲裁缺口。
- 本轮不以某项 precision 增加作为晋级理由。若只是错误转未决、没有 primary 净收益或有保护回退，
  停止该分歧即未决方案；下一步需要可验证的未覆盖命题合同，而非无条件扩大 veto。
- 无论结果如何，都不能据此次离线诊断启动完整 1,000 条、holdout、20,000 对、release 或 commit/push。

原 75% / 75% / 79% 门槛与独立人工 holdout 要求不变。
