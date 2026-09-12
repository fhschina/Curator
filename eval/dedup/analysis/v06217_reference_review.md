# H0177 / H0634：同输入冲突复核，待确认建议

本文件是读取已保存可见文本后的 AI 开发复核，不是独立人工盲审，也不是已生效的 gold label。
不覆盖旧参照、不生成自动可导入的改标 CSV，不计算“改标后提升”的候选成绩。

## 可核验事实

- 两个 canonical pair ID 不同，但 A/B 完整文本、顺序、semantic spans、payload 摘要完全相同。
  packet=COMPLETE，truncated=false；不是截断隐藏的差别，也没有交换 A/B。
- 两条权重都为 10.045，同属 V0_NO__V0.5_YES 分层。复核不使用 URL、host 或 SUT 决策推断语义。
- B 只有 D-Link 论坛公共提示：发问前看 FAQ 或搜索；如需及时回复可联系区域办公室。
- A 保留 B 全部文本，另有查询语法（+、−、|、*）、按作者/论坛搜索、子论坛选项和搜索范围选择。
- 原 H0177：YES，A→B YES、B→A NO，CONTAINMENT，MAJOR，meaningful_addition。
- 原 H0634：NO，no/no，RELATED_NON_DUPLICATE，MAJOR，page_role。
- 这两个参照中的 pre_policy_review 字段均为空；不能凭字段缺省认定其具体标注人或声称其中一条由后来的 AI 改错。

## 建议统一为 no/no，但尚未应用

按照当前已约定的“非空同记录包含”与 non-main-only 边界，建议将两条统一为：

- same_duplicate_group=NO；a_can_replace_b=NO；b_can_replace_a=NO。
- relation_type=RELATED_NON_DUPLICATE；material_difference=MAJOR；reason_code=page_role。

理由：A/B 共享的是公共提示，不是已双侧锚定的同一搜索功能记录。A 独有完整查询语法及搜索选项，
不仅是一个可忽略的 Search 按钮；B 没有这些功能性内容。相同品牌 D-Link 和公共说明不足以把搜索功能绑定为
同一 substantive record 的新增字段。若两侧都路由为 non-main-only，完整消息/功能亦不等价，不能作非正文 containment。
此判断不需要推断看不见的 HTML 布局，也不否定“同一明确 FAQ 加一个真实步骤”仍可构成单向扩展。

另一种政策——只要文本语义为超集就允许包含，即使共用部分只是公共提示——会支持 H0177 的旧结论；
但那会重新放开本项目已经明确要收紧的非空主内容门槛，不能在一个例子上悄悄切换政策。

## 处理状态

待用户确认统一口径后，才能建立有 provenance 的新 reference 版本，并对全部历史候选使用同一新版重新计分。
不能只重算新候选、覆盖旧 CSV，或称本 AI 建议为新的独立人工校准。
本轮 .17 仍在原 258 对、原标签、原权重上评分；两条都保留在分母，矛盾也保留并显式报告。
这组矛盾最多涉及一个权重 10.045 的必然不一致项，不能解释或豁免多个负例保护和目标样本失败。
