# V0.6.2.9 完整 1,000 条开发集诊断

## 结论

这次完整诊断发现了旧 127 条残差集没有覆盖的召回问题。V0.6.2.9 的加权 precision 为 **73.86%**、recall 为
**59.01%**、主决策完全一致率为 **88.27%**。Precision 和 recall 均未达到原定 75% 门槛；不能发布或进入 holdout。

下一步不宜继续全面收紧 containment，也不宜直接撤掉 critic。优先修复可定位的翻译路由缺陷，同时将 critic
争议中的真实过度否决与尚未统一的标签边界分开。此次没有实施新版本、修改标签或生成 release approval。

本次依据用户明确请求扩大诊断范围，不改变先前残差实验失败的历史结论。执行前冻结范围见
[诊断协议](v0629_full_development_protocol.md)。完整机器可读结果见
[统计与同输出归因](v0629_full_development_summary.json)，全部 154 条主决策分歧见
[错误清单](v0629_full_development_errors.csv)。

## 参照与统计口径

- 使用冻结的 `v0628_policy_reconciled_labels_1000.csv`：795 条 NO、204 条 YES、1 条 UNRESOLVED。
- 相对最初参照，134 条主决策经过此前复审修改，包含 prediction-aware AI 判断；不是独立人工 holdout。
- 权重沿用 CSV 的 `stratum_population_n / stratum_sample_n`，合计 10,023。加权结果不等于完整 20,000 对的实测质量。
- 原有 reason tags 作为诊断分组保留，不保证它们与复审后的语义类型完全一致。
- 对参照重复的 UNRESOLVED 计作召回漏判；历史 translation MINOR 只影响附属 taxonomy，不决定主版本选择。
- 全判 NO 的主决策一致率已达未加权 79.50%、加权 84.94%。类别不平衡下，只看 88.27% 一致率会掩盖严重漏判。

## 总体结果

| 指标 | 未加权 | 加权 |
|---|---:|---:|
| Duplicate precision | 76.76% | 73.86% |
| Duplicate recall | 53.43% | 59.01% |
| Duplicate F1 | 63.01% | 65.61% |
| 主决策完全一致率 | 84.60% | 88.27% |
| 附属 taxonomy 一致率 | 55.50% | 57.65% |

33 条 over-group 中，29 条被判 NEAR_SURFACE，只有 4 条被判 CONTAINMENT：H0468、H0723、H0928、H0221。
错误重心已经不是早期那种几乎全部由 containment 造成的误合并。

以下互斥划分合计 **154 条**主决策分歧：

| 主决策错误 | 条数 | 权重质量 |
|---|---:|---:|
| 误合并：参照 NO，预测 YES | 33 | 312.49 |
| 明确漏合并：参照 YES，预测 NO | 89 | 590.39 |
| 预测 UNRESOLVED，且不匹配参照 | 28 | 246.20 |
| 分组相同，但替代方向错误 | 4 | 26.84 |

实际共 29 条 UNRESOLVED，其中 H0239 与参照一致，不在上述 28 条错误中。6 条 UNRESOLVED 的参照为重复，
所以总召回漏判为 89 + 6 = **95 条**。另有 291 条仅 taxonomy 不一致，不能混入上述主决策错误总数。

## 为什么 127 条残差实验掩盖了问题

| 本次运行分区 | 样本数 | 参照重复数 | 误合并 | 召回漏判 | 加权 precision | 加权 recall | 加权主决策一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原残差集 | 127 | 21 | 21 | 4 | 43.48% | 79.36% | 78.56% |
| 其余开发样本 | 873 | 183 | 12 | 91 | 86.05% | 56.10% | 89.59% |

**91/95 的漏判发生在其余 873 条；critic 新引入的 38 条主决策参照分歧也全部发生在这部分。**
这些样本仍是开发集，不是未接触过的 holdout。

39 条参照 CONTAINMENT 中，仅 8 条主决策方向完全一致、9 条仍被分到重复组。
长期检查的 4 条 containment 保护样本本次全部正确，但其余 35 条只有 4 条方向正确。
这说明 4/4 的窄保护集不能代表完整 containment 能力；其中也有需要重新确认参照口径的例子。

原 127 条可见 payload 与历史运行完全一致，但 12 条主决策元组发生变化。本次该部分未加权一致率为 77.95%，
历史为 76.38%；这是同版本重复执行波动，不能算版本提升。H0017、H0572 本次正确，但 23 条诊断负例中 H0373
重新出现误合并；H0748 仍是方向错误。没有以本次局部改善撤销旧失败结论。

## 同输出归因：critic 既有收益，也有损失

对本次保存的原始输出做离线对照；未发起额外模型调用。启用原 V0.6.2.9 仲裁的重放与全部 1,000 条公开字段完全一致。

| 相同主 Judge 输出 | 加权 precision | 加权 recall | 加权主决策一致率 | 误合并 | 召回漏判 | false containment |
|---|---:|---:|---:|---:|---:|---:|
| 主 Judge，关闭 critic 仲裁 | 63.63% | 80.23% | 87.06% | 71 | 50 | 33 |
| 实际 V0.6.2.9 仲裁 | 73.86% | 59.01% | 88.27% | 33 | 95 | 4 |

Critic 修正 38 条主决策，同时把 38 条原本符合参照的主决策变成分歧。后者的历史 reason tags 为：
meaningful_addition 22、translation 12、chrome_only 4。还有 7 条原本已是方向错误的重复样本被改成漏合并。
这些是**相对冻结参照的归因**，不能未经复审就将每一次否决都视作真正的语义错误。

因此，关闭 critic 虽能恢复召回，但 precision 掉到 63.63%，不是可接受方案。它消除的 38 个误合并必须保留为防回退集。

## 可定位的问题与边界复审

### 1. 完整非正文翻译被确定性路由拒绝：23 条

`_v3_span_ledger_target` 在 `COMPLETE_FAITHFUL` 分支只接受 `SUBSTANTIVE_MAIN` 与 `VERIFIED_SUBSTANTIVE_RECORD`。
两个 NON_MAIN_ONLY 页面即使具有 `VERIFIED_EQUIVALENT_NON_MAIN_MESSAGE`、双侧语义覆盖和完整翻译判定，仍被改为
`SPAN_UNVERIFIED_TRANSLATION_IDENTITY`，走 no/no。这 23 条全部在其余 873 条中，并全部与当前重复参照相悖。

- H0007：Skechers 西班牙语／葡萄牙语 cookie 说明，主 Judge 和 critic 均认定语义相同，适配器仍拒绝。
- H0641：法语／意大利语同一 cookie 说明，主 Judge 正确认定完整非正文消息翻译，仍被该分支拒绝；critic 另有语言即状态变化的错误推理。

这是应优先修复的内容类型／翻译路由不一致，不需要再增加一个 LLM 阶段。但新分支仍需检查完整消息身份、
双侧覆盖与真实 material conflict，不能见到翻译标记就无条件合并。

规模上也应保持清醒：即使将这 23 条全部恢复为参照答案，加权 recall 的理想上界也仅从 59.01% 到 **62.37%**，
precision 到 74.92%。这是理想化计数推算，不是实测新版本，不能独自解决总体召回问题。

### 2. Critic 将等价与单向扩展混为一谈

主 Judge 判定完整翻译后，critic 又否决 10 条、将 2 条改为 UNRESOLVED。H0545 的 critic 明确承认芬兰语／瑞典语
政策文本含义相同，却因为双侧都有独有字符串、不是单向扩展，最终认定它们是 separate records。

修复方向是让 critic 区分双向等价与单向扩展的证明要求，按语义识别双侧覆盖；不同语言不构成 state/identity 冲突。
H0312 则展示了另一种真实过度否决：额外日历导航和 `úřadem a`／`úřadema` 的空格粘连被解释成记录状态或语法事实冲突。
这些案例适合作为小范围 prompt 对照，不应借此削弱所有 record/role veto。

### 3. 最大加权争议之一仍需确认“记录单位”，不能追着标签放宽

`same-record extension -> separate-record/template attachment veto` 分支共 53 条：28 条符合参照，25 条为参照漏判，
后者权重合计 230.84。这里不能统一删除 veto：

- H0048：共同部分是 Common Dreams 募捐横幅，A 额外增加 Persis Yu 作者简介。参照为 containment，但新同记录口径下，
  募捐内容与作者记录不天然是同一个实质内容记录。
- H0196：共同部分是三个文章推荐摘要，A 额外出现 Taylor Swift 内容。参照为 containment，但不同文章与推荐集合的角色需先明确。
- H0800：PRO RETINA 资料库介绍与新增具体播客条目。必须先明确 catalog 的记录单位，以及 list membership 变化与合法内容扩展的边界。

这些是标签／policy 冲突嫌疑，不是此次宣布的标签错误。建议先单列 22 条 meaningful_addition critic 回退案例复审，
不展示版本预测给新的独立评审者，固定记录单位后再调整 critic。本轮保留全部原标签和 59.01% 的诊断结果。

### 4. 非正文分支的误合并不能忽略：25/33

89 条走 `ASYMMETRIC_CRITIC_DEFERRED_TO_NON_MAIN_LEDGER` 的正例中，25 条是参照误合并，占全部误合并的 75.8%，
权重合计 248.42。H0189 中，B 在免责声明外增加三种具体水处理服务类别，主 Judge 把它们当作无关 chrome；
critic 指出了页面角色／服务集合问题，但被非正文优先权覆盖。

应区分无意义界面差异与可见的服务目标、身份、用途、权限、范围变化。不能重复 V0.6.2.11 的做法，将任何 consent
按钮或措辞差异都升级为 veto。真实否决须有对应内容证据和明确适用范围。

剩余 4 个 false containment 仍需作为小回归集保留。例如 H0468 仅共享通用保修／配送服务，不能据此将 `180W`
认定为同一产品的新增字段；缺少具体产品身份锚点。

### 5. 输入／证据覆盖限制：24 条无法评估的 packet

13 条截断与 11 条差异片段超限全部得到 UNRESOLVED；其中 1 条匹配参照，23 条形成主决策分歧。
其余 976 条完整 packet 中另有 5 条 UNRESOLVED。应单独评估长文、压缩差异片段或有界补充证据流程，
不能以伪造证据或强制输出 YES/NO 提高完成率。此次 schema completion 100% 不代表语义覆盖率 100%。

## 下一版建议：分开验证，不做一次大改写

1. **确定性修复优先**：从 .9 冻结分支修复完整非正文翻译的路由，补真实语义冲突与 faithful translation 回归测试。
   先对已保存的全 1,000 条原始响应做离线重放，确认影响范围；不要把它误称为新模型实验。
2. **先统一记录单位，再优化 critic**：独立复审上述 22 条 meaningful_addition 边界；随后单独验证双向等价、真实单向扩展、
   模板／独立记录三类的 critic 规则。保护这次 critic 修正的 38 个误合并，不批量翻回旧标签。
3. **单独收紧非正文等价的作用对象／角色要求**：针对 25 个非正文误合并，配对加入正确 chrome/translation 正例，防止再牺牲召回。
4. **证据覆盖与吞吐另做实验**：24 个不可用 packet 与 158 次 HTTP 429 是不同层的问题，不与语义 prompt 改动合并归因。

后续至少使用完整开发集作为回归面，同时报告 127/873、参照修改／未修改分区、precision/recall、保护队列及各错误类型。
不以少量保护样本通过、总体准确率升高、DeepSeek agreement 或 proxy 总分决定晋级。任何 holdout 或正式发布仍需另行授权与准入。

## 历史结果：仅作同参照重算的背景

| 历史／当前结果在相同当前标签上评分 | 加权 precision | 加权 recall | 加权主决策一致率 |
|---|---:|---:|---:|
| V0.6.1 历史输出 | 41.38% | 89.30% | 74.03% |
| V0.6.2.2 历史输出 | 60.94% | 94.35% | 89.29% |
| 本次 V0.6.2.9 | 73.86% | 59.01% | 88.27% |

这些数值使用相同当前参照，但 prompt、可见 payload 表示和执行 contract 不同；不能称为等 payload 的受控消融，
也不能与当年旧标签报告直接混用。134 条曾修改参照的样本加权主决策一致率为 93.32%，其余 866 条为 87.46%。

## 运行、置信档与可复现性

- Run root：`/raid/hfang/ihb/runs/v0.6.2.9-full-development-diagnostic`。
- 原 .9 main/critic prompt 资源哈希不变；旧 127 条原始响应重放与历史公开结果全部一致；重复样本的新旧可见 payload 完全相同。
- 全部 1,000 条使用新 run root/cache 执行。1,000/1,000 schema 有效，0 终态错误，0 样本级外层重试；3,831 段证据通过可见文本与 offset 精确匹配检查。
- Runtime 接口事件 2,212 次：HTTP 200 为 2,054 次，HTTP 429 为 158 次；另有预检 20 次成功调用。不能将“0 外层重试”解释为“0 API／内部修正重试”。
- 两个 500 条批次分别耗时约 22 分 52 秒、13 分 26 秒；限流后的 SDK 降并发及逐步恢复是主要额外运行开销。
- LOW：33 条，主决策一致率 9.09%、加权 10.01%；MEDIUM：967 条，一致率 87.18%、加权 90.48%。没有 HIGH。这是本开发集上的经验正确率，不是概率校准。
- 全部 280 项 dedup 测试通过；新增诊断工具的 Ruff lint/format 检查通过。冻结 Judge 源码、prompt、标签未改动；本任务的 Ray 进程已关闭。
- 全部 1,000 条诊断行位于 run root 的 `reports/development_rows.csv`；仓库仅保存摘要与错误清单。原生 SUT/DeepSeek 输出仅作诊断，不用于版本选择。

复现统计（输出需使用新路径，防止覆盖已冻结报告）：

```bash
python -m eval.dedup.analysis.development_diagnostic \
  --run-root /raid/hfang/ihb/runs/v0.6.2.9-full-development-diagnostic \
  --labels eval/dedup/analysis/v0628_policy_reconciled_labels_1000.csv \
  --original-labels eval/dedup/analysis/hs_blind_adjudication_1000.csv \
  --residual-subset eval/dedup/analysis/v0622_reconciled_residual_candidates.csv \
  --historical-run /raid/hfang/ihb/runs/v0.6.1 \
  --historical-run /raid/hfang/ihb/runs/v0.6.2.2-dev-final \
  --replay-v0629 --output <new-summary.json> --rows <new-rows.csv> --errors <new-errors.csv>
```
