# V0.6.2.12：完整 1,000 对在线开发评估

## 结论

按冻结版本 → 20 对技术预检 → 完整 1,000 对在线评估 → 离线归因的顺序完成。
**技术通过，质量不通过，.12 保留为失败的开发实验，不晋级 holdout、20,000 对运行或 release。**

加权 recall 达到 77.59%，但 precision 只有 67.99%，未达到 75%；原有 38 条 critic 修好的负例仅保住 20 条。
不能把测试通过、schema 完整、总体一致率超过 79%，解释成 precision/recall 同时达标。
翻译与召回有真实的参照一致率改善，但过宽的记录绑定重新打开了误合并。

评估时间：2026-09-10 UTC。完整运行 02:33:28–02:42:03，约 8 分 35 秒。
全部语义统计在运行完成后统一生成，没有边跑边调整 prompt、代码或标签。

## 运行完整性与比较口径

- Run root：`/raid/hfang/ihb/runs/v0.6.2.12-full-development`。全新缓存，不迁移 .9 结果。
- 模型 `nvidia/qwen/qwen3.8-27b`；temperature=0、top_p=1、max_output_tokens=4096、并发 64；两块各 500 对。
  .9/.12 的 1,000 个可见 payload 完全相同，相关模型、截断和执行参数一致。
- 20 对预检单独保存输出并停止于正式提交前；通过后完整运行重新调用全部 1,000 对，没有复用预检预测。
- 预检 20/20 有效、0 终态错误、0 样本级外层重试、0 UNRESOLVED。
  4 条 retained-conflict 引用警告位于不适用的分支，没有参与最终裁决；不应写成“所有内部警告为零”。
- 完整运行 1,000/1,000 有效、0 终态错误、0 样本级外层重试。所有 public v3 字段与最终 evidence offsets 均通过验证。
- 完整运行记录 2,037 个 HTTP 完成事件：2,029 个 200、8 个 429；预检另外 40 个 200。
  外层重试率为 0%，不代表没有内部纠错调用或传输重试。
- 保存的 .12 main/critic 原响应按 provider digest 精确匹配；以 `v6` 离线重放，1,000 条的全部 public 字段与正式结果相同。
- 本轮 runner 增加了可独立停止的 `preflight --pairs 20`，保留普通 `run` 原有默认行为；新增 6 条测试。
  在线调用前 dedup 全套 344 个测试通过，Ruff 检查通过。运行开始后未修改冻结的推理实现与资源。
- 报告完成后再次通过实现、资源、contract、标签、候选集及 payload 的冻结摘要校验；现有缓存中的适用 pre-commit
  文件检查与 Ruff 均通过。未安装环境、执行依赖同步或提交代码。本任务启动的 Ray 进程已停止。

计分使用未改动的 `v0628_policy_reconciled_labels_1000.csv`：204 个参照重复、795 个分离、1 个未决，权重合计 10,023。
其中 134 条主决策经历过 prediction-aware AI 修订；这是开发参照，不是独立人工盲审或 holdout。
.9 是历史同设置实测，并非同时随机对照；route-only 是旧原响应离线重放，不能当成新 prompt 的在线成绩。
faithful translation 的历史 `MINOR` taxonomy 不影响主决策版本选择。未使用 DeepSeek agreement、proxy 总分或 SUT 一致率选版本。

## 总体结果

| 指标 | .9 在线 | .9 原响应 + 翻译路由修复（离线） | .12 在线 |
|---|---:|---:|---:|
| 加权 duplicate precision | 73.86% | 74.92% | 67.99% |
| 加权 duplicate recall | 59.01% | 62.37% | 77.59% |
| 加权主决策完全一致率 | 88.27% | 88.77% | 88.28% |
| 加权 duplicate F1 | 65.61% | 68.07% | 72.47% |
| 未加权 duplicate precision | 76.76% | 80.00% | 74.44% |
| 未加权 duplicate recall | 53.43% | 64.71% | 81.37% |
| 未加权主决策完全一致率 | 84.60% | 86.90% | 87.60% |
| 误合并数 | 33 | 33 | 57 |
| 参照重复漏判数（含未决） | 95 | 72 | 38 |
| false containment：参照分离却预测 containment | 4 | 4 | 22 |

.12 相对 .9 修正 75 个原主决策分歧，新增 45 个分歧；801 个共同正确，79 个共同错误。
这些“修正/回退”均相对于冻结参照，不等于新的人类仲裁结论。
未加权准确率提高 3 个百分点，而加权只提高 0.014 个百分点；新增错误与修复样本的权重差异不能忽略。

本轮 124 个主决策分歧的互斥分布：误合并 57、已决漏判 33、错误未决 29、仅方向错误 5。
共有 30 个 UNRESOLVED，另 1 个与参照相同；其中 5 个参照重复未决计入召回漏判，所以 duplicate misses=33+5=38。
39 个参照 containment 中，21 个方向完全正确、1 个误判双向、17 个判为分离。

## 门槛与保护检查

| 检查 | 本轮 | 判定 |
|---|---:|---|
| 加权 precision >=75% | 67.99% | 失败 |
| 加权 recall >=75% | 77.59% | 通过 |
| 加权主决策完全一致率 >=79% | 88.28% | 通过 |
| 误合并 <=66（相对最初 110 至少下降 40%） | 57 | 通过；但相对 .9 的 33 明显变差 |
| identity / meaningful-addition / translation 加权准确率降幅均 <=3 pp | -1.50 / +1.69 / +45.18 pp | 通过 |
| schema completion=100%，外层重试率 <=1% | 100% / 0% | 通过 |
| 38 条 critic 修正负例保持 NO/NO | 20/38 | 失败，18 条重新误合并 |
| 原有 4 条真实 containment 方向保护 | 4/4 | 通过；不代表全体 containment 达标 |
| 原有 5 条 benign 正例方向保护 | 3/5 | 失败：H0679 未决，H0748 分离 |

38 条负例中的回退 ID：H0597、H0789、H0872、H0278、H0119、H0132、H0790、H0141、H0771、
H0828、H0267、H0242、H0700、H0253、H0594、H0977、H0413、H0558。
H0748 在 .9 已有方向错误，本轮从方向错误变为漏判；不能把两条 benign 失败都说成新增错误。

## 错误集中在哪里

### 分区不能相互替代

| 分区 | .9 加权 precision / recall / 主决策一致率 | .12 加权 precision / recall / 主决策一致率 | 误合并 .9→.12 | 漏判 .9→.12 |
|---|---|---|---:|---:|
| 旧 127 条困难集 | 43.48% / 79.36% / 78.56% | 28.55% / 75.58% / 65.15% | 21→37 | 4→5 |
| 其余 873 条 | 86.05% / 56.10% / 89.59% | 84.15% / 77.87% / 91.44% | 12→20 | 91→33 |

.12 的大部分召回收益发生在其余 873 条，负例保护回退则集中于旧困难集。不能只用其中一部分宣布成功。

### 语义错误簇

以下按照固定参照 reason code 分组，不同于模型自己声称的 content profile。

| 簇 | 样本数 | 加权主决策一致率 .9→.12 | 误合并 .9→.12 |
|---|---:|---:|---:|
| translation | 54 | 44.63%→89.81% | 0→1 |
| meaningful_addition | 156 | 75.72%→77.42% | 6→16 |
| identity_slot | 296 | 95.24%→93.73% | 6→11 |
| boilerplate_only | 91 | 92.64%→89.69% | 2→6 |
| page_role | 94 | 90.47%→88.99% | 10→12 |
| chrome_only | 78 | 76.58%→86.38% | 0→0 |
| factual_state | 49 | 81.07%→75.59% | 6→9 |
| identical | 12 | 100.00%→81.09% | 0→0；新增 2 个未决 |

translation 主决策正确数从 17/54 到 52/54；不能因为总体 precision 失败就丢掉这项收益。
但完整翻译仍必须核对具体对象和状态，不能“主要内容是翻译”就忽略局部记录差异。
13 个截断 payload 全部未决，其中 12 个与参照分歧；另 11 个非截断但 diff packet 达到上限的样本全部未决。
这部分需要改善可见证据覆盖，不能通过把不完整证据强行判为 HIGH 或 YES 来补召回。

### 1. Containment：具体记录被扩大成网站、品牌或话题

22 个 false containment 全部来自同一主 Judge 扩展分支：

- 15 个经 `ASYMMETRIC_CRITIC_CONFIRMED_ATOMIC_EXTENSION` 放行；该分支共 33 条，只有 16 条主决策正确，另有 2 条方向错误。
- 7 个走 `ASYMMETRIC_CRITIC_BENIGN_DID_NOT_OVERRIDE_EXTENSION`；该分支共 12 条，3 条正确、2 条方向错误。

核对可见文本与新旧原响应后的代表例子：

| ID | 可见依据 | 本轮错误机制 |
|---|---|---|
| H0267 | S001–S004 是付款、配送、退换模板；B001 新增鞋履材质，没有共享具体商品身份 | critic 把 All Seasons 商店背景当成已识别商品，放行“同一产品字段扩展” |
| H0278 | S001 是 Anita 指甲油通用介绍；只有 B001 标识 `085 Dubai` | 未先证明两侧同一具体商品，便把首次出现的 SKU 当成允许补全的普通缺失字段 |
| H0253 | 共享馆藏介绍之外，B001–B002 新增多个书目成员 | critic 把列表成员变化当作普通缺失字段，绕过已有 list-membership 边界 |
| H0132 | S001–S003 是 Unimar 公司介绍；A001–A002 是 Utilities 照明类别及技术内容 | “同一公司提供的服务”被直接推成“同一记录扩展”，放开了旧 critic 的产品类别/公司简介边界 |
| H0786 | S001–S002 是镜像站通用介绍；A001 是明确目录路径的 `Index of` 标题 | critic 称其为普通导航；adapter 又因 benign 不覆盖 extension 而留下 containment |

这些是带预测的开发归因，不构成独立盲审。尤其目录标题、产品类别与组织介绍的记录单位必须在新版本中显式固定，
不能靠 case ID 特判，也不能简单规定“所有标题/导航都是冲突”。
`BENIGN` 一律覆盖成双向也不是修复：H0786 按冻结口径需要分离，改为双向仍然是误合并。

### 2. 非正文：补充冲突字段有局部收益，但覆盖不足

最终 NON_MAIN_ONLY/NON_MAIN_ONLY 共 212 条：174 主决策正确、25 误合并、12 已决漏判、1 未决。
25 条误合并全部处于“主 Judge 双向等价 + retained_conflict=NONE”的 89 条分支内。
与 .9 相比该 profile 分组的误合并数同为 25，不能据此断言成员完全相同。

在固定 .12 原输出上仅关闭新的 retained-conflict override，会多出 4 个主决策变化：
H0850、H0537、H0514 从正确 NO/NO 回到误合并；H0995 从未决回到误合并。
因此这层修正了 3 个参照负例、使 1 个不支持的结论保留未决，没有吞掉参照正例；它不是本轮 precision 回退的主要来源。

107 条记录带 retained-proof 校验警告，类别非互斥：缺双侧依据 80、缺实质差异片段 53、packet 不完整 24、无效引用范围 1。
只有 H0995 的警告改变最终主决策，其余 106 条未改变该裁决。不要将不适用分支的警告计为终态 schema 失败，
也不要将它们说成已成功验证的冲突证据。

H0597 的 `Hello BelPatty86 ...` 与 `Hello Robinsyl. ...` 展示另一个遗漏：critic 将受欢迎对象的名字解释为无关署名。
本轮 main 把该 pair 判为 SUBSTANTIVE_MAIN/SUBSTANTIVE_MAIN，非正文补充仲裁根本不会接管。
下一版的身份判断需依据名字在句中的实际功能，不能仅依赖模型先选中的 profile，也不能把所有人名差异都当冲突。

### 3. 完全相同文本被不存在的“独有片段”要求阻断

H0417、H0810 的两侧可见原文完全相同、未截断、packet=COMPLETE、A_ONLY=B_ONLY=0。
主 Judge 判双向等价，critic 也判 benign；但 `_optional_record_binding_critic` 对 benign 统一要求 A/B 独有片段引用，
最终降为 `CRITIC_VERDICT_WITHOUT_UNIQUE_SPAN` / UNRESOLVED。

这是可单独验证的适用性边界，不应靠 prompt 要求模型编造独有片段来消除。
下一版可以对完整原文 exact equality 建立确定性保护；canonical 情形只能使用明确的 canonical 证明，不能仅凭 token diff 没有独有片段。
本轮保持冻结实现，没有事后改写 .12 成绩。

H0679 则是另一个独立问题：共享隐私说明仅多出 `Оплата`，main 将其升级为 PAGE_ROLE，但没有双侧角色冲突依据，最终未决。
需要约束“普通按钮/入口”与“确切页面角色变化”的证据边界，而不是放松现有证据校验。

## 固定输出对照：为什么不整体回滚 critic

所有对照都使用保存的响应，不触发线上调用。最后一行是跨运行拼接的反事实诊断，**不是实际在线候选成绩**：
保留新 main 与新 retained-conflict；仅在新 main 不判 COMPLETE_FAITHFUL 时换用 .9 的 record_binding_verdict。
选择规则不读取标签或 review ID。

| 固定输出解释方式 | 加权 precision | 加权 recall | 误合并 | false containment |
|---|---:|---:|---:|---:|
| .12 main，无 critic | 61.74% | 87.12% | 82 | 43 |
| .12 main + critic，不启用 retained override | 66.48% | 77.59% | 61 | 22 |
| .12 正式 v6 仲裁 | 67.99% | 77.59% | 57 | 22 |
| 跨运行反事实：除完整翻译外换回旧 binding verdict | 76.43% | 65.90% | 32 | 3 |

新 critic 仍在帮助 main 拦截误合并，但不足以达到 precision 门槛；全局恢复旧 binding 又损失过多召回。
下一版应分别处理真实等价、已证明的单记录扩展、模板附着/身份冲突，不应一次性把整个 critic 调松或调紧。

## 之前 22 条扩展争议的追踪

在原封不动的冻结参照上，.12 是 7 条主决策相符、14 条已决漏判、1 条方向错误。
但此前逐条开发复审建议为 17 条分离、1 条真实扩展、4 条待独立复审。
本轮参照相符的 H0019、H0023、H0066、H0168 仍与此前“应分离”的开发建议冲突，不能把这 4 条说成已确认正确的召回修复。
唯一明确真实扩展 H0945 本轮方向正确；4 条不确定样本保留原参照，不删除或改权重。
逐条旧/新/参照方向见 attribution JSON；应另行独立评审，而不是把全 22 条作为 prompt 的必须恢复正例。

## Confidence tiers

本轮没有 HIGH。MEDIUM 970 条，未加权主决策正确率 90.21%、加权 90.77%；LOW 30 条，分别为 3.33%、4.58%。
这里是相对开发参照的经验正确率，不是校准后的概率；不能据此承诺后续样本有相同可靠性。

## 下一版建议顺序

1. **先做确定性小修的离线隔离实验**：解决完整 exact 被 unique-span 要求阻断的问题；保留完整非正文翻译路由。
   不以 token diff 零差异推断全文完全相同，不动截断 fail-closed 边界。旧版本结果和缓存保持不可变。
2. **先明确记录单位，再收紧扩展证明**：要求双侧已有具体记录锚点；品牌/公司/站点/主题相同不足以绑定产品、文章或目录。
   把缺失普通字段、首次引入具体 SKU、冲突字段值、列表成员变化分开。组织使命/方法的真实扩展继续保留正例。
3. **分别证明等价与扩展**：benign 不能自动确认 containment，也不能一律转双向。
   要求“差异被语义覆盖/纯 chrome”的证据与“同一记录新增实质内容”的证据分别成立；保护实际身份、服务对象和页面角色，
   同时加入按钮、导航、完整翻译的对照，避免过度否决。
4. **同输出消融后再在线验证候选**：必须同时看 38 负例、4 containment、5 benign、全体翻译及完整 1,000 条分布。
   先确认哪个局部变更恢复保护而不吞掉召回，再使用新版本/新 run root 做相同技术预检和完整开发评估。
   不因为某个困难子集或 F1 提高就跳过 75%/75% 门槛。

本轮到诊断和建议为止，没有实现下一版、重打参照标签、读取 holdout、启动 20,000 对或标记 release。
SUT 原始 outcome 只作独立诊断；没有取得真实 resolved MinHash contract，也没有用 evaluation retriever 参数或 LLM 猜测补全 SUT。

## 审计产物与复现

- 冻结协议：[v06212_online_development_protocol.md](v06212_online_development_protocol.md)。
- 总体指标、分区和完整交叉矩阵：[v06212_online_development_summary.json](v06212_online_development_summary.json)。
- 正式开发门槛对照：[v06212_online_development_comparison.json](v06212_online_development_comparison.json)。
- 固定输出归因、保护集、22 条追踪和原响应摘要：[v06212_online_development_attribution.json](v06212_online_development_attribution.json)。
- 124 条主决策分歧队列：[v06212_online_development_errors.csv](v06212_online_development_errors.csv)。
- 完整 1,000 行展开表：run root 的 `reports/development_rows.csv`；正式原结果：`data/judge_results.jsonl`。
- 预检：`hub_preflight.json`、`preflight/pilot-57836aef/terminal_results.jsonl`；执行冻结：`execution_freeze.json`。

关键 SHA-256：

| 产物 | 摘要 |
|---|---|
| 冻结标签 | `6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096` |
| .12 实现 | `967dfb931e109a3bbaa89782fbec9b9f8f0f4bc70a1946ba185193d7028d6c6e` |
| .12 contract | `3b9adb66def652f5831569e527c64779555e47d4ca809083052ee40697270f56` |
| .12 正式结果 | `5f09493cc16958953b9123cdc268ed7809b653ee7308c7fed3edceec851f3b68` |
| .9 正式结果 | `a238e986acdba2f9b29a83cd81c96e7fc76f1750321560d715942aea049481d4` |

以下命令只读已保存结果并重新生成开发报告，不调用模型；在已有 NeMo Curator 环境执行：

```bash
python -m eval.dedup.analysis.development_diagnostic \
  --run-root /raid/hfang/ihb/runs/v0.6.2.12-full-development \
  --labels eval/dedup/analysis/v0628_policy_reconciled_labels_1000.csv \
  --original-labels eval/dedup/analysis/hs_blind_adjudication_1000.csv \
  --residual-subset eval/dedup/analysis/v0622_reconciled_residual_candidates.csv \
  --guard-replay eval/dedup/analysis/v06212_translation_route_replay.json \
  --output eval/dedup/analysis/v06212_online_development_summary.json \
  --rows /raid/hfang/ihb/runs/v0.6.2.12-full-development/reports/development_rows.csv \
  --errors eval/dedup/analysis/v06212_online_development_errors.csv

python -m eval.dedup.analysis.judge_calibration \
  --labels eval/dedup/analysis/v0628_policy_reconciled_labels_1000.csv \
  --baseline-run-root /raid/hfang/ihb/runs/v0.6.2.9-full-development-diagnostic \
  --candidate v06212=/raid/hfang/ihb/runs/v0.6.2.12-full-development \
  --output eval/dedup/analysis/v06212_online_development_comparison.json
```
