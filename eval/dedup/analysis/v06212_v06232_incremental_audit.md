# .12 → .32 增量退化账本与独立 X 小实验冻结

2026-09-11。使用保存输出离线重放，**新增模型调用 0、改标签 0、没有新 Judge 版本、没有 release**。
这是同一冻结开发参照上的增量分析，不是独立人工真值或同时运行的单变量实验。

## 结果：收益没有抵消新增退化

完整 1,000 条原文 payload 逐对象相等，参考 SHA-256 为
`6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`，原分层总权重 10,023。
`.12` 主输出、critic 及 v6 仲裁按保存响应摘要重放；main-only 用保留主路由的 v6-route，
不是伪装成完整 v6。`.32` 使用冻结运行的严格原始输出重放，三个终态失败仍在分母中。
主决策只比较同组与两个替代方向，不用历史翻译 MINOR taxonomy 决定胜负。

| .12 最终 → .32 最终 | 条数 | 原分层权重 |
| --- | ---: | ---: |
| 两版都与参照一致 | 724 | 7,335.54 |
| 原不一致 → 一致 | 47 | 445.15 |
| 原一致 → 不一致 | 152 | 1,512.99 |
| 两版均不一致 | 77 | 729.32 |

152 个新增分歧包括：121 误合并、18 已决漏判、9 方向错误、4 未决/失败。
121 个新增误合并中，111 是 containment，10 是 near-surface。
77 个持续错误中有 15 个主三元组发生变化；“持续错误”不等于输出没变。

加权主一致率净下降 **10.65pp**。对应加权 P/R：`.12` **67.99% / 77.59%**，
`.32` **42.90% / 80.16%**。分组混淆量的变化可精确对账：

- 新增 FP 权重 1,273.68，移除 FP 权重 223.90，净增加 **1,049.78**。
- 恢复 TP 权重 214.49，损失 TP 权重 175.99，净增加 **38.50**，FN 同量减少。
- 这些是混淆量，不是能相加的 precision 百分点修复收益。

这里的“新增 FP”还包括 1 条从旧未决变为误合并的持续错误；它不在“旧正确→误合并”的 121 条里。

## 新增退化发生在哪一段

| 可观察的组件变化 | 新增分歧 | 权重 | 其中误合并 |
| --- | ---: | ---: | ---: |
| .12 主/最终正确 → .32 主/最终错误 | 132 | 1,300.75 | 101 |
| .12 主错误、后置修正 → .32 主错误、最终未修正 | 20 | 212.24 | 20 |
| .32 主正确、被新后置改错 | 0 | 0 | 0 |

132 条包含 3 个明确工程终态失败 H0104、H0330、H0560，以及 H0080 的有效未决。
不能把这些都称为语义错误，也不能把未决一概算成工程失败。

因此不能把本次下滑简单归为“critic 否决过多”：新 critic 没有把主正确改错，
但丢失了旧链路曾经提供的 20 条负例修正；大部分新增分歧已经出现在主阶段。
这只是观察到的位置，不证明哪一条 prompt 或工程修改导致变化。

同主输出内的后置组件比较也保留：`.12` 加权主决策净收益 +1.17pp，`.32` +0.29pp。
旧 critic 的收益伴随真实 TP 损失；不能仅根据 20 个丢失的负例修正就恢复旧 critic 全部 veto。

## 哪些边界值得先查

按旧参考原因划分 121 个新增误合并：

| 旧参考类别 | 条数 | 权重 |
| --- | ---: | ---: |
| meaningful_addition | 48 | 493.75 |
| boilerplate_only | 37 | 413.56 |
| page_role | 14 | 156.65 |
| identity_slot | 9 | 80.26 |
| material_content | 9 | 89.62 |
| factual_state | 3 | 19.74 |
| list_membership | 1 | 10.05 |

**参考原因是抽样/排查入口，不是本轮已经确认的错误原因。** 121 条在 `.12` 中最终正确，
全部不在旧 124 条错误复核清单里。因此不能把旧审计的语义错误/争议百分比直接套到这些新增错误。
账本对新输出的原因明确保留 `NOT_ADJUDICATED_FOR_NEW_OUTPUT`，原复核说明另列，不继承成新结论。

全量同时保留 63 条历史争议/待核标记、19 条复合政策范围标记及两组共 4 条相同原文参照矛盾。
这些标记相互重叠，且不覆盖所有潜在争议；不从总体评分中删掉任何一条。
19 条复合政策范围中 **16 条由不一致变一致，3 条仍不一致**。这支持保护潜在收益，
不证明这 16 条已由独立评审确认，也不把差值全部归因于“独立 Y”规则。

查看预测后的原文抽查（不是盲审，未改参考）：

- H0360：A 仅 cookie/隐私提示，B 加入 Child Heroes 项目介绍。旧主/最终均分离，新主/最终均 containment。
  适合验证模型是否把共享 cookie 消息升成实质 X。
- H0266：共享 Currys 信贷/免责声明，B 加安防摄像头正文。旧主/最终分离，新主/最终 containment。
  需要把“可重复的信贷页脚”与“独立实质政策正文”讲清并给出证据，不能凭包含具体数字就自动批准 X。
- H0419：旧主 containment，被旧后置修正；新主与最终又回到 containment。代表“原有负例修正丢失”。
- H0119：相反，它在 `.32` 已从旧误合并变为正确分离，应进入**修复保护**，不能继续列为新版本错误。
- H0521/H0822：具体闭馆通知/Top Drawer 申请保留，另一侧增加独立馆藏介绍/机会。
  两条现在符合旧参照方向，也符合已批准复合规则的开发复核，应保护，不恢复旧的单记录限制。

## 冻结的小实验：独立判断实质 X，不接入自动 veto

[机器协议](independent_x_probe_v1.json)、[完整 system prompt](independent_x_probe_v1_system.txt)、
[子任务输出 schema](independent_x_probe_v1_schema.json)已保存。冻结摘要：
`9afd5e6e5be16afbfd629a0cc6a44cff1b21daa28993bcdde00e9cfd2946685c`。

| 选样用途 | 条数 |
| --- | ---: |
| 可疑模板 containment | 12 |
| .32 已修复的负例保护 | 4 |
| 参考 containment 保护候选 | 8 |
| 完整翻译保护候选 | 8 |
| chrome/非正文等价保护候选 | 8 |
| 身份/页面角色边界 | 8 |

共 48 个唯一的无序原文对，均保留完整输入，不为适配长度裁剪文本。
先固定机制样例，其余按原权重降序和 review_id 选取；具体选择及原因只在 private 文件。
这是看过历史预测后选择的开发面板，不是代表集、holdout 或当前 X 标签真值。
其中 6 条带历史争议/政策复核标记，全部保留可追踪；“保护候选”不等于其 X 已被确认 PRESENT。

唯一子任务：`shared_substantive_x = PRESENT / ABSENT / UNRESOLVED`，附双侧内容属性、精确字符证据及短理由。
不提供主 Judge 预测或理由，不要求完整 duplicate、direction 或 relation 判断。
ABSENT **不能派生 no/no**，因为等价 cookie-only 页面仍可能双向替代；PRESENT 也不证明没有实际冲突。

在任何在线执行之前，要另行确认这 48 条的 X 标注并冻结，而不是从旧同组标签反推。
已导出不带答案/抽样原因的原文包及空白标注表，当前 **没有独立 X gold**。
评审需记录人/AI、预测接触、证据与分歧；预测可见 AI 开发意见不冒充人工盲审。

后续调用设计为同现用模型、温度 0、两轮各 48 条 fresh 响应，另 4 条技术预检，基础 100 次调用。
两轮都报告，计 48 个配对样本而非 96 个独立样本；全 HTTP 尝试预算 256，沿用有界限流与传输退避。
正式执行尚未授权；运行器/纠错模板/实际 token 边界验证须在开跑前完成独立执行冻结，不能把本次设计冻结当作已运行。

预设的子任务筛选条件：标注先冻结、schema/证据全部有效；每轮确认 ABSENT 的目标检测率 ≥80%，
确认 PRESENT 的真实包含保护不漏判，48 条 X 判断重复翻转不超过 2 条。失败/未决不从正确率分母消失。
这些只是子任务可行性筛选，不替代完整集 75/75/79、原保护门槛或 holdout。
若子任务仍错，不接 gate；若可行，再设计单独的同口径全链路对照。现在不调 critic，不更换模型。

## 产物与复现

运行目录：`/raid/hfang/ihb/runs/v06212-v06232-incremental-v1`。

- [全量逐条账本](/raid/hfang/ihb/runs/v06212-v06232-incremental-v1/ledger_1000.csv)
- [152 条退化](/raid/hfang/ihb/runs/v06212-v06232-incremental-v1/regressed.csv)、
  [47 条改善](/raid/hfang/ihb/runs/v06212-v06232-incremental-v1/improved.csv)、
  [77 条持续错误](/raid/hfang/ihb/runs/v06212-v06232-incremental-v1/persistent_error.csv)
- [加权统计与四路组件比较](/raid/hfang/ihb/runs/v06212-v06232-incremental-v1/summary.json)
- [48 条仅原文材料](/raid/hfang/ihb/runs/v06212-v06232-incremental-v1/probe/inputs_blind.jsonl)、
  [空白 X 标注表](/raid/hfang/ihb/runs/v06212-v06232-incremental-v1/probe/annotation_template.json)
- `probe/requests_blind.jsonl` 保存全部渲染请求；`private_selection.json` 不给评审或模型。
  `review_queue_blind.jsonl` 另覆盖所有 276 条非稳定正确项，原因和 ID 映射只在 private 文件。

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup.analysis.incremental_regression \
  --output /raid/hfang/ihb/runs/v06212-v06232-incremental-v1
```

同内容允许重放，不同内容拒绝覆盖。原 .12/.32 run、cache、参考、配置和 prompt 未变。
单元测试覆盖精确 join、四路归因、权重闭合、终态分母、历史复核作用域、匿名导出、机制/完整性过滤、
不可变冻结、非正文 ABSENT 不自动变 no/no、多语字符证据对齐。

验证结果：

- 新模块 25 项加已有瓶颈/重放相关测试，共 **48 passed**；
  日志 `/raid/hfang/ihb/incremental-regression-targeted-tests.xml`。
- 最终全套为 **1,041 passed、1 failed、50 skipped、2 GPU deselected**。失败仍是旧
  `test_actual_parallel_requests_are_spaced_and_admission_persists_across_stages` 的 10ms 级服务端到达间隔断言，
  与 `.32` 审计已记录的波动相同；本轮没有修改 relay 或该测试，也没有将全套报告为通过。
  日志 `/raid/hfang/ihb/incremental-regression-final-cpu-tests.xml`；之前少一项新增保护测试时的
  1,041 项全绿运行日志同样保留在 `incremental-regression-cpu-tests.xml`。
- 正式离线账本再次完整重放成功，产物及冻结摘要完全一致；40 个输入/来源文件和 12 个输出文件摘要逐项通过。
- Ruff lint/format、`git diff --check` 及新增文件的末尾换行/空白/大小/私钥标记检查通过。
  现有环境和 PATH 没有 `pre-commit`，完整 hooks 未运行；遵守环境约束，未安装或创建其他环境。
- 没有改 `.12/.32` 历史文件、没有 commit/push、没有读取 holdout、没有 20,000 对运行。
