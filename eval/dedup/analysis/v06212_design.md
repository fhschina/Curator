# V0.6.2.12：完整消息翻译与有证据的非正文冲突

## 范围

从冻结的 .9 行为分支实现，保留 .9、.10、.11 资源和历史结果，不以 .11 为基础继续叠加宽泛 policy veto。
本轮只有本地实现、开发复审、合成测试和旧响应离线重放；不运行在线模型、holdout 或 20,000 对，
不修改参照，不授予 release approval。

两个新注册版本均沿用 public output v3 / visible payload v3，未来在线执行必须使用自己的 run root、
contract digest 和空缓存：

| 版本／CLI policy | 主／critic 资源 | 适配器 |
|---|---|---|
| `dedup-judge-hs-v0.6.2.12-dev-route` / `hs-v06212-route` | .9 原 main、critic、rubric 原封不动 | `v6-route`：只修改完整翻译路由及其 delta 一致性检查 |
| `dedup-judge-hs-v0.6.2.12` / `hs-v06212` | 新 system、pair、YAML 同步描述 | `v6`：翻译修复 + .9 实质记录仲裁 + 窄非正文冲突证据 |

默认版本不变。模型、温度、推理预算和并发默认配置与 .9 一致；未通过换模型或加一个模型阶段改变比较条件。
Runner 的 3,072 默认输出预算沿用 .9 YAML；未来与本次 .9 完整开发运行作受控比较时，需要同时沿用该运行 manifest
中的实际 4,096 输出预算、20,000 可见 token 等设置，不能把 runner 默认值冒充已执行配置。

## 1. 版本化翻译路由

旧 `_v3_span_ledger_target` 的默认行为保留。只有 `v6-route` / `v6` 开启：

- SUBSTANTIVE_MAIN + VERIFIED_SUBSTANTIVE_RECORD，或 NON_MAIN_ONLY + VERIFIED_EQUIVALENT_NON_MAIN_MESSAGE，
  均可在 COMPLETE_FAITHFUL 下得到 yes/yes、NEAR_SURFACE、material NONE。
- 两侧 delta 必须为 NONE、SEMANTICALLY_COVERED 或无害 UNIVERSAL_UI_OR_REPETITION；不能用翻译标记覆盖新增事实。
- 仍先检查可读性、完整 packet、精确双侧引用、共享身份、profile mismatch 与 hard conflict。
- 未知／截断／未决证据不获救；完整非正文消息永不形成 containment；边界 confidence 最高 MEDIUM。

## 2. 记录归属复审与 critic prompt

完整逐条意见见 [22 条开发复审](v06212_extension_review.md)：17 条建议分离、1 条支持同记录扩展、4 条仍不确定。
这不是独立人工盲审，建议不写回标签，也不参与本轮重新计分。

新 critic 明确先区分双向等价、单向扩展、实质分离三种证明任务。A-only/B-only 是词面差异，不能被当作语义损失；
“不能证明 containment”不等于“不能证明等价”。保留原 `record_binding_verdict` 枚举，并同步扩展 benign 的描述，
使完整翻译／改写的语义覆盖具有明确合法分支。

新增成对边界：具名公司介绍 + 本公司的使命／方法 vs 网站介绍 + 独立新闻；同一 FAQ + 排障步骤 vs 共享配送说明 +
未绑定产品规格；明确同诊所的团队字段 vs 独立人物记录；完整消息翻译 vs 实际新增目的／权限。
对那 4 条仍模糊的署名、套餐标签、团队归属、空评论 widget，不加入猜测页面结构的硬规则。

正文仍复用 .9 的非对称仲裁，不批量撤销 separate-record veto，不让 benign score 抹掉已经举证的真实内容扩展。

## 3. 非正文冲突：窄例外，不是任何措辞变化都否决

新 critic 在同一次调用中增加 `retained_conflict`，不用新模型阶段。可选择 NONE、NOT_APPLICABLE、UNRESOLVED，或：

- IDENTITY_OR_SERVICE_TARGET_CHANGE
- PAGE_ROLE_CHANGE
- POLICY_PERMISSION_CHANGE
- STATE_CHANGE
- MEMBERSHIP_CHANGE

仅当主 Judge 已判两个 NON_MAIN_ONLY 页面等价时，这个字段有权改变决策。实质正文的 record veto 与已有主 Judge
no/no、UNRESOLVED 不被该字段重开。

材料冲突必须有自己的双侧上下文（S，或 A+B）、决定性 unique span、明确保留意义及后果，且与通用 critic 的负面
verdict 一致。有效证据产生 no/no、MAJOR；缺证据或两种 critic score 自相矛盾则 UNRESOLVED，不制造已证明的负例。
NONE／NOT_APPLICABLE 下，不能仅凭通用 critic 的笼统警告推翻非正文等价。

明确区分实际服务对象／产品标题／权限／页面功能与普通 Accept、Settings、Wishlist、导航、语言和空格差异。
不能把按钮出现解释为实际同意状态变化。适配器能验证字段、引用、分支和输出一致性，**不能确定性证明 LLM 的语义解释正确**；
因此此新 proof 规则的真实精确率仍须未来在线评估，而不是靠合成测试宣称已解决 25 个误合并。

历史 .9 critic 没有 `retained_conflict`。最终 `v6` 合约会拒绝缺失字段，不能凭空填 NONE 来声称已经重放新 critic。

## 4. 隔离重放与防回退

使用 .9 完整开发 run 的 1,000 条最终原始响应，按 pair ID + provider response digest 匹配，而不是取最后一个文件。
先经旧 `v3` 仲裁验证所有公开字段仍与原结果相同，再仅使用 `v6-route` 重放。
结果、payload、参照和原响应均有摘要校验；输出保存在独立 offline artifact，不是 judge cache。

固定参照仍为 `v0628_policy_reconciled_labels_1000.csv`，SHA-256
`6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`。
重放摘要 `v06212_translation_route_replay.json` 包含变化 ID、指标、来源摘要和 38 条 critic-repaired negative guards。

未来完整开发诊断可用 `development_diagnostic --guard-replay <该摘要>` 自动检查这 38 条。
缺失、UNRESOLVED 或方向不匹配都算防回退失败，不能缩小分母。也须继续报告真实 containment、chrome、translation、
127/873 分区及未修改／已修改参照分区，不能只报告这 38 条。

```bash
python -m eval.dedup.analysis.replay_translation \
  --source-run-root /raid/hfang/ihb/runs/v0.6.2.9-full-development-diagnostic \
  --labels eval/dedup/analysis/v0628_policy_reconciled_labels_1000.csv \
  --output <new-summary.json> \
  --predictions <new-offline-predictions.json>
```

新版本选择仍需完整开发集加权 precision/recall 均 >=75%、主决策一致率 >=79%，以及原有错误簇、防回退、
schema 和重试门槛。仅离线重放通过不能晋级、不能打开 holdout、不能宣布 prompt 改进。
