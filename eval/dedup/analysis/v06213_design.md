# V0.6.2.13：具体记录范围与完整等价证明

## 范围与版本

用户授权：修复后先局部评估，再重新评估。新版本保持 public output v3 / visible payload v3、同 Qwen 模型、
同温度与 payload 设置；不改冻结参照、不迁移 cache，不读取 holdout、不运行 20,000 对、不自动 release 或提交代码。
所有旧资源和结果保留。共享 adapter 增加版本分支，旧 .9/v3 与 .12/v6 的全部 1,000 条 public 结果已逐字段重放一致。
新 runner 实现摘要自然不同，不能拿新代码恢复旧在线 run，也不能事后改旧 manifest。

- `hs-v06213-exact` / `dedup-judge-hs-v0.6.2.13-dev-exact`：.12 原 system/pair/rubric 不变，只启用 `v7-exact`。
- `hs-v06213` / `dedup-judge-hs-v0.6.2.13`：新 main/critic system、pair、YAML 同步收紧，使用 `v7`。
- exact-control 仅使用保存的 .12 原响应离线评估；不为它重复发起在线调用。

## 代码修复与语义边界

1. Exact 保护：主 ledger 已合法判 yes/yes，完整原文字面相同、非空、明确未截断且 packet=COMPLETE 时，
   独有片段要求不再允许 critic 将其降成 UNRESOLVED。仅 token diff 的 A_ONLY=B_ONLY=0 不够；
   不把 canonical、空文本、截断或主 ledger 的冲突/未决强行升级为 exact。
2. 新 critic `record_scope` 在 rubric 中先于 binding verdict：分别识别同一具体记录、同一组织自身介绍、
   完整等价消息、仅通用上下文、不同记录、列表成员变化、未决/不适用。每个活跃分支需要自己的双侧引用；
   分离还要有实际 unique delta，不能靠没写内容推测冲突。
3. 同品牌、站点、公司所有权、商品系列或主题，不足以绑定某个 SKU/文章/产品类别/目录/列表。
   普通缺失字段必须在其他共享证据已经证明同一具体记录之后才算扩展；首次引入记录身份、实际列表成员变化另行处理。
   组织历史、使命、编辑方法和明确归属的组织字段保留扩展正例；独立产品/服务类别不能因为同一所有者而混为组织字段。
4. 新 scope 的已举证分离且 binding verdict 同意时，允许否决主 Judge 的正例，包括被错分成正文或非正文的消息。
   缺证据/意见不一致则未决，不制造有证明的负例；主 Judge 既有 no/no 和 unresolved 不被重开。
5. containment + benign 不再无条件保留 containment，也不无条件改双向：需要正向 scope、binding 引用覆盖全部独有片段、
   无 retained conflict，才可重分类双向。否则未决。真实 extension 仍需独立记录身份和扩展侧证据。
6. 保留 .12 完整正文/非正文翻译路由和原有证据安全门槛。名字按句中功能区分收件人和署名；目录标题与普通下载导航区分；
   Payment/Settings 等孤立按钮不升级成角色/状态冲突。大段翻译仍逐一核对局部身份、服务对象、日期、权限和成员。

适配器验证字段、引用、范围覆盖与决策一致性，不能确定性证明模型对“具体记录”的语义解释正确。
新 scope 不存在于 .12 原响应中，不能凭空填值后声称离线测试了新 prompt。

## 评估顺序与预先固定的停止条件

1. 合成回归：测试 exact 适用性、旧策略兼容、字段缺失/冲突、实际成员/模板附着、真实扩展、全部 delta 覆盖、完整翻译与新版本注册。
   当前 dedup 全套 370 条通过。
2. `.12` 原响应 exact-only 离线重放：主决策仅修正 H0417、H0810，无其他主决策变化；误合并仍 57，负例保护仍 20/38。
   加权 precision/recall 为 68.30%/78.73%，不代表新 prompt 的效果。见 `v06213_exact_replay.json`。
3. 冻结 258 条局部集：旧 127 困难集、全部 54 translation、12 identical、6 canonical、39 参照 containment、
   38 旧 critic 负例保护和 .12 全部 57 误合并的并集。重叠去重，保持原标签和权重，仅投影计分列。
   来源与原因固定于 `v06213_local_selection.json`；标签在 `v06213_local_labels.csv`。
   此集合刻意富集错误，不能拿它的 precision/recall 代替完整开发成绩。
4. 局部 run：先独立 20 条技术预检，再评估完整 258 条，预检不复用为正式预测。
   同时展示 .9、.12 在同一局部集的已有结果；在线 main/critic 必须使用新字段和新 contract。
5. 局部进入完整诊断的条件：100% 有效、0 终态错误、外层样本重试 <=1%；12 exact 全对；4 条真实 containment
   保持正确；.12 原本保护住的 20 条负例和 3 条 benign 主决策不新增回退；translation 加权主决策一致率
   相对 .12 降幅不超过 3 个百分点。全部 38 负例和 5 benign 是否恢复另外完整报告，不能缩小最终保护分母。
   若出现这些局部回退，先停止更大规模调用并分析，不在已调用的 .13 上改 prompt；任何再修复须新实验版本。
6. 局部未触发上述停止条件后，以相同冻结实现、新 full run root、空缓存重新评估全部 1,000 条。
   局部预测不灌入 full cache。该步骤是用户授权的开发诊断，不代表局部或完整质量准入已通过。
7. 完整集仍要求 weighted precision/recall >=75%、weighted primary >=79%、over-group <=66、三类关键 cohort
   相对 .9 加权准确率下降 <=3 pp、schema=100%、外层重试 <=1%，并检查 38/38 负例、4/4 containment、5/5 benign。
   单列 127/873、translation、identity、meaningful-addition、非正文、page-role、truncation、置信档、HTTP 429/内部调用。
   原 22 条争议仍不改标签，不用其 AI 复审建议提高分数；不使用 proxy、DeepSeek 或 SUT agreement 选择版本。

## 运行冻结

- 局部 root：`/raid/hfang/ihb/runs/v0.6.2.13-local-development`。
- 完整 root：`/raid/hfang/ihb/runs/v0.6.2.13-full-development`，仅局部条件满足后提交调用。
- 实际模型 `nvidia/qwen/qwen3.8-27b`；temperature=0，top_p=1，输出 4096，可见 token 20000，窗口 4096/512，
  timeout=600，并发=64，max_retries=2。完整集两块 500，局部单块 258。
- 开始调用前固定 implementation/resource/contract/payload/labels 摘要；调用开始后不修改。
- 完整参照 SHA-256：`6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`。
- 这些参照含既有 prediction-aware AI 修订，是开发集而非新人工盲审。本轮不改变该局限，也不放宽晋级条件。
