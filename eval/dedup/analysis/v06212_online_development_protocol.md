# V0.6.2.12 在线开发评估协议

## 授权与冻结

用户在 .12 本地实现之后明确授权：先做 20 条在线技术预检，正常后完成全部 1,000 条在线开发评估，再根据结果决定是否迭代。
这是开发诊断授权，不是 holdout、20,000 对运行、release 或修改参照的授权。之前设计文档的“未在线运行”记录的是当时状态。

- 版本：`dedup-judge-hs-v0.6.2.12`，新 main/critic 资源和 `v6` 仲裁，public/visible contract 均为 v3。
- Run root：`/raid/hfang/ihb/runs/v0.6.2.12-full-development`，全新缓存，不复制 .9 或 route-only 预测。
- 冻结参照：`v0628_policy_reconciled_labels_1000.csv`，SHA-256
  `6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`。
- 比较对象：`.9-full-development-diagnostic` 实测输出，以及 `v06212_translation_route_replay.json` 的代码修复离线对照。
  前者是历史同设置运行而非同时随机对照；后者不是新的在线模型输出，须分别说明。
- 模型 `nvidia/qwen/qwen3.8-27b`；temperature=0，top_p=1，max_output_tokens=4096，max_visible_tokens=20000，
  window_tokens=4096，window_overlap_tokens=512，max_parallel_requests=64，timeout=600，max_retries=2；每块 500 对。
- 准备完成后验证 .9/.12 全部 1,000 个 pair 的可见 payload 完全相同。代码、prompt、资源、参数、payload 和参照摘要记录于 manifest。
  开始调用后不修改这些内容；不根据中途语义结果调 prompt。

## 执行顺序

1. `prepare` 新 root，确认 1,000 对完整性、可见 payload 一致、无历史缓存。
2. `preflight --pairs 20`：使用准备后固定候选顺序的前 20 对，不按版本预测或参照正确率挑选。
   在同一模型和合约下调用新 main/critic，保存原始响应及独立 preflight terminal records。20 对全部 schema 有效、
   无终态错误且引用可对齐才继续；单列重试和 UNRESOLVED，不把小样本语义准确率当作版本晋级门槛。
   预检不进入正式结果缓存，完整运行重新评估全部 1,000 对，因此预检调用另外计数。
3. 技术预检通过后运行两块 500 对。复用通过预检的同一合约，但不复用其预测。
   任何终态失败按现有有界重试流程处理；不为拿到完整结果而放宽 schema 或证据要求。
4. 完成后统一计算质量，不使用途中样本成绩调整后续执行。

## 预先固定的报告与判定

- 加权和未加权 precision、recall、主决策完全一致率、F1、误合并、false containment、漏判、方向错误、UNRESOLVED。
  参照重复上的 UNRESOLVED 仍计为召回漏判，主决策分歧互斥计数；历史 translation MINOR 只作为 taxonomy 背景。
- 127/873 分区；未修改/已修改参照；translation、meaningful_addition、identity、chrome、非正文、page-role、truncation；
  置信档经验正确率；schema completion、终态错误、样本级外层重试，以及内部 HTTP 状态事件分别计数。
- 自动检查 .9 critic 修正的 38 条负例：必须仍为相同 NO/NO 主决策；缺失、未决也为保护失败。
- 单列原有 4 条真实 containment 保护样本、5 条 benign 正例，以及整批参照 containment 的方向与召回，避免窄保护集掩盖问题。
- 单列已复审的 22 条内容扩展争议，但不按 AI 建议重写或删除参照来提高分数。整个集合包含先前 prediction-aware AI 复审，
  不是独立人工盲审或 holdout。
- 完整开发准入仍要求 weighted precision/recall 均 >=75%、primary exact >=79%、误合并 <=66、identity/
  meaningful_addition/translation 主决策准确率相对 .9 不下降超过 3 个百分点、schema completion=100%、样本级重试率 <=1%，
  并报告全部保护检查是否通过。预检通过只授权继续开发评估，不能代替这些质量条件。
- 即使开发条件全部满足，本轮也只提出下一步 holdout 建议，不读取 holdout、不写 release approval。

若质量未通过，保留 .12 实验结果及失败门槛，按新错误分布建议下一版，不在这个版本上边评估边改写。
