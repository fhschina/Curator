# V0.6.2.14：仲裁归属与可执行内容边界

## 固定范围

本轮只做开发修复、离线归因和局部在线评估；局部检查全部通过才运行完整 1,000 条。
不改任何旧版本 prompt、历史输出或标签，不复用旧缓存，不读 holdout，不跑 20,000，不 release、commit 或 push。
参照仍含 prediction-aware AI 修订，不宣称是新的独立人工盲审。

- `hs-v06214-arbitration`：.13 的 main/critic system、pair、rubric 原样保留，仅用新 `v8` 仲裁离线重放。
- `hs-v06214`：同一 v8 加针对性边界示例；不新增模型输出字段。
- 旧 v7 保持不变；重放逐字段验证已发布结果和双侧 evidence offsets。

## 修复

1. scope 不得把已合法成立的 no/no 因补充字段缺引用降成未决。
2. 两侧 non-main 的判断仍归 retained-conflict 分支：真实且独立举证的身份/角色/权限变化可以否决，
   generic scope 不能越权把 Category、Payment 等孤立标签变成冲突。
3. containment 与 benign 意见相反时，不再把“全部 span ID 都出现”当成另一侧语义覆盖证明。
   保留原来的单向决定并降为 LOW；不据此恢复原本未决的判断。
4. 已有主 ledger 支持的部分/增量翻译，scope 证明同记录且 critic 同意 extension 时，保留原方向。
   这不是合并不同字段引用来伪造 binding 证明；常规非翻译缺证明仍按旧门槛处理。
5. main/critic system、pair 和 YAML 同步加入：申请目的地/邮件主题/FAQ 操作不是旁边的 UI；
   作者简介与独立文章、活动与组织者新闻、博客规则与政治正文是不同内容单位；
   实际注册/权限要求与可选按钮分开，通用分类/作者 teaser 不推测成记录。

## 冻结协议（候选输出查看前）

- 固定同一 258 对：使用 `v06213_local_evaluation_labels.csv`，其权重取原始 stratum population/sample。
  保留旧 selection 文件不动；新分析器逐对验证语义标签、cohort、原权重完全等于 1,000 条冻结参照。
- 原参照 SHA-256：`6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`。
- 基线始终为 .12 完整开发输出在同一局部集的投影；另列 .13。局部是错误富集集，不等于完整集质量。
- 局部准入维持 .13 预先规定的底线：258/258 有效、0 终态、outer retry <=1%；12 exact 全对；
  4 条真实 containment 方向全对；.12 保住的 20 负例和 3 benign 无回退；54 translation 的加权主决策
  相对 .12 不下降超过 3 pp。同时报告全部 38 负例、5 benign，不减少最终分母。
- 离线通过不能替代在线候选通过。先独立 20 对 preflight，然后完整 258 对新调用；预检不灌入正式 cache。
- 同 Qwen 模型 `nvidia/qwen/qwen3.8-27b`，temperature=0、top_p=1、输出 4096、可见 20000、
  窗口 4096/512、timeout=600、并发 64、max_retries=2。调用前冻结源码、资源、配置、payload 和标签摘要。
- 新 local root `/raid/hfang/ihb/runs/v0.6.2.14-local-development`；完整 root 如需启用另建 .14-full-development。
  调用中不编辑推理代码/资源。若局部失败，本版本停止更大规模调用；不可事后放宽门槛。
- 完整集质量标准不变：weighted P/R >=75%、weighted primary >=79%、over-group <=66；identity、
  meaningful-addition、translation 相对 .9 不回退超过 3 pp；schema 100%、outer retry <=1%，
  全部 38 负例、4 containment、5 benign 保护。通过完整开发也不自动授权 holdout 或发布。
