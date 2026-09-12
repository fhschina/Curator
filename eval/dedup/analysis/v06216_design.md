# V0.6.2.16：固定主输出的单项 critic 重复诊断

## 冻结边界

这是开发诊断，不是正式候选、不生成 release approval，也不把混合响应写入 judge cache。
保留 .14 主输出、v8 仲裁和原始可见 payload；不采用 .15 critic 重写或 v9 仲裁。
不改原标签、权重、历史资源或结果；不调用主 Judge、不评估 holdout、不跑完整 1,000/20,000、不 commit/push。

三个实验臂都只有原名 qwen_dedup_record_binding_critic，字段与 options 不变：

1. control：.14 critic 原 system、pair、rubric 逐字不变。
2. consent：仅增加完整同意条件的双侧对应。两侧都有继续浏览即同意时，额外 Accept 不推出 explicit-only；
   明示的启用条件、准入、目的、受众、权限等仍保留。同步 system/pair/rubric。
3. pointer：仅增加短指针的可见内容边界，不臆测未展示的文章或人物事实；实际 biography、成员、服务对象、
   控制者、目录和 badge-for-user 功能仍保留。同步 system/pair/rubric。

保留全部 .14 独立产品、SKU、事件、文章、政治评论、列表、身份与证据边界示例。没有按 ID 路由或词匹配代码。
两项修改不叠加；仅某项重复诊断稳定通过，才值得下一步另立合同独立验证，不自动放大调用。

## 抽样、原输出与请求

- 同一冻结 258 对；selection SHA-256 `977eb95ba066655d03ff533ba98d5c5fe92fa3cb3f63148856159f39d09c1f2a`。
- 1,000 条参照 SHA-256 `6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`；按原分层权重计分。
  这是错误富集开发集，含历史 prediction-aware AI 修订，不是独立人工盲审。争议仍保留原分母。
- .14 main 按已发布结果的 provider digest 匹配，离线验证历史 v8 public 输出逐字段相同后冻结。
- 输入只含 pair ID、payload 摘要、可见 payload、初始空 repair_feedback；模型不看 main、标签、review ID 或抽样原因。
- 同一 .14 模型与设置：nvidia/qwen/qwen3.8-27b，temperature=0，top_p=1，output=4096，visible=20000，
  window=4096/512，parallel=64，timeout=600，最多 2 次 outer retry。继续用原 NDD runtime/schema 生成链路。
- 每臂先独立 8 对技术预检，共 24 次 critic 请求，不回灌正式输出。
- 两次重复各 258 对：R1 顺序 control→consent→pointer，R2 pointer→consent→control；
  每轮各臂使用同一个按固定 hash 排序的样本顺序，总计 1,548 次正式 critic 请求（不含内部/外部重试）。
  反转顺序降低单向时间混杂，但两次重复不能证明统计显著性，必须报告控制臂自身波动。
- run root `/raid/hfang/ihb/runs/v0.6.2.16-critic-diagnostic`；独立临时目录、checkpoint、原响应、摘要。
  冻结执行源码、模型设置、全部资源、协议、原标签、原响应、main 和实际输入；运行后不再改推理文件。

## 预先定义的检查与停止

每个臂每次重复完整报告加权/非加权 P/R、主决策、误合并、误判 containment、漏判、未决、各 cohort、
38 负例、22 critic 修复负例、5 benign、4 真包含方向、12 exact、54 translation，以及相对同时段控制和自身重跑的变化。

单项修改只有在两次重复均满足以下条件时，才可建议另跑一次独立完整局部候选；本实验本身不批准全量：

- 258/258 有效，0 终态错误，outer retry <=1%；失败不丢样本，不以未决冒充正确。
- 原 local gate 不放宽：.12 已保住的 20 负例与 3 benign 无回退，4 真包含与 12 exact 全对，
  translation 加权主决策相对 .12 不低于 −3 pp；全部额外 22 负例全对。
- consent 的 H0347 或 pointer 的 H0748 主决策正确；其他保护仍完整计分。
- 加权主决策不得低于本轮 control，误合并数不得高于本轮 control。

只报告单次目标修好不算通过，不挑最好一次、不做跨次投票或拼接挑选，不因 control 也失败而放宽门槛。
任何候选未通过则停止在诊断报告，不叠加第二项、不自动注册默认版本。
与旧参照冲突的非空包含、翻译和短角色口径，仍需用户确认仲裁；不以本次 AI 诊断自动改标。
