# V0.6.2.21 coverage 配对执行协议

本协议承接 [设计](v06221_design.md) 与 [.20 失败审计](v06220_coverage_audit.md)。
正式冻结后不改代码、提示、标签、输入、schema、main 或预算；失败不原地重启。
不是 release 准入结果，不读取 holdout，不调用更大模型。

## 两臂合同

- control：原 .14 critic 的全部提示、rubric、schema、模型配置与 v8 仲裁。
  新 `.21 control` YAML 仅改 stage 名并开启 all_messages trace；冻结前逐项比对配置。
- coverage：新 `.21 system/pair/YAML rubric` 同步状态组合，内部 `dedup-retained-coverage-v1`
  与公共 v3 校验、`.20 coverage` 仲裁规则保持不变，不替旧响应清字段或改状态。
- 两臂都用严格 payload transport binding，逐项核对原文与原 hash。
  feedback 初始为空字符串，重试为带固定前缀的安全 JSON 字符串，防止 NaN/空字典/自动解码改变提示。
- 两臂原始会话的初始 system/user 消息必须逐字等于该尝试输入的实际原生 renderer。
  最终 assistant 响应须以非 pruning 原生 schema 解析，并与输出列完全一致。
  保存原始完整行、payload echo、trace、request-message 和 response 摘要；证据只绑定冻结原文。
- 记录 native correction、外层 retried pairs 和两者并集；两臂都可观测，不再把 control 的缺失 trace 当零重试。

## 固定样本与顺序

- 新根 `/raid/hfang/ihb/runs/v0.6.2.21-coverage-diagnostic`；所有输入、checkpoint 与输出全新。
- 使用 .20/.18 的相同 258 对、原始权重、固定 .14 main 和原两轮 deterministic pair 顺序。
  `.20` 的已观察输出绝不进入新输入或代替新调用。
- 同一 NVIDIA Qwen 27B endpoint，temperature=0、top_p=1、max_tokens=4096、关闭 thinking，
  其余并发64、timeout600、外层 max_retries=2 不变。
- 两臂各自全部 258 对实际 renderer/tokenizer 预算重新验证：输入+4096+2048≤32768，
  不裁剪、不删除超长样本。预算是客户端保守值，不声称服务端上下文上限。
- 先 control、再 coverage，各为 R1 前 8 对。两者必须 8/8 有效、0 errors、0 外层重试、0 native correction，
  且 HTTP 状态恰为 8 次 200；任何一臂不满足就不提交正式 cell。
- 正式 R1 control→coverage；R2 coverage→control；每个 cell 全部 258 对，共 1032 次基础请求。
  16 条预检单独保留，不计入正式语义分数，也不复用响应。
- 输入身份、实际提示、feedback、trace 等确定性运行边界失败停止运行，不能通过额外模型请求修复。
  同 cell 已产生的输出、错误、接受与终态 pair 全部保留；模型 schema/证据失败才走有界单 pair 重试。
- 正式 cell 任何终态错误停止后续 schedule；所有 pair 都在完成率分母中。

## 冻结前与完成后校验

冻结前对两臂分别做 mixed-span 的首轮/重试完整 Ray→本地 HTTP→writer→binding 测试，共四个用例。
校验实际 HTTP 提示和 native trace；只用本地固定响应，不请求外部模型。常规测试单独运行，
避免在 source snapshot 测试期间并发修改分析报告。

冻结包括当前 harness/test、两臂配置、新提示、传输 helper、边界报告、预算、固定 main、输入与所有旧依赖。
回放按 pair + 实际 outer attempt + 完整 raw row digest 定位接受响应，而不是选某次最好的答案；
重放使用该次实际反馈，不把 retry 错当第一轮提示。终态 ID 集合必须恰好补全原输入集。

## 评分及晋级范围

沿用原 local_gates 和 38/22 负例、.12 的 20 负例/3 benign 底线、4 true containment、12 exact、
54 translation 保护。H0347/H0453/H0748 三目标均须正确并检查原理由。
每轮 coverage 的 weighted primary 不低于当轮 control，over-group 不增加；
coverage 的两轮 primary 翻转数不得超过 control。两臂另报告 native+outer 并集重试率，coverage 必须≤1%。
每轮、每个原 gate 都须通过，不能择优选轮次、降低阈值或用未决当正确。

技术预检失败不发布语义准确率。完整 schedule 才发布全样本加权/非加权与 cohort/置信档指标。
结果只能支持下一次 fresh 完整 Judge 局部验证，不代表完整 1000 条能力。
原开发集 P/R≥75%、primary≥79%、over-group≤66 与分类保护、独立400 holdout 和最终20000流程均保留。
reference 含 prediction-aware AI 修订的局限继续披露，不称作独立人工盲审。
