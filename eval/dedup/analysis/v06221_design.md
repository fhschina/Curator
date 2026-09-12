# V0.6.2.21：coverage 执行边界与状态规则同步

## 状态

实现及冻结前验证阶段。传输 helper、版本化 system/pair/YAML、两臂 raw trace binding、
在线执行和重放/汇总入口已实现，正式实验尚未开始。执行规则见 [冻结协议](v06221_run_protocol.md)。
不把 .20 预检重放当 .21 正式输出，不复用其缓存，不修改任何 .20 冻结文件。

## 必须先完成的改动

1. 在线执行与重放使用相同的严格原 payload 传输校验，保留原始 echo、response 和 trace 摘要。
   不仅核对 payload hash 字符串，也逐项核对实际内容；不对模型 witness 做自动“纠正”。
2. 两臂首轮输入均用空字符串反馈，所有重试均调用 `encode_repair_feedback`。
   用实际 HTTP 消息验证 renderer 一致，不再只验证 DataDesignerStage 或 echo 字段。
   这改变了实际执行提示，因此是新的受控实验，不声称与 .20 在线请求完全相同。
3. 传输、pair 身份、模型可见请求不一致等确定性故障直接停止并留档，不能消耗 LLM 外层重试预算。
   schema/证据生成问题继续独立计数；保留 native correction 与 outer retry 的并集门槛。
4. system、pair、YAML rubric 同步明确下面的现有合法状态，内部 schema/validator 保持原严格语义。
   所有新提示另存版本文件；不能只追加 system 文本或自动把旧 MIXED 改成 NOT_COVERED。

| status | coverage_mode | coverage_counterpart_ids | witness |
| --- | --- | --- | --- |
| COVERED，有语义对应 | SEMANTIC_COUNTERPARTS / MIXED | 必须有实际对侧/共享引用 | 必须 inactive |
| COVERED，仅 UI/noise/repetition | HARMLESS_ONLY | 必须空 | 必须 inactive |
| COVERED，本侧没有 unique span | NO_UNIQUE_SPANS | 必须空 | 必须 inactive |
| UNCOVERED，即使部分内容已覆盖 | NOT_COVERED | 可按原合同交代已覆盖部分；不能替代 witness | 必须有真实 unique source、类型与保留后果 |
| UNRESOLVED | NOT_COVERED | 必须空 | 必须 inactive |

`MIXED` 只表示整侧均已覆盖，其中一部分靠语义对应、一部分为无害差异；
不是“部分覆盖、部分丢失”。一旦还有实质未覆盖意义，必须 UNCOVERED + NOT_COVERED。
HARMLESS_ONLY 的 shared anchor 已在独立 anchor 字段提供，不能用 coverage_counterpart_ids 重新声明语义对应。
通过结构校验不能代替实际语义复核；不能为消除状态错误而把真实添加硬改成 covered。

## 新实验准入

- 使用独立的新 run root；相同 258 对、原权重、固定 .14 main、同一 27B/temperature/top_p/输出预算。
- 在线提交前先跑完整 mixed-span Ray→本地 HTTP→原生解析→writer→binding 边界，首轮和重试两种均通过。
- 然后冻结全部新依赖、token preflight、schema、输入、settings、源摘要与停止规则。
  旧 .20 已冻结 harness 不可原地改；新的执行入口须明确调用新 binding，不能通过隐式 monkeypatch 替换旧函数。
- 两臂各 8 对技术预检，都满足 100% 有效、0 重试、每对恰好一次 HTTP 200 后，才可提交两轮全 258 对诊断。
- 两轮顺序、全部正负保护、三个目标、paired primary/over-group、重复波动与 ≤1% 并集重试率门槛不变。
- 若再次技术失败，完整保留失败记录并停止；不得用旧预检或另一轮最好结果顶替。

这轮只解决已观测的运行和状态口径缺陷，尚无依据承诺语义提升。
固定-main 诊断通过后仍需要 fresh 完整 Judge 局部评估，再决定完整 1000 条。
原 75% precision、75% recall、79% primary、over-group ≤66、分类保护和独立 400 holdout 要求不变。
待确认 reference 修订与较大模型调用仍不采用，不自动 release/commit/push。
