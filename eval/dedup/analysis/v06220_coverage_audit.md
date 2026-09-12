# V0.6.2.20 预检审计：停止正式评估，先修运行边界与字段一致性

## 结论

**未晋级，未运行 258 条正式 cell、完整 1000 条、holdout 或 20000 对。**
这次失败不能用于判断语义方案优劣；也不能把离线修复后的输出改记成一次成功在线实验。
原 reference、权重、主响应、旧提示、缓存、历史冻结文件和运行结果均未修改。
不涉及更大模型、release、commit 或 push。

正式合同摘要：`1667746bc64b10dbd29305618b77c47abb07946c4171b3f566198db379780147`。
运行根：`/raid/hfang/ihb/runs/v0.6.2.20-coverage-diagnostic`。
[冻结协议](v06220_run_protocol.md)、[原始完成状态审计](v06220_coverage_assessment.json) 保留失败口径。

## 实际在线请求

| cell | 请求 pair | 接受 | 终态错误 | 外层重试 pair | HTTP 200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| control 预检 | 8 | 8 | 0 | 0 | 8 |
| coverage 预检 | 8 | 0 | 8 | 8 | 24 |

共 32 次外部 27B HTTP 请求，没有服务端 HTTP 错误；coverage 的三次外层尝试各 8 条。
coverage 原始 trace 中没有 native correction，但每一条都先被新增的 payload 比较拦截。
两臂预检结束后，预声明 gate 正常阻止正式提交。所有原始输出、trace、错误和事件均保留。
外层重试没有纠正代码的传输判定错误；下一执行器应把确定性传输/身份故障视为运行错误，不能继续要求 LLM 修复。
relay 没有记录 token usage，本报告不把缺失 usage 当作零消耗。

## 根因与修复

### 1. 嵌套 payload 的 Arrow 表示变化

新 harness 冻结前增加了 `row.payload == input.payload` 检查，但当时真实 NDD 阶段测试只覆盖 exact 文本，
没有覆盖混合 shared/unique span 的完整 Ray 流水线，这是测试覆盖缺口。
首轮 8 对输出仅观察到 1296 个新增 null 字段、688 个等值整数→浮点数变化，
pair ID 和原 payload digest 一致，没有文本、顺序、实质字段值变化。

[payload_transport.py](../judging/payload_transport.py) 新增严格传输校验，不修改冻结 harness：
只允许额外 null struct 字段和可精确还原的整数→浮点数；不允许删字段、变更文本、改变列表顺序/长度、
非空字段注入、布尔/整数混同或整数精度损失。
通过后仅在内存绑定回冻结原文，保留原始 echo、response/trace 摘要与实际原文 evidence 校验。

### 2. 空反馈和结构化反馈不能直接穿过整个执行链

首轮 `repair_feedback=None` 在 Arrow/Pandas 中成为真值 `NaN`，实际提示意外带有 `<repair_retry>nan`。
重试反馈中的空字典 `details={}` 又会变成 `[]`。因此 24 条实际 native 请求均不与直接原始 renderer 完全一致。
首轮 payload-only 提示保持一致，但这不等于完整请求一致；本轮不能作为已控制好的语义实验。

新编码把缺失反馈变为 `""`，真实反馈变成带固定文本前缀的 JSON 字符串。
不能只把字典变成裸 JSON 字符串：NDD 的 `deserialize_json_values` 会自动将其解码回字典。
固定前缀防止这种自动转换；空值继续为 falsey，不触发首轮重试提示。

### 3. 传输修复后仍有内部字段一致性失败

[最终离线传输验证](v06220_transport_verification.json) 验证全部 24 份原始响应，不调用模型、不改答案、不读取标签计分：

- 24/24 的原文 payload 均通过严格传输校验；保留完整提示与重试反馈表示差异。
- 内部/公共输出合同通过数按尝试分别为 5/8、5/8、6/8，共 16/24，而非 24 个独立样本。
- 8 份仍失败：2 次 `COVERAGE_COUNTERPART_MISSING`，6 次 `COVERAGE_WITNESS_MISSING`。
- 首轮 H0742 为 `HARMLESS_ONLY` 却填写了不允许的 counterpart；H0260、H0882 为 `UNCOVERED` 却选 `MIXED`，
  与 validator 要求的 `NOT_COVERED` 不一致。review ID 仅用于查找，不据此修改 reference。

不能直接清空引用或改枚举来让旧响应通过。需要在下个版本的 system、pair 和 YAML rubric 中明确同步状态组合，
再用新响应检验。当前证据不足以发布 precision/recall 或比较版本优劣。
早期 `v06220_transport_replay.json`、`v06220_transport_replay_v2.json` 与 `v06220_transport_contract_audit.json`
是定位完整提示差异期间的调试快照，最终实现与口径以 `v06220_transport_verification.json` 为准。

## 验证与下一步

[传输修复边界报告](v06220_transport_boundary.json)：555 个常规测试通过，2 个独立 Ray 集成用例默认跳过，
2 个 GPU 用例未运行；另行指定本次自有 Ray 集群后，两个完整集成用例均通过，共 4 条本地模拟 HTTP 请求。
两用例均含 mixed spans，分别覆盖首轮与带空 details 的重试反馈；逐字检查实际 HTTP 提示、raw trace、原始 payload 与 evidence。
运行完整 Ray 用例使用 `-s`，避免 pytest 捕获流被 Loguru/Ray actor 序列化。
全套测试期间不得并发生成评估目录中的报告，否则 source digest 检查会正确拒绝该运行。

本次自有 Ray head PID 3326942 已停止，52646 端口已释放；未停止其他集群。
.16、.17、.18、.20 的历史冻结校验均通过。
下一步见 [.21 修订范围](v06221_design.md)：先修清晰可复现的运行/合同问题，重新冻结预检，不扩大样本或降低原门槛。
