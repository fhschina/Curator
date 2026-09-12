# V0.6.2.21 在线预检审计：传输修复有效，零重试条件未通过

## 结果

**不晋级；正式 258 对 cell、完整 1000 条、holdout 和 20000 对均未提交。**
合同 `33f43bbf0f49b40e38a89ca361b5eba1c15b581b4b94afe316b2752d2dfdf2e0`，
运行根 `/raid/hfang/ihb/runs/v0.6.2.21-coverage-diagnostic`。
见 [冻结执行协议](v06221_run_protocol.md)、[完整预检状态与原始响应重放](v06221_coverage_assessment.json)。

| 预检臂 | pair | 最终有效 | 终态错误 | 外层重试 pair | native correction | 两者并集 pair | HTTP 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 8 | 8 | 0 | 0 | 0 | 0 | 8 |
| coverage | 8 | 8 | 0 | 1 | 1 | 1 | 10 |

实际共 18 次外部 27B HTTP 请求；其中 coverage 的外层重试与 native correction 都发生于 H0742，
不能把它们当成两个不同重试 pair，也不能只报告“8/8 最终成功”。
冻结预检要求零重试，故正常停止；没有临时放宽条件或复用一次成功结果。

## 本次技术改动验证了什么

- 两臂全部 17 份输出行的原始 payload、encoded feedback 与 native 初始提示都通过核对；
  17 行包含 control 8、coverage 首轮8及外层重试1，native correction 保存在该重试行的完整 trace 中。
- `.20` 的 null padding、整数偏移提升、NaN 初始反馈和空字典变形问题没有在本次校验中复现。
- 两臂都保存 all_messages trace，对原始最后一条 assistant 文本进行非 pruning schema 校验，
  并按实际 outer attempt、完整 raw row digest 和该次输入重放。
- 首轮 coverage 7/8 通过，另1条为 quote alignment 失败；本预检未再出现 `.20` 的 MIXED/NOT_COVERED 等状态组合错误。
  这仅是8条预检的技术观察，不是完整开发集稳定性或准确率证明。
- 冻结前569个常规测试通过，4个本版完整 Ray→本地 HTTP 用例分别覆盖两臂首轮/重试并全部通过。
  [边界报告](v06221_boundary.json) 保留源码摘要；模拟响应不计作在线 Judge 成绩。

## H0742 的两次失败原因

第一条响应把 `A001` 的 `Amazon Pay` 与 `A002` 的 `Kunden-Wiedererkennung` 拼接到同一个 source_quote。
两段在原文中并不连续：A001 为 [493,503)，A002 为 [579,601)，中间有 S004 的共享内容。
拼接引文不是任何一个引用 span 的连续子串，因此 `COVERAGE_QUOTE_INVALID` 拒绝是正确的。

外层第二次请求改为单个 A001 的 `Amazon Pay`，但第一次生成漏了 A 侧必需的 `coverage_counterpart_ids` 字段。
NDD native schema 提示缺字段后，模型补回空字符串，最终通过。没有代码替它补字段或改 quote。

该样本主 Judge 已是 no/no；两臂最终也均 no/no，旧 reference 为 no/no。
本报告只借此确认失败属于未使用 critic 的格式问题，不据一条预检做语义版本排名，更不修改旧标签。
主 Judge 不允许被 coverage critic 重新打开的路由规则，已有独立代码分支支持。

## 下一步：先验证依赖路由，而不是事后豁免

[离线 coverage 路由审计](v06221_coverage_routing_audit.json) 对全部258个冻结输入检查了现有仲裁的决策归属：

- 147 对：非 exact 正例，仍必须新调用 coverage critic。
- 97 对：主 Judge 已定 no/no。
- 12 对：完整原文 exact。
- 2 对：不完整输入，保持 LOW confidence 的 unresolved，不能变成“正确负例”。

独立的 [coverage_routing.py](../judging/coverage_routing.py) 尚未接入生产或已冻结的 `.21` 执行器。
它只读 main 与原始 payload，在观察任何 critic 响应之前做决定；不能按 critic 是否失败或标签是否正确来跳过。
111个确定分支均以合法的合成 unresolved probe 对照原覆盖仲裁，公共字段与 evidence 完全一致；
另有9个路由/审计单元测试，覆盖 owned NO/U、exact、不完整输入、正例必须继续请求以及保留样本分母。
这证明分支兼容性，不证明 main 的准确性；原 main 漏判会原样保留，仍需后续语义优化。

下一版本需预先冻结路由、全258输出集合、剩余147请求集合与新的预检顺序；
不得将 `.21` 已观察的10次请求删掉或重算成零重试。control 的路由兼容性需另行证明，不能直接套用 coverage 的证明。
还需针对真正被请求的 critic 明确单 span 连续 quote 和完整字段，不能把生成错误藏在自动修复中。
详细边界见 [.22 下一步](v06222_design.md)。

没有参考标签/权重/历史结果变更，没有较大模型调用、release、commit 或 push。
本次自有 Ray head PID4140768 已停止，52647端口已释放；其他集群未操作。
所有原开发集75/75/79、误合并≤66、分类保护和独立holdout要求保持不变，目标仍未达成。
