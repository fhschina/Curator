# Selected-v3：输出结构对照，而非新语义版本

三臂、原 19 条、各两轮，96 次实际 HTTP 200，零重试；固定 .12 主响应、原 payload 和两套参照。
唯一候选变化是使用真实服务已验证的嵌套 schema，服务端不发送不支持的 `uniqueItems`；本地全部校验保留。

| 臂 | 两轮有效调用 | 全 19 条主决策一致 | 部分草案加权 P/R |
| --- | --- | --- | --- |
| 原 .12 critic | 16/16、16/16 | 13/19、13/19 | 74.56%/36.17%、74.56%/36.17% |
| 同 prompt 的 v3 JSON-object | 14/16、15/16 | 12/19、12/19 | 67.47%/98.61%、60.71%/98.61% |
| 同 prompt 的 v3 strict-schema | 16/16、16/16 | 14/19、13/19 | 65.23%/100%、60.37%/100% |

候选有效率 32/32；原 JSON-object 对照 29/32，缺字段原样记无效。H0850 实际论坛主体冲突在候选两轮均正确，
而同 prompt 的 JSON-object 两轮均放过。这是输出约束下的观测差异，不是“schema 保证语义正确”。

仍未通过原 gate：H0701 待裁定旧 NO 保护失败；H0760 的确认声明在一轮为 INTERFACE_ONLY、另一轮为 RETAINED_CONTENT。
后者有真实重复不稳定，不能通过取最好一轮遮盖。其第二句还包含第三方事先告知/同意的声明，已请求用户明确实质 X 边界。
H0093/H0195/H0352/H0748 继续待裁定，旧标签和全量计分保留。H0748 固定主 NO/YES 无法由 veto-only 恢复为参考 YES/YES。
所有 12 个跨臂错误 ID 已逐条复核，另七条保护记录；无新增独立 gold。

下一步不为争议继续堆 prompt：另冻原 96 条材料的诊断协议，保留全部 19 条和旧 gate 失败，检验其他明确机制。
这是诊断范围修订，不声称旧小试晋级；不自动打开完整 1,000、holdout、20,000 或 release。

运行根：`/raid/hfang/ihb/runs/critic-five-hour-20260911T070052Z/retention-v3-constrained-pilot`。
manifest digest：`57b38aa58c7bb294e001d3a958f08869a7a2edf71d0bf4e021abf3f0826a5b45`。
assessment SHA：`210f3b62fcfb4fb1acf2fd975b3961b636f8b6dac5e327f18e596530f8a2e9dc`。
CPU：1,196 passed、50 skipped、2 GPU deselected；未覆盖冻结源文件或参考，未 commit/push。
