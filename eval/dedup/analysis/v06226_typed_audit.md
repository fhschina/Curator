# V0.6.2.26：typed候选预检通过，flat对照重试使正式实验停止

## 结论

**没有正式语义成绩，不晋级。** 新typed候选8/8首轮严格通过、零native/outer纠错；
flat对照7/8首轮通过，1条外层重试后8/8。依据[冻结协议](v06226_run_protocol.md)，
两臂必须同时通过零纠错预检，因此调度器停止，未提交正式258对cell、完整1000、holdout或20000。
[机器报告](v06226_typed_assessment.json)经完整原始输出重放，仅报告预检operations，未选择性重跑或发布语义排名。

运行根 `/raid/hfang/ihb/runs/v0.6.2.26-typed-diagnostic`；
合同 `9ba66e7f155596d4c8346964ae2f3632afaecccaa2dd7ea5fe010f6bf3c33aad`；
2026-09-10 18:56:43–18:57:17 UTC。17次外部请求全部HTTP200，无服务失败或native纠错。

| 预检臂 | 实际调用pair | 首次外层有效 | 最终有效 | outer重试pair | native纠错pair | 并集/实际调用 | HTTP200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control：flat v2 | 8 | 7 | 8 | 1 | 0 | 1/8 = 12.5% | 9 |
| coverage：typed v3 | 8 | 8 | 8 | 0 | 0 | 0/8 | 8 |

8条的零纠错仅证明本次候选预检通过，不代表全开发集重试率≤1%，更不证明语义目标达标。
计划中的1032公共输出/588基础请求未执行，不计完成；终态有效8/8不能抹除对照已经发生的重试。

## 改变范围与证据

本轮control是`.25`完整命题提示的flat v2，不是`.25`实验的旧control，也不是`.14`critic。
其147条预渲染消息与`.25`候选逐条一致。候选使用新内部dedup-retained-coverage-v3，
逐字嵌入同一`.25`完整命题、主体绑定、非空记录、真实方法及成对边界规则。
同步改动system/pair/YAML中的typed编码说明；这不是逐字完全相同prompt下仅换parser。
两臂保留固定main、scope后处理、ownership路由、原标签/分层权重、27B模型、温度、预算和并发16。
同258对、147 called/111 owned；最大渲染输入token为17302/17363，没有裁剪或丢弃输入。

新合同只要求模型输出所选COVERED/UNCOVERED/UNRESOLVED分支的字段。
模型仍决定status、harmless归属、实际支持或损失；程序根据显式字段派生mode与inactive字段。
存在未标harmless的unique时，必须给出对侧语义support；全harmless时支持ID仅作解释上下文，
保存为context_only_ids，不转成语义蕴含证明。未知/错侧/错位证据、错误分支字段及原始非法null仍严格拒绝。
公共v3输出与历史内部v1/v2规则未修改，旧失败不会按新合同重新算成功。

## H0384：重复的flat编码失败与新的typed记录

pair `cp1_0d38a9acc7c7e85afc4f9e7e6c8f1c0b413a87659577438e8f645aebb17e61a9`。
对照第一响应两侧均COVERED、HARMLESS_ONLY，同时分别给出非空counterpart IDs：
A侧S004,B002；B侧A002,A004。旧v2明确要求harmless模式的该字段为空，故严格失败。
COVERAGE_COUNTERPART_MISSING的历史错误提示提到了缺少语义引用，但本例实际是**harmless模式带非空引用**，
与`.25`失败同簇。原响应、反馈及重试原样保留，没有自动删除引用或改mode。

typed候选显式将A001–A004与B001–B002分别标为harmless，两个status为COVERED，
支持ID分别为S001,S004,B002和S001,S004,A004。编译记录派生HARMLESS_ONLY、retained_unique_ids为空，
保留全部支持ID为context-only。该组合符合预先定义的新合同，而不是修复旧flat响应。
本条typed原始结果与落盘列完全相同，response transport没有表示变化；payload则有已严格绑定的
64个null补键和40个等值整数转浮点表示。二者的审计分开保存。

这仍不是“模型所有harmless判断正确”的证明：模型可能把真实身份、条件或方法错误归为harmless，
存在合法ID也不证明命题被对侧保留。不能从本条推出translation、真实containment或负例保护已达标。
按协议不对预检发布语义分数。

## 工程验证与清理

- [边界报告](v06226_boundary.json)：CPU测试726通过、22项需自有Ray的测试跳过、2项GPU排除。
  本版两臂first/retry共4个真实Ray→native parser→本地HTTP→writer→bind用例单独通过，模块16项通过。
- 完整边界每个用例混合三种side分支、真实单向扩展及LOW U，并保留未调用分支的无proof状态。
  最终4条链路12次本地HTTP；此前一次测试错误地期待native writer补null，在6次本地HTTP后断言失败。
  修正为实际观察到的“分支字段原样保留”，另加真实PyArrow往返用例验证确实会出现的null补键。
  合计18次本地HTTP、0外部调用；没有修改生产接受规则来使测试通过。
- 原始assistant始终非pruning严格解析，再独立绑定native列；消息、反馈、输入全文、schema分支和证据均核验。
  未知新增字段属于fatal boundary，不送给模型重复修复。共享执行器对两臂均按实际selection列统计native纠错。
- 本轮自有Ray head PID221175已退出，52651端口已释放，没有停止其他集群。
- 历史reference、权重、运行根和缓存保持不变。AI参与修订的开发集不称独立人工盲审。

## 下一步

见[`.27`语义块对照设计](v06227_design.md)：两臂统一为typed v3，在相同公共任务/编码说明下，
比较原简短规则块与完整命题/主体边界规则块。这样才能直接测量提示差异，而不反复让已知flat冗余字段失败阻断它。
`.25/.26`停止结果仍保留，不以新实验覆盖、放松预检或从重复运行挑最好一次。

所有目标不变：先完成局部配对及fresh完整Judge局部验证，完整1000须75/75/79、over-group≤66和分类/重试保护；
之后未见400条独立人工holdout，最后20000。本轮未release、commit、push、调用较大模型或修改reference。
