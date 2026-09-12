# V0.6.2.23 span-selection 配对执行协议

本协议在新响应之前冻结；承接[设计](v06223_design.md)及[`.22`失败审计](v06222_coverage_audit.md)。
新根`/raid/hfang/ihb/runs/v0.6.2.23-selection-diagnostic`，不重启旧run、不复用旧cache或某个成功响应。

## 两臂与合同

control保持原v8仲裁和`.21`仅启用trace的原生配置、提示、rubric与schema。
coverage采用新的dedup-retained-coverage-v2：模型选择单个own unique source_span_id、
可选单个opposite/shared counterpart_span_id；程序提取完整span并验证原始偏移。
旧v1响应仍按旧规则拒绝，不能用v2替旧quote补字段或自动纠错。
新system/pair/YAML同步非空记录锚点、重复/翻译与真正扩展边界；公共v3不变。
实际schema生成字段清单，不手写字段数量；critic的scope不含IDENTICAL_TEXT，exact由既有确定路由负责。

所有selectable span预先验证唯一ID、side/kind、完整原文对齐和≤240字符长度。
未知/列表/范围/跨侧/共享source/过长/错位选择必须失败；不模糊匹配、不截断、不拼接、不变更语义结果来绕过错误。
已生成schema和原始回应先严格非pruning解析，再显式编译。原始selection与编译结果均有独立摘要；
选到真实ID只证明来源，不证明模型关于保留意义的判断正确。

## 固定输入、预算与调度

相同258对、原权重及reference、固定`.14`main和两个deterministic repeat顺序。
coverage调用集合仅由main和原payload预先决定：147对需要新critic，111对确定分支保留公共结果。
control调用全部258。所有对都进入公共完成率和语义分母，不能删去重复payload或失败对。
未请求行保持NOT_REQUESTED_OWNED_BRANCH，不伪造response、proof或schema成功。

同NVIDIA Qwen27B，temperature0/top_p1、输出4096、关闭thinking、timeout600、外层max_retries2。
本轮两臂统一max_parallel_requests=16（此前64），提前作为操作变化冻结；不改变抽样/模型/内容或中途调参。
已检查现有DataDesigner支持通过inference_parameters设置容量，其默认request admission仍负责限流恢复。
不引入新调度器或宣称降低并发一定消除429。逐项保存HTTP200/429/其他错误，并与native+outer模型纠错分开报告。

所有实际调用按实际原生renderer和冻结tokenizer检查输入+4096+2048≤32768。
此为客户端预算，不声称服务端上限。不裁剪或移除超长样本。
冻结全pair清单、实际call sets、确定分支结果、schema、提示/代码/测试、main、权重、消息摘要和预算后才请求模型。

先control后coverage，各取R1该臂实际调用顺序的前8对，不按难易或旧错误选择。
已见开发样本不称holdout。两臂必须8/8首轮有效、0errors、0native/outer重试、恰好8次HTTP200才进入正式评估。
预检结果不计语义分数、不复用为正式响应。
正式R1 control→coverage，R2 coverage→control；共1032个公共输出、810个基础模型请求，另16次预检。
模型合同失败只对失败pair按固定预算重试；确定性传输/输入/提示边界错误不通过模型请求修复。
任何正式cell终态错误则停止后续schedule；所有原始尝试与终态ID保留，失败不选择最好一次或原地重跑。

## 验证、评分与晋级

冻结前通过两臂首轮/encoded retry四个完整Ray→本地HTTP→writer→binding用例，
coverage包含真实有损方向的非拉丁原文选择及公共双侧证据；这不是外部模型准确率。
每个接受响应按pair+实际outer attempt+完整raw digest回放，并核对实际反馈/初始消息/原payload。
确定分支按冻结main和payload重新计算，公共字段完全一致，不能带虚假response或proof字段。

完整schedule后才发布语义评分；保留local_gates、38/22负例、.12的20负例/3benign底线、
4true containment、12exact、54translation及H0347/H0453/H0748三目标。
每轮coverage weighted primary不低于当轮control、over-group不增加；两轮翻转不超过control。
每臂native+outer重试pair并集/该臂实际called pairs≤1%，公共完成率仍用258分母。
HTTP限流另列，不伪装为无重试；100%最终严格schema/evidence完成率不可放宽。

结果只支持下一次fresh完整Judge局部验证，不能直接推广到完整1000。
之后仍须完整1000达到P/R均75%、primary79%、over-group≤66与原分类保护，再到独立人工400 holdout、最终20000。
不改reference，不把AI复核称为独立人工，不请求较大模型或自动release/commit/push。
