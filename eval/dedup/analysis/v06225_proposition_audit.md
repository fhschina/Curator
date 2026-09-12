# V0.6.2.25：完整命题候选未通过在线预检，不提交正式配对

## 结论

**本轮不晋级，没有正式258对cell、完整1000、holdout或20000提交。**
对照预检8/8首轮通过，候选7/8首轮通过、1条外层重试后8/8。
按[事先冻结的协议](v06225_design.md)，候选未满足零纠错预检条件，调度器自动停止，未重启或择优重跑。
[机器报告](v06225_proposition_assessment.json)只发布预检operations，不发布语义分数或版本优劣。

运行根 `/raid/hfang/ihb/runs/v0.6.2.25-proposition-diagnostic`；
合同 `f35ef898541dd4cf406c0a084f05f228e7c244bdc9e36989fe6cd1fcfe7f9150`；
2026-09-10 18:27:19–18:27:59 UTC，17次外部HTTP请求，全部200，无HTTP失败或native纠错。

| 预检臂 | 实际调用pair | 首次外层有效 | 最终有效 | outer重试pair | native纠错pair | 并集/实际调用 | HTTP200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 8 | 8 | 8 | 0 | 0 | 0/8 | 8 |
| coverage | 8 | 7 | 8 | 1 | 0 | 1/8 = 12.5% | 9 |

12.5%是8条预检的实际比例，不外推成整个开发集重试率，也不能用计划的147或258稀释。
最终严格合同8/8并不取消已经发生的重试。原计划1032公共输出/588基础请求并未执行，不计完成。

## 实验改变了什么

本轮control是`.23`selection提示加同一套`.24`scope后处理，不是旧`.14`record-binding critic。
候选同步更新system/pair/YAML中的完整命题、实际主体与具体方法判断，schema展示将最终status放到末尾。
两臂同model27B、温度、payload、固定main、原分层权重、scope算法、routing、预算和并发16。
同147个实际called输入、111个确定输出；control的147条渲染消息与`.23`selection逐条摘要完全相同，
候选147条均为新消息。最大输入token为16978/17302，未裁剪或排除输入。

输出仍为dedup-retained-coverage-v2：只调整候选schema展示顺序，不改变字段、枚举、字节证据或接受条件。
scope后处理名称仍带OFFLINE_SCOPE_POLICY，表示复用的算法来源；模型响应是本轮fresh生成，
每个响应都有新请求/trace/原始输出摘要和实际尝试，不能理解为复用了离线cache。

## 唯一外层失败：H0384

pair `cp1_0d38a9acc7c7e85afc4f9e7e6c8f1c0b413a87659577438e8f645aebb17e61a9`。
两页含相同cookie政策，差异包括导航/按钮及“have an effect on”对“affect”等局部表达。
第一响应两侧均COVERED，却同时输出：

- coverage_mode = HARMLESS_ONLY；
- 非空coverage_counterpart_ids（A侧S004,S006，B侧S001,S004）。

旧严格v2要求HARMLESS_ONLY的coverage_counterpart_ids为空；语义counterpart模式才允许非空。
因此触发COVERAGE_COUNTERPART_MISSING。错误消息描述了“semantic coverage需要引用”，
但本例实际是**harmless模式带了引用**，不是引用为空。这一历史反馈和失败保持原样。
第二响应保留COVERED，改为MIXED并给出对应的独立引用，通过严格编译及公共证据检查。
没有自动删除第一响应的引用、改mode，或把第二次当作首轮。

这表明仅让status最后生成，不能消除其余mode/引用冗余组合错误。
不能因此宣称第一条语义一定正确：引用的ID存在，不证明该片段实际蕴含对应完整命题，
更不代表H0347/H0453/H0748等目标已修好。按协议不对这批预检发布语义分数。

## 已验证的工程结果

- 候选9个实际响应共18个side对象都遵循“reviewed_unique_ids在前、status最后”；
  control8个响应的16个side对象保持旧status在前。输出顺序干预实际生效，不只是代码意图。
- `.25`两臂使用[同一单臂selection执行/重放接口](proposition_experiment.py)，复用冻结执行器。
  control也按selection列统计native纠错；测试显式覆盖了这项容易漏计的边界。
- 原始模型文本严格非pruning解析，实际消息、payload传输、retry反馈、字段泄漏、最终双侧证据全部核对。
- [边界报告](v06225_boundary.json)：676个CPU测试通过，18项需自有Ray的测试跳过、2项GPU排除；
  本版4个完整Ray→本地HTTP→writer→bind用例单独通过（模块20项），4次本地HTTP，0外部调用。
- 新文件ruff/格式、私钥、大小、大小写、YAML与空白检查通过。未改依赖、旧标签、旧合同或旧缓存。
- 本轮自有Ray head PID166194已退出、52650端口已释放；没有停止其他集群。

## 下一步

见[`.26`合同设计](v06226_design.md)：在新内部合同中减少模型需要协调的冗余/非活动字段，
保留真正的语义决定与严格来源验证。优先验证有判别字段的typed side结果能否穿过真实native解析、
Arrow/writer传输与strict raw绑定；不能只凭JSON schema单元测试就上线。
原v1/v2失败仍失败，不能给H0384历史响应补字段或宣告修复后首轮成功。

目标不变：先局部配对与fresh完整Judge局部验证，完整1000须75/75/79、over-group≤66及分类/重试保护，
然后未见独立人工400 holdout，最后20000。本轮没有release、commit、push、较大模型调用或reference改动。
