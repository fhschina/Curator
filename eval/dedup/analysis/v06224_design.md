# V0.6.2.24：scope 分类与保留语义的离线因果检查

本协议在计算新规则的开发集分数之前写下；这是看过`.23`错误后提出的开发假设，不是盲测。
不修改`.23`已冻结adapter、模型响应、main ledger、reference、权重或运行完成状态。
尚未授权晋级、全量在线评估或release。

## 假设与固定规则

`.23`每轮20条公共UNRESOLVED中，2条由不完整输入确定路由，18条来自main内容分类与critic scope不一致。
后者的critic已经输出完整且可校验的方向判断，不是critic自己弃权。
正文/非正文分类对containment很重要，但未必应该否定已有的双向等价或明确保留语义冲突。

新增**仅离线**探针，复用原选择合同的严格编译、证据验证及主Judge决策归属：

1. 仅处理旧公共结果为UNRESOLVED、输入完整、main为YES、非exact、支持的main ledger、
   critic两侧均已resolved，且确为NON_MAIN_MESSAGES与main非双侧NON_MAIN_ONLY、
   或SAME_SUBSTANTIVE_RECORD与main双侧NON_MAIN_ONLY不一致的情况。
2. 双侧COVERED且scope为上述两个已绑定类别：直接使用既有双向coverage语义，输出yes/yes。
   保留NONE/MINOR规则、双侧原文证据与MEDIUM，不依据reference或main的translation标签额外挑选。
3. 任一UNCOVERED为POLICY_MEANING、RECORD_IDENTITY、PAGE_ROLE、STATE_VERSION、MEMBERSHIP、OTHER_RETAINED：
   明确损失不依赖正文taxonomy，输出no/no、MAJOR，沿用原损失类型映射。
4. 只有MAIN_CONTENT增删却scope分类矛盾：仍UNRESOLVED，不放宽containment。
   DISTINCT_OR_UNBOUND_RECORDS、critic真正UNRESOLVED、不完整、main NO/U和exact均维持旧结果。

程序不能修写main profile或critic scope来骗过旧adapter；新输出标注OFFLINE_SCOPE_POLICY，
不产生新模型响应、补证据、重试成功记录或production版本注册。
模型把真实identity当HARMLESS_ONLY的错误也会暴露并计入误合并；不得按样本ID例外处理。

## 评估与审计

先一对一单元测试，包括双侧等价、保留冲突、真实翻译与伪等价风险、单向MAIN_CONTENT不放宽、
未绑定scope、不完整输入、main NO/U、失效证据和输入不变性。
然后对`.23`两轮全部258对分别重放，仅使用每个已接受结果实际绑定的raw行摘要和尝试编号。
原control与coverage均通过冻结文件和完整原始response/transport检查，不能择优取早期无效响应。

同时报告加权/非加权P/R/primary、误合并/漏判、所有原保护集、目标三例和两轮翻转。
保留原在线operations与重试门槛；离线程序成功不构成新模型schema completion或重试率。
全258对、原分层权重和两轮均保留；不把516个重复输出当516个独立样本。

任何收益仅是固定响应的adapter因果结果，不是`.24`在线模型效果，也不能覆盖`.23`成绩。
所有原门槛继续生效：完整1000加权75/75/79、over-group≤66及保护/重试要求，之后独立400 holdout，最后20000。
若局部语义/保护仍失败，先根据保留命题错误设计后续critic，不直接跑完整1000。
