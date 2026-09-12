# V0.6.2.26 待实施设计：显式语义选择，程序派生冗余字段

本文件在`.25`预检失败后提出，尚无`.26`模型响应、运行根或成绩。
主要证据是H0384的HARMLESS_ONLY加非空counterpart IDs组合，以及此前反复出现的inactive witness/mode冲突。
目标不是放松已冻结v2验证或只提高表面completion，而是减少模型反复编码同一语义决定的机会。

## 设计边界

保留原输入、完整句义/主体/具体方法核对、公共v3输出和严格原文选择；新建不可变内部合同。
主Judge的NO/U、exact与不完整确定路由、scope解耦及严格非空containment归属不变。
旧v1/v2响应及其失败永远用旧parser评估，不根据新合同补空字段或更改历史mode。
新的schema通过不算蕴含证明；precision/recall/primary、全部原保护和≤1%重试条件继续独立评估。

## 优先验证的结构

每个side只选择一种语义结果：COVERED、UNCOVERED、UNRESOLVED。
使用带status判别字段的互斥对象结构，避免要求covered结果反复生成NONE/空source/空consequence等inactive字段。

- COVERED：完整reviewed unique清单、简短全句义解释、harmless_unique_ids和opposite_support_ids。
  harmless IDs必须为own unique子集；存在未标harmless的unique时，support必须为实际语义counterpart，不能只引共同主题。
  只有全harmless/无unique时，support可作为可选的解释上下文；程序记录为context-only，不把它算作语义counterpart。
  是否确属harmless仍由模型判断，不能由“引用存在”自动决定。
- UNCOVERED：完整reviewed清单、简短命题解释、一个own unique loss source、可选opposite/shared counterpart、
  真实loss type、NO_EQUIVALENT_FOUND或有反向命题证据的CONTRADICTS、实际保留后果。不能只给一个status。
- UNRESOLVED：简短原因与已检查的引用，不声称已覆盖或已证明损失。不完整输入不能生成resolved side。

coverage_mode、inactive空字段和最终replacement方向由程序按新合同显式派生，
不是解析自由解释来改status。例如没有unique时no-unique；全harmless时harmless-only；
harmless与有语义counterpart的unique并存时mixed；其余被对侧覆盖时semantic-counterparts。
编译器不得把缺少对侧证据的非harmless unique自动升级为covered，或把真实损失改成UI。
全harmless时非空support本身不矛盾：它仅是解释性上下文，仍保留在原始新合同与编译审计中，
不是给旧HARMLESS_ONLY响应删引用来使其通过。新合同不再让模型同时选择mode再满足冗余的空字段规则。
定义需清楚处理同一diff片段同时包含UI和保留命题的情形：不能仅因包含按钮就把整span标harmless。
完整翻译material NONE；普通chrome差异和真实条款缺失继续区分。

## 必须先解决的真实执行边界

当前安装的DataDesigner支持oneOf/discriminator，并在默认pruning时用判别字段避免试错分支删字段；
但这只是读过实现的能力线索，仍须实际执行验证。

1. 正式native response recipe的schema呈现、分支选择、缺字段/错分支/额外字段拒绝；
   另以pruning=False核对**原始assistant响应**，不能信任默认pruning后的对象。
2. 同一batch混合三种side对象经Ray/Arrow/writer后可能被补null字段。需要按已严格验证的原始响应，
   对传输层补null做独立可审计的绑定；仅允许已验证的representation变化，不能宽容LLM自身的非法null/额外字段。
   优先复用既有payload transport精确绑定原语，不复制或放宽源文本验证。
3. first/retry均需全链路测试；未调用分支不得出现伪造的response/proof/schema成功。
4. 新编译器为两侧保留原文source/counterpart位置；未知/跨侧/错位/过长ID仍严格失败。
5. inactive字段由新合同派生的行为必须显式记录为编译，不作为旧失败响应的修复。

若native分支或传输无法可靠支持，先解决该边界，不用更宽松parser或抹去失败绕过它。

## 后续实验原则

先合成成对回归与真实native边界，再预先冻结具体schema、提示、两臂、call sets与运行协议。
两臂同27B、同payload、同scope算法、同原标签/权重与既定预算；原已见样本不称holdout。
需要避免把提示变化、合同变化和scope修复混成一个无法归因的比较；同步对照和两轮全部保留。
旧`.25`未完成正式schedule，不能当作一个有语义成绩的完整baseline。
所有准入不变，不自动调整reference、扩大模型权限、运行全量或release。
