# V0.6.2.23 设计：选择原文证据，保留语义审查责任

本文件为`.22`停止后的下一步设计，尚无`.23`在线请求、运行根或成绩。
依据[`.22`审计](v06222_coverage_audit.md)，目标仍是原完整开发集与holdout达标，不以schema成功替代准确率。

## 1. 新的内部证据选择合同，旧公共v3不变

增加新的不可变内部版本，保留dedup-retained-coverage-v1和所有旧响应的严格拒绝结果。
模型对每个UNCOVERED侧选择一个own unique source_span_id，必要时选择一个opposite/shared counterpart_span_id，
而不重抄source_quote/counterpart_quote。程序从原payload的指定span提取**完整原文**并计算原始偏移。
复用coverage_witness中已有的span evidence逻辑，不引入模糊匹配、翻译、字符纠错或跨片段拼接。

对本次实际需要coverage的147对进行只读检查：共有4572个span，最大220字符，无超过240字符者。
这支持当前数据上的整span选择，但不是通用上限保证。新合同必须逐对验证完整原文对齐、长度、ID唯一性、
side归属和own unique要求；未知/跨侧/过长/错位ID一律拒绝，不截断证据或跳过pair。
若将来出现长span，需要另行预先冻结细分规则，不能观察模型结果后选择更有利片段。

reviewed_unique_ids仍须覆盖本侧全部unique，coverage_counterpart_ids仍独立支持语义覆盖。
保留status/mode、损失类型、实际比较关系、retention consequence与两个方向；
“选择了存在的ID”只证明来源，不证明对应内容真的不可替代，不能自动当作逻辑蕴含验证。
只从新v2选择结果显式编译证据，绝不能给历史v1响应补字段或自动修quote来重新宣告成功。

## 2. 移除critic没有决策权的exact选项

coverage前置路由已经处理不完整、main NO/U和原文完全相同分支。
实际请求中的record_scope不再允许IDENTICAL_TEXT：它只做记录/消息归属与语义覆盖。
soft hyphen、重复段落、重排、完整翻译可以双向覆盖，但不能声称原始文本字节一致。
重复同一事实不等于实质扩展；真实新增程序、条件、数字或成员不能因此当作重复。
不能因删除一个枚举就把所有scope强行映射成SAME_SUBSTANTIVE_RECORD。

字段布局直接由schema产生和检验，不手写“13字段”等数量。本次实际v1为每侧12字段。
说明保持短且聚焦保留命题与对应关系，不在JSON字段里输出长篇自我推演；maxLength严格执行，不截断旧响应。

## 3. 语义检查仍单列

H0119提示真实风险仍是非空containment gate：共同商店导航不能给A的具体商品列表提供B侧记录核心。
新增通用成对边界例并保持system/pair/rubric同步：

- 同站导航+某个产品tag/listing，对仅共同导航：不满足双侧实质记录锚点，不能containment。
- 两侧明确是同一个FAQ/产品/招聘记录，只有一侧新增程序或缺失字段：允许真实单向扩展。
- 同段正文重复、普通按钮或完整翻译：可以双向保留；不升级为identity/page-role冲突。
- 名字/类别/列表成员只出现在导航中，不据此臆造主体；但真实独立列表、服务对象、状态冲突应优先no/no。

先用合成回归和保留的失败簇检查编码边界，不把看到过的旧例称为holdout。
不改旧reference，也不把AI复核当独立人工；所有旧负例、true containment、translation、exact及三目标保护继续计分。

## 4. 执行边界与准入

在现有可配置入口之上增加明确的新合同绑定/renderer接口，保留`.22`冻结模块，
不再复制整套每版本执行器，不monkeypatch旧模块。新旧raw schema及adapter都须一对一测试。
两臂first/retry完整本地HTTP链路必须验证新合同、真正的请求消息和字节证据，再冻结新的run root与call sets。
结构编译的离线测试不算一次模型判对，也不产生替代`.22`结果的分数。

`.22` control出现46次429。下一次在线前检查现有请求准入/限流支持，若调整并发或节奏，
必须同时作用于两臂并提前写入新合同，不能中途改变，也不能删去旧429。
HTTP失败/重发与native+outer纠错分开报告；实际模型调用分母不含111个确定分支。
仍用同一个获授权27B模型；较大模型、reference修改、独立人工标注均需已有范围以外的明确授权。

同258对、固定main的实验只能判断critic增量；每次预检和两轮配对均保留，不择优。
只有完整schedule、100% schema completion、≤1%实际called-pair重试、原保护集及语义比较都通过，
才支持fresh完整Judge局部验证。之后才是完整1000的75/75/79与误合并≤66，再到独立400 holdout、最终20000。
