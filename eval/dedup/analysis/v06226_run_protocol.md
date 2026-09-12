# V0.6.2.26 typed coverage运行协议

本协议在任何`.26`模型输出之前冻结。新根`/raid/hfang/ihb/runs/v0.6.2.26-typed-diagnostic`，
不重启旧run、不复用旧judge cache、不将旧非法响应按新合同重新宣告首轮成功。

## 两臂与变量

control为`.25`原完整命题提示、先证据后status的flat v2输出；coverage使用新的typed v3输出。
候选从冻结`.25`system中逐字嵌入ESTABLISH THE ACTUAL RETAINED SUBJECT至DIRECTIONS AND STRICT OUTPUT之前的
语义规则/成对边界块；新system/pair/YAML只另行说明typed分支、harmless归属和编译规则。
这测试编码合同及对应输出说明的变化，不把它称为完全逐字相同的prompt，也不混入新的标签修订。

两臂均应用同一`.24`scope后处理、固定`.14`main和原ownership路由。
flat control不是`.25`那次实验的旧control，更不是`.14`record-binding critic；报告须明确其角色。
公共输出仍为原v3。source NO/U、visible exact和不完整输入分支完全保留，不伪造critic proof。

typed合同严格区分COVERED/UNCOVERED/UNRESOLVED，status由模型决定；模式、inactive字段与replacement方向由程序派生。
harmless只能标注整个意义无保留内容的own unique ID；混合片段只要含保留命题就不得整体标harmless。
未标harmless的unique存在时，必须有对侧实际语义support；全harmless或无unique时允许解释性context，
该context记录在原始结果和编译审计，不变成语义counterpart。选择存在的ID不证明模型的蕴含结论。
UNCOVERED仍需独立own unique source、对侧实际counterpart或明确缺失、loss type、实际比较关系和保留后果。
typed原始结果不得携带其他分支字段，包括非法null；v1/v2原验证完全不改。

## 真实链路和传输

必须先验证两臂各first/encoded retry，共4个完整Ray→native parser→本地HTTP→writer→binder用例。
同批次混合covered/uncovered/unresolved及双向保留/真实单向扩展/LOW U；无请求分支独立保留。
实际请求必须与预渲染消息相同，输入全文、反馈及原始span偏移都保持。
native默认pruning结果不可信，始终对原始assistant文本pruning=False解析后再绑定落盘列。
只允许已知其他typed分支字段的Arrow null padding；非null新增、丢字段、改status/文本、未知字段都拒绝。
这只绑定传输表示，不修改原始响应或旧结果。每个输出保留raw、compiled、transport、scope来源摘要。

## 输入、预算、调度

同258对、230种可见payload、原分层权重与reference、相同两个repeat顺序；不去重或删弃难例。
两臂各147个实际called、111个确定输出，均进入258公共评分分母；重试分母仅实际called。
同NVIDIA Qwen27B、temperature0/top_p1、输出4096、关闭thinking、timeout600、max_retries2、并发16。
逐对检查实际渲染input+4096+2048≤32768，不截断或跳过超预算pair。

冻结所有代码/资源/测试/schema、fixed main、call sets、消息摘要、token预算和边界证据后才运行。
先control后coverage，各取R1实际called前8个作预检；两臂须8/8首轮有效、零native/outer纠错、各HTTP200恰好8。
预检不发布语义分数、不作为正式响应复用。失败则停止，不重启一个已观察的实验。
正式R1 control→coverage，R2 coverage→control；1032公共输出，588基础模型请求，另16预检请求。
任何终态cell或确定性输入/传输/消息边界失败停止后续schedule；所有失败与实际attempt保留。
HTTP服务限流/失败/重发与模型native+outer纠错分别记录。

## 评分及晋级

完整schedule才比较，两轮全部保留，不择优。原local_gates、38/22负例、20负例/3benign底线、
4true containment、12exact、54translation与H0347/H0453/H0748三目标全部继承。
两轮各自candidate weighted primary不得低于同步control、over-group不增，两轮primary翻转不得超过control。
每臂最终严格schema/evidence完成100%、native+outer重试pair并集/实际called≤1%，不能由编译器通过数冒充模型首轮成功。

全通过才支持fresh完整Judge局部验证；完整1000仍需加权75/75/79、over-group≤66和分类/重试保护，
之后未见400独立人工holdout，最后20000。reference含预测可见AI修订的局限继续披露。
不更改reference或权重，不把AI复核当人工，不调用更大模型，不自动release、commit或push。
