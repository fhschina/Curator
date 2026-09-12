# V0.6.2.23 审计：证据选择完成率通过，语义、保护和重试门槛未通过

## 结论与比较范围

**不晋级，不运行fresh完整Judge局部评估、完整1000、holdout或20000。**
本次同27B、temperature=0、同payload与固定`.14`main的两轮配对schedule全部完成。
内部合同改为选择原文span，最终严格schema/证据完成率达到100%；这项工程结果不等于语义质量达标。
candidate的加权primary两轮均略低于各自control，漏判与翻译保护变差；两轮primary翻转17对，control仅6对。

运行根 `/raid/hfang/ihb/runs/v0.6.2.23-selection-diagnostic`；
合同 `d0b58ed03bceee36d472140f6a403369eb41ffce557c74bfcf56b231d8a3e330`。
正式运行2026-09-10 17:43–17:53 UTC。
[机器报告](v06223_selection_assessment.json)逐项重放原始响应摘要、实际attempt/反馈、native请求和确定路由输出。
所有258对和原权重均保留；258对只有230种可见payload，不把两轮516个输出称为516个独立样本。
这些是反复使用过的局部开发集诊断，**不是完整1000的成绩，也不是独立人工holdout**。

## 加权与非加权语义结果

| cell | 加权precision | 加权recall | 加权primary | 非加权precision | 非加权recall | 非加权primary | over-group | under-group | UNRESOLVED |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 control | 68.60% | 62.50% | 74.51% | 79.65% | 74.38% | 77.52% | 23 | 31 | 4 |
| R1 selection | 74.68% | 65.37% | 74.45% | 81.44% | 65.29% | 70.54% | 18 | 42 | 20 |
| R2 control | 68.05% | 63.68% | 73.98% | 78.26% | 74.38% | 76.36% | 25 | 31 | 4 |
| R2 selection | 76.28% | 59.11% | 73.60% | 82.95% | 60.33% | 69.77% | 15 | 48 | 20 |

不选择R2的precision作为单一版本成绩。candidate加权primary相对control为−0.053/−0.378个百分点；
over-group虽下降，recall的配对变化为+2.87/−4.57个百分点，并不稳定。
权重仍为stratum_population_n / stratum_sample_n，权重总和2072.772211350298，
不是局部抽样频率或按错误类型再加权。UNRESOLVED保留在primary与适用的漏判分母中。

| 保护项 | R1 selection | R2 selection |
| --- | ---: | ---: |
| exact | 12/12 | 12/12 |
| 四个真实containment | 3/4 | 3/4 |
| baseline负例floor | 17/20 | 17/20 |
| 全38个critic修好负例 | 30/38 | 31/38 |
| 额外22个负例保护 | 17/22 | 17/22 |
| baseline benign floor | 0/3 | 0/3 |
| 全5个benign | 0/5 | 0/5 |
| H0347/H0453/H0748三目标 | 0/3 | 0/3 |
| translation加权primary相对`.12` | −16.01 pp | −16.81 pp |

H0333真实扩展两轮均丢失；H0748仍被错误判为单向containment而不是双向保留。
baseline负例失守H0964、H0959、H0419；额外负例失守H0132、H0141、H0419、H0959、H0964。
完整cohort矩阵、方向错误、逐对配对变化与reference见机器报告，不能只看duplicate group布尔值。

## 实际调用、schema与服务重发

并发由旧64在新合同中预先对两臂同步设为16；其余模型、温度、输出4096预算不变。
preflight均按冻结的实际called顺序取8对，零native/outer纠错。

| cell | 公共有效/总数 | 实际调用pair | 确定路由 | native纠错pair | outer重试pair | 并集/实际调用 | HTTP200 | HTTP失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| preflight control | 8/8 | 8 | 0 | 0 | 0 | 0/8 | 8 | 无 |
| preflight selection | 8/8 | 8 | 0 | 0 | 0 | 0/8 | 8 | 无 |
| R1 control | 258/258 | 258 | 0 | 3 | 0 | 3/258 = 1.16% | 261 | 429 × 5 |
| R1 selection | 258/258 | 147 | 111 | 0 | 2 | 2/147 = 1.36% | 151 | 无 |
| R2 selection | 258/258 | 147 | 111 | 0 | 0 | 0/147 | 147 | 500 × 1 |
| R2 control | 258/258 | 258 | 0 | 3 | 0 | 3/258 = 1.16% | 261 | 503 × 3 |

总计845次外部HTTP请求：836次200、5次429、1次500、3次503。
正式810个基础模型请求、另16个预检；10次额外成功响应来自native或outer纠错。
服务失败由同一运行的请求层重发成功，不是新运行或挑选较好响应；这些失败另列，不能宣称“所有重试为零”。
R2 selection的一次500在约340秒后返回。两臂control与R1 selection仍超过1%的既定模型纠错门槛。

candidate每轮147/147最终严格模型合同完成、258/258公共完成、0终态错误。
111个无请求输出为97 main NO、12原文exact、2不完整→LOW U；不能把这些行加进实际模型重试分母。
R1两个失败pair都直到第三次才通过，即147+2+2=151个outer请求，所有早期失败响应保留。

## 技术进步与仍然存在的判断错误

内部`dedup-retained-coverage-v2`只让模型选择source_span_id/counterpart_span_id；
程序严格从原span提取完整原文，保留原始偏移、unique归属、长度及双侧证据验证。
不抄写quote，不做Unicode修正、拼接或模糊匹配，不把存在的ID当作蕴含证明。
schema直接生成字段说明，移除了由确定路由负责的IDENTICAL_TEXT选项；公共v3未变。

- H0119不再因泰文抄写终态失败，首轮no/no；H0485重复段落首轮双向NEAR_SURFACE，不再伪称字节exact。
  这是已见样本的改进，不代表整个版本胜出。
- H0333首两次同时写COVERED/MIXED与MAIN_CONTENT损失证据，解释末尾却说UNCOVERED；严格拒绝正确。
  第三次生成一致的COVERED/HARMLESS_ONLY，格式通过但错误地丢弃实际PhD申请联系人、邮件/主题要求。
  不能从旧解释中选出一句有利结论覆盖最终标签。
- 另一重试pair在B没有任何unique span时声称B有UNCOVERED POLICY_MEANING，source_span_id为空，
  却引用本来两侧共享的S017来宣称A缺失政策；严格拒绝正确。这是空损失证据，不是字符抄写问题。
- control R1首响应H0060是无fence的裸JSON，H0837是fenced但结构损坏的JSON，H0968是XML式字段文本。
  R2 H0060裸JSON、H0761外包response_schema标签、H0968 XML式文本。
  不能将六次纠错都归为一个可安全自动清理的包装问题。

## 20条UNRESOLVED的路由归因

每轮2条是确定路由的不完整输入H0239/H0751，其余18条均为main profile与critic scope不一致：
main双侧NON_MAIN_ONLY而critic选择SAME_SUBSTANTIVE_RECORD，或反向的类别不一致。
这18条critic均已给出resolved方向，而不是自行弃权。

其中既有两侧已经COVERED的完整翻译，也有明确的identity/state/policy损失，
以及仍不应直接允许的单向MAIN_CONTENT增删。H0186两侧私密个人主页的姓名不同，critic却称HARMLESS_ONLY；
它也在分类矛盾中被转为U。因此不能只按translation标签挑选恢复正例。

下一步[`.24`离线协议](v06224_design.md)先隔离这个代码级gate的影响，
对全部两轮258对使用同一预声明规则，包括暴露出的identity误合并，绝不修写旧main或critic字段。
此外，固定main已把12个reference YES放入NO/U确定分支；理论完美critic的当前加权recall上限约86.59%。
这不是预测成绩，也不是75%不可能；说明最终还必须验证fresh main，而不能无限只调critic。

## confidence、验证及局限

candidate两轮LOW均20条，实际primary正确1/20（5.00%，加权14.07%）；
MEDIUM均238条，primary正确率76.05%/75.21%（加权77.33%/76.44%）。
没有HIGH；这些是已见局部开发集条件频率，不能解释成独立校准概率。
完整control及逐档数据保留在机器报告。

冻结前627个CPU测试通过，14项自有Ray测试明确跳过、2个GPU测试排除。
本版4个完整Ray/HTTP first/retry边界另行通过，模块共14项通过，共8次本地HTTP、0外部请求；
见[边界报告](v06223_boundary.json)。运行结束后仅停止本次自有Ray head PID70364，未操作其他集群。
所有历史冻结文件、reference、权重、缓存与原始结果未改；没有release、commit、push或较大模型调用。

reference仍含134条曾看过预测的AI修订，不能称为独立人工真值；未批准的reference争议仍不改变。
完整开发集75/75/79、误合并≤66、分类保护、重试与独立400 holdout要求均不变。
