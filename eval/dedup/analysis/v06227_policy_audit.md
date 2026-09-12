# V0.6.2.27：完成两轮同合同语义对照，未达到晋级条件

## 结论

**四个正式cell全部258/258最终有效、无终态错误，但本轮不晋级。** 完整命题规则块相对简短规则块，
两轮加权主决策一致率分别提高1.71、0.94个百分点，over-group数量不增；然而加权precision/recall仍低，
负例与benign保护未通过，候选重复翻转14对多于control的10对，四个cell的实际纠错率均超过1%。
不提交fresh完整Judge局部验证、完整1000、holdout或20000，也不发布release。

见[冻结协议](v06227_design.md)、[严格原始输出重放报告](v06227_policy_assessment.json)及
[错误/运行归因](v06227_response_attribution.json)。
运行根 `/raid/hfang/ihb/runs/v0.6.2.27-policy-diagnostic`；
合同 `2369ce14f3c3a04417c97714420c22428aa111464ea83f3b8e80fc9df555f62f`；
2026-09-10 19:15:51–19:24:46 UTC。

## 同一typed任务，仅替换声明的语义块

control使用`.23`NONEMPTY RECORD ANCHORS至SCOPE AND LOSS TYPES之前的简短规则块；
coverage使用`.25/.26`完整命题/实际主体/具体方法规则块。共同system框架、pair、YAML rubric、typed v3、
编译器、严格binder、scope后处理、ownership、固定main、标签/权重及模型参数全部相同。
共同pair/rubric本来就含简短consent、Stories、identity、method提示，所以这不是纯`.23`对`.26`版本比较。

147个实际called输入逐条通过单一变量验证：去掉声明块后两臂system完全相同、user消息相同，
coverage初始渲染与`.26`typed候选相同。最大输入token为17027/17363，没有截断或删除输入。
冻结`policy_contrast.json`保存块原文、来源摘要、每条请求及共同消息摘要。
两臂同27B、temperature0/top_p1、输出4096、关闭thinking、并发16；没有调用更大模型。

原258对含230种可见payload，147对调用critic、111对由既有路由确定，全部进入公共评分分母；
未调用分支没有伪造critic proof。两个repeat分别control→coverage、coverage→control，均fresh生成。
这仍是固定main的错误富集开发子集，不是完整1000分数、独立随机校准集或未见holdout。
reference中预测可见AI修订的局限继续保留，不把它称为独立人工真值。

## 全部配对语义结果

下表P/R/primary均使用原分层权重；over/under/U为pair数。两轮不择优。

| cell | weighted P | weighted R | weighted primary | over-group | under-group | U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 control | 66.58% | 68.60% | 73.22% | 29 | 27 | 7 |
| R1 coverage | 66.70% | 69.73% | 74.93% | 29 | 26 | 6 |
| R2 control | 65.79% | 68.74% | 72.74% | 30 | 26 | 7 |
| R2 coverage | 64.99% | 68.20% | 73.68% | 30 | 27 | 6 |

候选非加权P/R分别为76.61%/78.51%、75.81%/77.69%，但不能拿这些数替代不达标的加权结果。
R1改变23对：12对修正、7对退化、4对改变后仍错；R2改变20对：11对修正、7对退化、2对仍错。
candidate跨repeat主决策变化14对，control10对；不能把第二轮三目标全对解释为稳定修复。

- 四个真实containment保护均4/4，12个exact均12/12。
- 候选translation加权primary两轮均89.01%，相对`.12`保护基线均−0.80pp，通过−3pp门槛。
- 三目标候选R1为2/3（H0748未过），R2为3/3；control两轮均1/3。
- 候选20负例底线两轮均16/20，22个追加修复保护均9/22；38负例分别26/38、27/38。
- 候选3个benign底线均2/3（H0038未过）；全部5个benign分别2/5、3/5。
- 候选identity_slot加权primary相对同步control下降18.66pp、4.62pp；不能以翻译改善抵消身份退化。
- 公共输出只有LOW/MEDIUM：候选MEDIUM各252条的加权主决策正确率为76.65%/75.35%，
  LOW各6条为20.79%。这是当前开发子集的实测档位表现，不是经过独立校准的概率；没有HIGH样本可估计。

## 主要错误不只剩下输出格式

候选over-group的cohort分布R1/R2为：page-role 8/8、identity-slot 6/5、factual-state 5/5、
boilerplate-only 4/5、meaningful-addition 4/5，另各1个list-membership和material-content。
错误relation分别为10/10个CONTAINMENT、19/20个NEAR_SURFACE，不能只统计containment就宣称总体改善。
参考containment仍有14/16个被判不合并，真实扩展召回仍需保护。

可定位的机制包括：

- H0119：A有具体法律书籍编号/价格，B只有同店铺导航。candidate的逐侧解释已经承认B没有这些商品，
  但scope把共同商家当作共同具体记录，仍得到CONTAINMENT。这违反非空记录要求，不是引用ID不存在。
- H0419：商品夹克描述对仅共享配送/退货信息，类似地把商店政策当作该商品的共同记录核心。
- H0907：不同商品组ID对应同样的报错模板。候选R1识别实际ID差异，R2又将其当模板变量忽略，暴露身份不稳定。
- H0809：不同服务/组织名称的报错页，候选将名称归为branding而判等价；按现有identity reference计错。
  不由本轮自行重标，也不推广为“所有品牌或导航文字差异必须冲突”。
- H0480：将一侧独有的法律告诫归为harmless footer，并以大范围ID引用声称完整覆盖。
  现存引用只能证明来源位置，不能证明完整命题蕴含或harmless归类正确。

一次只要求固定main双侧profile为substantive_main的代码收紧不会解决这里的containment误合并：
两臂四个cell的全部错误containment都已满足这一profile组合（13/10/11/10对）。
因此不实现一个在本批错误上没有作用的profile条件，也不把旧main分类当作共享具体记录的证书。

## 运行质量和原始响应可观测性

两臂预检均8/8首轮有效、零纠错，各8次HTTP200；通过后才提交正式schedule。

| 正式cell | outer重试pair | 留存trace的native纠错pair/次数 | native+outer pair并集/实际called | HTTP200 | HTTP其他 |
| --- | ---: | ---: | ---: | ---: | --- |
| R1 control | 5 | 3/3 | 7/147 = 4.76% | 155 | 无 |
| R1 coverage | 2 | 6/6 | 7/147 = 4.76% | 157 | 10×429 |
| R2 coverage | 4 | 4/4 | 6/147 = 4.08% | 155 | 1×500 |
| R2 control | 3 | 1/1 | 3/147 = 2.04% | 151 | 无 |

合计645次外部HTTP：634次200、10次429、1次500。计划基础请求为588正式+16预检，额外成功请求30次；
留存trace可数14次native纠错及14次外层重试，另2次成功HTTP没有对应的落盘assistant trace，不能隐藏或伪造补齐。

这2次差额与R1候选H0066的native失败相联系：输入索引35与worker日志row35匹配，该行首次完全未落盘。
日志保留了反复循环的scope_explanation、缺anchor_a_ids及达到max_tokens的解析失败说明；
外层重试重新返回有效结果。日志路径/摘要与局限保存在归因文件。
该pair已计入outer重试并集，不能把缺失trace算作无纠错；也不能声称所有失败assistant文本都能逐条重放。
机器报告中的native次数指**留存trace中观察到的次数**，不是完整的服务响应档案。

其余13个outer失败为严格原始响应ParserException。R1 control的5个均在UNCOVERED结果中携带了
COVERED分支字段；默认native pruning能产出对象，原始严格binder仍正确拒绝。
没有删字段、补空值或用最后一次成功覆盖首次失败。HTTP服务问题与上述模型合同问题分开呈现。

## 验证和下一步

[边界报告](v06227_boundary.json)：756个CPU测试通过、26个自有Ray测试跳过、2个GPU排除；
本轮4个完整native链路单独通过（模块22项），12次本地HTTP、0外部调用。
新增测试覆盖两臂实际typed列的native纠错统计、严格原始分支、注入隔离、重试反馈及公共方向。
一次单元断言曾将judge_retried_pairs误当outer-only，改为验证并集后通过；生产统计未修改。
自有Ray head PID275071已退出，52652端口已释放；未停止其他集群。

见[`.28`待实施设计](v06228_design.md)：保留语义内容与严格合同，改变“具体损失/对侧核心核对、scope归属、
分支字段”的生成顺序，检验是否同时减少空记录式containment与跨分支输出。
这只是下一轮可证伪假设，不保证仅调整顺序就满足身份和召回要求。
所有旧结果/标签/权重/门槛保持不变；不自动release、commit、push或扩大模型权限。
完整1000的75/75/79、over-group≤66及分类/重试门槛，之后独立人工400 holdout与20000目标仍未完成。
