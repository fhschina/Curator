# V0.6.2.28 待实施设计：先核对保留命题，再绑定共同记录

本设计在`.27`完整两轮评估之后提出，尚无`.28`运行或成绩。
`.27`候选加权P/R为66.70%/69.73%、64.99%/68.20%；虽保住4个真实containment、translation底线，
仍有29/30个over-group、负例/benign保护失败及4.76%/4.08%的模型纠错并集率。
不能把下一轮目标缩成schema completion，也不能仅因非加权P/R超过75%就运行完整开发集。

## 可证伪假设与优先级

H0119的scope先认定“同商家即同记录”；后续逐侧解释承认只有A有具体书籍/价格，却仍输出containment。
H0419也把配送/退货政策当成独有夹克描述的共同记录核心。模型存在“先选择较宽共同对象，再吸收真实损失”的风险。
全部错误containment的固定main双侧profile已是substantive_main，新增同名profile条件在这批错误上不会改变结果。

另一方面，R1 control有5个UNCOVERED原始对象残留COVERED分支字段；当前展示顺序在status之前生成分支专属字段。
因此下一轮的单一处理是**证据核对与决策序列的重排**：先完成逐侧比较，逐侧status在专属字段之前，
共同record_scope在逐侧比较之后。保留完整命题规则内容，不再同时新增一长串语义例外或修改标签。
此假设可能失败，尤其不能保证解决实际身份、服务主体及页面角色判断；需如实保留结果。

## 合同与提示呈现

继续使用内部dedup-retained-coverage-v3与原编译器/strict binder/公共输出，不引入宽松parser或新语义接受条件。
只改变schema展示中的properties/required顺序，字段集合、枚举、互斥分支、必填、证据约束全部不变：

- root先contract_version/input_status，然后a_meaning_in_b/b_meaning_in_a，最后独立anchor、scope_explanation、record_scope。
- side先reviewed_unique_ids、简短coverage_explanation，再status，最后仅该分支专属字段。
- scope核对必须回答：逐侧实际保留或缺失的是哪个具体记录？对侧有什么独立的同记录核心？
  不能改选共同站点/商家/法律模板来制造该具体商品、文章或列表在对侧也存在。
- 同一FAQ的新增步骤、同一招聘记录的新增具体申请方式仍是扩展；同正文加chrome和完整翻译仍是双向覆盖。
  身份、服务对象、真实页面角色/状态/成员冲突仍按原规则处理，不把普通按钮、导航或语言变化自动升级为冲突。

system/pair/YAML同步描述新生成顺序；schema字段指南必须从实际重排后的schema生成，不能显示旧status-last清单。
共同`.27`完整命题policy块保持逐字相同，报告明确这是“生成顺序+配套编码说明”的干预，不声称所有消息逐字相同。
严禁从解释文字自动推断/改写status、删除跨分支字段、修改实际loss type，或把原始非法响应转成有效输出。
H0066的循环解释提示还需留意输出长度，但本轮不扩大4096预算来容纳循环，更不截断输入。

## 比较与验证

control为`.27`完整命题typed提示的原生成顺序；candidate为上述新顺序，两臂共同语义规则及算法相同。
这是对一个未达标实验基线的配对诊断，不称control为正式最佳版本或release。
新运行根、新合同摘要、新cache；旧`.27`四个cell与失败完全不动。

实现前后需验证schema语义约束完全相同：将properties/required顺序规范化后深度相等；
所有旧typed合法/非法对象的接受结果不变。原始字段出现顺序只作为呈现干预核验，不作额外成功条件。
两臂各first/retry，共4个真实Ray→HTTP→native解析→writer→binder边界先通过；
继续混合三种分支、真实containment、LOW U、未调用分支，并严格检验真实消息/反馈/输入全文。
增加原始跨分支字段回归，验证其仍失败；不能靠默认pruning接受这些对象。

然后冻结两臂代码/资源/测试、同258对、230种payload、固定main、原权重/标签、147 called/111 owned及原两轮顺序。
模型仍27B、temperature0/top_p1、输出4096、关闭thinking、timeout600、max_retries2、并发16；
实际input+4096+2048≤32768，不删除难例。预检各8/8首轮有效、零native/outer纠错、各HTTP200恰好8，失败即停止。
R1 control→candidate、R2 candidate→control，所有正式结果齐全才比较，两轮均报告且不择优。

主要观察空记录式containment、实际身份/page-role误合并、真实containment漏判、三目标和重复稳定性；
同时报告跨分支错误、循环/截断、缺失native输出、留存trace统计与全部HTTP请求，不把缺失trace解释为无纠错。
所有既有local/paired/38与22负例/20负例与3benign底线/4containment/12exact/54translation/三目标保护不变。
candidate primary不低于同步control、over-group不增、重复翻转不多于control；两臂100%最终完成且实际called纠错并集≤1%。

全部通过才支持fresh完整Judge局部验证，之后完整1000仍需加权75/75/79、over-group≤66和分类/重试条件。
本实验不替代400条未见独立人工holdout，不把AI复核称人工，不用proxy总分决定版本。
若顺序重排仍不足，下一步依据剩余错误重新选择干预，不能因已投入多轮就降低门槛或改reference。
较大模型、独立人工标注、重要口径修订仍需用户明确授权；不自动release、commit或push。
