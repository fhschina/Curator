# V0.6.2.22 确定路由与单段 witness 配对协议

承接 [设计](v06222_design.md)。使用可配置 coverage_experiment 入口；冻结后不改资源、代码、
main、标签、输入或预算，失败不重启，不复用旧缓存或技术预检响应。

## 调用、输出与分母

仅 coverage 使用 dedup-coverage-call-routing-v1，在请求前仅观察 main 与原始 payload：
incomplete、main NO/U、完整可见 exact 分支保持原 coverage 仲裁；所有其他正例必须新请求。
每个正式 cell 仍有258个公共输出与评分分母，保留原重复 payload 对和权重。
未请求行明确 NOT_REQUESTED_OWNED_BRANCH；不生成 critic proof、原始响应摘要或 schema 成功声明。
预计 coverage 实际请求147对、确定分支111对；最终以冻结的逐对路由清单为准。

control 保留原 v8 的全部258个请求，使用 .21 仅增加 trace 的配置。
离线比较 .21 control 的两个 main-NO 样本发现 reason_codes、evidence 不同于 coverage 确定路由，
故不声称 control 可以兼容跳过，也不改变其公共结果。两臂调用数不同，语义评分分母相同。

分别报告公共完成率、实际 called-set 的最终严格 schema/quote 完成率、外层重试、native correction、
native/outer 重试 pair 并集、HTTP 次数。重试 gate 的分母为实际 called pairs，不能用未请求行稀释。

## 输入、提示、顺序及技术准入

使用 .21 冻结的原258开发对、.14固定 main、相同两轮顺序、原标签和权重。
同 NVIDIA 27B、temperature=0/top_p=1、输出4096、关闭 thinking、并发64、timeout600、max_retries=2。
不调用更大模型，不读取 holdout，不声称这是 fresh 完整 Judge 结果。

coverage 的 system/pair/YAML 同步单段精确引文和完整字段示例；严格 parser、输出契约和仲裁不变。
一条 quote 必须完整来自一个引用 span，不能拼接不连续片段。所有13个侧字段必须显式输出，
不静默填空或清理字段；无关独立片段在说明中交代，真实损失不改为等价来逃避证据。

冻结重复1/2的完整清单、实际调用清单、确定分支公共结果、main/hash、实际初始消息摘要、token审计和资源。
全部实际调用输入+4096+2048≤32768；仅为客户端预算，不裁剪或移除超长样本。
预检按各臂 R1 实际调用序列的前8对，按既有确定顺序选择，不根据标签、失败史或难易选择。
这些是开发样本，不宣称未见。先 control 后 coverage；两臂8/8、0 errors、0 native/outer重试且8次HTTP200
才提交正式 cell。预检不进入语义评分且不复用其响应。

正式 R1 control→coverage，R2 coverage→control。每臂每轮258公共输出，共1032。
预计基础模型请求810次，另预检16次；实际重试另计。不择优挑选轮次。
任何确定性 payload/feedback/实际提示/trace/路由边界错误停止，不用模型重试修复。
模型 schema/quote 错误仅对失败 pair 有界重试；任一正式 cell 有终态错误则停止后续提交。

## 校验与评分

冻结前通过两臂首轮/重试四个完整 Ray→本地 HTTP→writer→binding 测试，不调用外部模型。
每次绑定均检查严格原始响应、完整原文传输、真实初始消息与 renderer 一致。
回放按 pair + 实际 attempt + 原始行 digest 定位，保留真实反馈，不选择最好答案。
确定路由回放必须逐字段等于重新计算的公共输出，不允许附加虚假 response/proof。

完整 schedule 后才比较语义分数。沿用 local_gates、38/22负例、.12的20负例/3 benign 底线、
4 true containment、12 exact、54 translation 保护及 H0347/H0453/H0748 三目标。
每轮 coverage weighted primary 不低于当轮 control，over-group 不增加；两轮翻转数不超过 control。
每臂重试并集/实际 called pairs≤1%；其他完成率分母仍是258。任何原 gate 不降低。

技术预检失败不发布其语义准确率。结果最多支持下一次 fresh 完整 Judge 局部验证。
完整1000的75% precision/75% recall/79% primary、over-group≤66、分类保护、独立人工400 holdout
与最终20000仍须逐层通过。reference 含预测可见的 AI 修订，不称独立人工盲审。不自动 release/commit/push。
