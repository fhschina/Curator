# V0.6.2.17：只改变 critic 证据呈现的配对重复实验

## 范围与不可变合同

先复核 H0177/H0634，但仅保存 [待确认建议](v06217_reference_review.md)，不改变 reference 或权重。
本轮不注册生产 Judge、不调用主 Judge、不重跑完整 1,000 条、不读 holdout、不跑 20,000、不 release/commit/push。

沿用 .16 已冻结并验证的 .14 main、全部 258 对 payload 和 v8 仲裁，读取它们是固定输入控制，不是复用 judge cache。
不采用 .15 的 v9 仲裁或 .16 的 consent/pointer 补丁。旧脚本、配置、prompt、原响应和结果均不可修改。

两个实验臂：

1. control：逐字使用 .14 critic system/pair/rubric 和原 NDD 列名。
2. ordered：system、全部 rubric/options、所有 pair 语义规则均不变；只在 semantic_diff packet 内增加同一份
   document_a.text、document_b.text 的原始顺序全文。原有 span ID、范围和引用合同不变，没有外部补充内容。
   全文只展示一次；span 列表改为字符区间及头尾各 32 字符的短预览（不足 64 字符时完整显示），用作定位索引。
  省略号只标识预览压缩，不删原文。字符区间按现有 payload 的 Unicode 字符偏移，不伪称字节偏移。
  仅在完整且未截断时展示全文；不完整输入标 unavailable 并保留旧完整 span 展示，不截取新窗口或伪造全文。

定位索引在本地根据原 payload 确定性生成，逐段验证字符范围与原文字面相同，不推断语义或给样本打新标签。
两臂使用相同记录，其中增加 text-only presentation_index 和 presentation_full_available；control 模板不引用它们。
原 payload 与其 hash 不变；新的输入文件及派生索引单独冻结。保留原 NDD secure renderer 的安全限制。

新增呈现同时改变原文顺序可见性与 span 引用的展示密度；本对照检验的是整体呈现，不声称单独隔离“句序”因果。

冻结前的纯长度预检发现，直接追加全文且保留全部重复 span 引用需要 28300 输入 +4096 输出 +2048 余量，
超过 32768 客户端预算。该 [初稿](v06217_append_only_prototype.jinja) 和 [检查结果](v06217_append_only_budget_audit.json) 已保留；
没有创建 run root 或调用模型。随后才在任何候选输出出现前改为“全文一次＋定位索引”，不提高预算、不改样本或规则。

## 技术检查与样本

- reference SHA-256：`6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`。
- selection SHA-256：`977eb95ba066655d03ff533ba98d5c5fe92fa3cb3f63148856159f39d09c1f2a`。
- 原 258 对及原分层权重全部保留，含 230 份不同的完整 payload；不把重复 pair 当独立文本样本，不删除参照矛盾。
- model=nvidia/qwen/qwen3.8-27b，temperature=0、top_p=1、max_output_tokens=4096、parallel=64、timeout=600、max_retries=2。
  保持原 20000 可见 token 和 4096/512 窗口合同；只改变如何展示已存在的可见内容。
- 上线前用原冻结本地 tokenizer 和实际 NDD renderer/response recipe（含 rubric/schema 指令）计数。
  按本地 runner 的 32768 保守客户端预算预留 4096 输出及 2048 安全余量；这不是对 Hub 真实上下文限制的声明。
  超预算则在模型调用前停止，不静默截断、降输出上限或挑掉长样本。
- 两臂先各 8 对技术预检，不回灌正式输出。两次重复各 258 对：R1 control→ordered；R2 ordered→control。
  同一轮采用 .16 相同冻结顺序，总计 1032 次正式 critic 判断，另有 16 次预检，不含重试。
- 新 run root：`/raid/hfang/ihb/runs/v0.6.2.17-presentation-diagnostic`；Ray temp：`/raid/hfang/ihb/r17diag`。
  模型不看 main、review ID、标签、抽样理由或实验臂。所有请求参数、输入、main、资源、代码、协议和 token 预检均冻结。

## 冻结检查

每次重复完整报告 weighted/unweighted P/R、主决策、错误 containment、漏判、未决、全部 22/38 负例、
5 benign、4 真包含、12 exact、54 translation、目标原理由、同输入组内差异及重复波动。

只有 ordered 两次重复都满足以下条件，才可建议另立合同进行独立局部验证；本诊断本身不批准全量：

- 258/258 有效、0 终态错误、样本级 outer retry <=1%。不丢样本，不以未决当正确。
- 原 local gates：.12 的 20 负例及 3 benign 底线无回退，4 真包含和 12 exact 全对；
  translation 加权相对 .12 不低于 −3 pp；额外 22 条 critic 修复负例全对。
- H0347、H0453、H0748 三个目标全部正确；复核其原理由，不能凭一次恰好正确的标签宣布理解修复。
- 加权主决策不低于本轮 control，误合并数不高于本轮 control。
- ordered 两次重复的最终主决策变化条数不多于 control。

全部保留原评分口径；不因重复运行的 control 也失败而豁免保护，不选最好的一轮、不投票拼接。
失败就停止更大规模调用并报告；即使机器检查通过，也必须复核目标的语义依据，不能自动晋级。
两次重复只作描述性稳定性对照，不能声称统计显著性。
