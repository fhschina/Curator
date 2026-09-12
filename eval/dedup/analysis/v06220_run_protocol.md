# V0.6.2.20 在线 coverage 诊断冻结协议

## 适用边界

这是 [原设计](v06220_design.md) 的执行补充，不覆盖其离线准备历史。
原 preparation.json 保持不变；正式 manifest 重新固定当前代码、提示、schema、测试与本协议。
本实验同时改变 critic 合同与仲裁，只能评价该组件组合，不声称隔离了 prompt 单项收益。
固定 .14 main，不调用较大模型、不改 reference、不读 holdout、不发布 release。

冻结前须通过真实 DataDesignerStage → 本地 HTTP 端点 → 原生解析的两条请求测试。
这不是完整 Ray executor 或外部模型集成测试；完整执行链由下面的在线技术预检验证。
原生 trace 的 assistant content 实际为 text-block 列表，读取时逐块无损拼接，
不允许混入非文本，不允许利用 NDD 的字段 pruning 掩盖原始 schema 失败。

## 输入、调用与顺序

- 唯一运行根：`/raid/hfang/ihb/runs/v0.6.2.20-coverage-diagnostic`，缓存和 checkpoint 全新。
- 沿用 .18 已冻结的全部 258 对、原始权重与 .14 main；重建原始 span packet，删除 .18 派生上下文。
  不传标签、review ID、主模型预测、旧 critic 或抽样原因给模型。
- control 为 .14 critic + v8；coverage 为 `dedup-retained-coverage-v1` critic + 新仲裁。
- 同一 27B 模型和 endpoint；temperature=0、top_p=1、输出上限 4096、关闭 thinking，
  并发、timeout、外层重试上限沿用 parent settings。只重试未通过校验的 pair。
- 冻结前对两臂全部 258 对使用实际 renderer 与固定 tokenizer 再检查完整输入预算。
  输入 + 4096 输出 + 2048 安全余量不得超过 32768；不能截断或删掉超长样本。
- 先 control、再 coverage，各取 repeat 1 固定顺序前 8 对技术预检。
  两臂均必须 8/8 有效、0 终态错误、0 外层重试、0 已观测 native correction，
  且 HTTP 状态恰为 8 次 200。任一不满足，就停止正式提交，保留全部记录。
- 两个预检通过后，正式 R1 control → coverage、R2 coverage → control，每 cell 全部 258 对。
  正式基础请求共 1032；16 条预检独立保留，不回灌、不复用、不计入正式分数。
- 任一正式 cell 不完整则停止；运行失败不原地重启、不补挑最好输出。

## 校验与重试口径

每个 coverage 输出必须有完全一致的 pair ID、原始 payload 与 payload digest，
仅有新 critic 列；原始最终 assistant 文本通过非 pruning 原生 schema 校验，
并与 parsed column 一致。随后逐项验证内部 coverage 引用、公共 v3 与双侧 evidence offsets。
正式重放按 pair ID + response digest + trace digest 精确定位实际接受的原始响应。
不能取最有利的一次 retry；每个 cell 的输出、错误、原始 trace、事件及终态 ID 均固定摘要。
失败样本不得从分母消失；技术预检失败不发布语义准确率。

分别报告外层 retried pairs、native corrections、native-corrected pairs 和两者并集。
coverage 的并集重试率也必须 ≤1%，不能把原生结构纠正隐藏在外层零重试中。
旧 control 未保存同等 native trace，因此该项为不可观测而不是实际零；
HTTP 请求数独立报告，预检额外请求亦视为不通过。

## 固定准入条件

复用原全部 local_gates、38 条负例、22 条额外保护、.12 的 20 负例/3 benign 底线、
4 containment、12 exact、54 translation；H0347/H0453/H0748 三项均须正确并审查理由。
两次 coverage 的 weighted primary 均不得低于当轮 control，over-group 不得增加，
每轮均须通过原局部门槛及上述并集重试率门槛。
coverage 两次 primary 翻转数不得超过 control；不能挑最好一次晋级。

报告保持原始标签与权重，包括现有 reference 中 prediction-aware AI 修订的局限；
这些数据不能称为独立人工盲审。完整诊断通过也仅支持另一次 fresh 完整 Judge 局部验证。
之后才可考虑完整 1000 条开发集：weighted P/R ≥75%、primary ≥79%、over-group ≤66，
原分类与运行质量门槛不变，再走未见 400 条独立人工 holdout 流程。
