# V0.6.2.29：复合包含的新鲜主判断与 critic 分解诊断

目的：把用户确认的 [X＋独立 Y 包含 X](composite_containment_policy_v1.md) 落实为可执行候选，
并区分主 Judge 漏判与 critic 误否决。该版本是 opt-in 实验，非正式 release；不覆盖旧配置、输出或缓存。

## 契约与职责

- 新内部 `dedup-retained-coverage-v4`，继续导出公开 `dedup-judge-output-v3`。
  用 `shared_basis` 明确共享实质 X、non-main 消息、无实质共享和未决；不再暴露旧 `record_scope`。
- 两个 typed side 分别记录 A 的含义是否被 B 保留、B 的含义是否被 A 保留。
  `COVERED` 不意味着该侧能替代对侧：adapter 按正确的反向对应派生两个 replacement direction。
- 独立 Y 的未覆盖实质内容为 `MAIN_CONTENT / NO_EQUIVALENT_FOUND`；不要求 Y 属于 X 的具体记录。
  共享非空 X、恰好一侧实质缺失且无实际冲突时才是 containment。
- `hard_conflict` 单列实际冲突类型、A/B 原文 span 与解释，必须与一个具体 `UNCOVERED / CONTRADICTS`
  方向证据吻合。不能用“不同标题/独立板块”直接替代身份或角色冲突证明。
- 主 Judge 与 critic 共用一份 versioned policy，system/pair/YAML rubric 同步。
  主 Judge 看原文和差异证据重新判断，**不读取旧 `.12/.14/.28` main**。
  critic 仅审核本次有效正例，接收精确绑定的新主 Judge 证书；它的输出与主 Judge 输出分别保留。
- 完整原文相同由确定性路由处理；截断、空白或不完整 packet 由 LOW UNRESOLVED 路由处理。
  exact 的 overlap 标为可核验的 LOCAL_PASSAGE，并明确不推断正文/页面角色，不把相同 cookie 自动标为正文。
  新主 Judge 的 NO/UNRESOLVED 不调用 critic，保持介入范围不额外扩大。H0822 必须由新主 Judge 重新判断。
- 历史 V1/V2/V3 不可作为 V4 响应使用。借用旧 typed compiler 只校验逐侧 inventory/证据，
  内部 neutral scope 占位不参与语义判断，绝不调用旧 same-record adapter 决定 V4 verdict。
- 非 exact 的公开结果保留双侧原文精确 evidence，最多四段；完整翻译双向且 material NONE。
  resolved confidence 暂派生 MEDIUM、未决 LOW，不声称校准或给出数值概率。

运行入口为 `analysis.composite_experiment` 和 `judging.composite_runtime`。
旧全局 presets/source digest 被历史实验冻结，故不向其旧 runner 注册不兼容的 V4 配置；
新入口自己校验 `dedup-judge-hs-v0.6.2.29`、V4、公开 v3 及 policy 版本，不会退回旧 adapter。

## 工程边界

复用现有 cell engine 的唯一输入、原始 trace、严格 parser（禁止 pruning）、native/outer retry 统计、
完整落盘和按 accepted attempt/digest 精确 replay。不把非法原始响应的 null/跨分支字段删除后当成成功。
允许的 Arrow null padding 只限原始有效对象在 writer echo 中的已知 typed 分支字段。

主/critic 请求均包含同一份完整原文和 evidence，标签、选择原因和版本成绩不进入请求。
critic 的 main claim 使用带前缀的规范 JSON 文本，避免 NDD 将 JSON-looking seed 字符串重新解析成 struct；
实际 request、回显 claim、payload hash、反馈均与输入核对。

先跑四个真实 Ray→本地 HTTP 模拟服务→NDD→writer→binder 边界：主/critic × 首次/带反馈。
模拟返回不是线上模型能力；运行时使用 `-s`，避免 pytest 捕获的日志流被 Ray actor 序列化。
只使用本次专用 CPU Ray 集群，不使用 `ray stop` 影响其他任务。

## 预先固定的小范围在线诊断

模型和生成条件沿用 `.28`：27B FP8、temperature 0、top_p 1、输出 4096、thinking off、并发 16、
timeout 600、outer max_retries 2。每个实际 main/critic prompt 都在提交前检查
`input_tokens + 4096 + 2048 <= 32768`；不得截断、删除超长案例或沿用旧 main 过关。

1. 先做 8 条合成预检：独立扩展双向镜像、chrome、同记录字段增补、cookie 空锚点、SKU、状态、翻译。
   要求 main/final 的 8 条主决策全部正确，所有 requested 响应首轮有效且 HTTP 200，native/outer 纠错均为 0。
   若失败，保留失败结果，停止正式样本；不在同一冻结 run 中改 prompt 后继续。
2. 通过后做两轮相同 50 对，每轮 fresh main→实际被路由的 critic，顺序预先固定。
   24 条真实开发案例 = 原 22 条明确/保护面板 + H0521/H0822；保留原标签、原分层权重。
   26 条合成材料 = 13 类及 A/B 镜像，单独评分，绝不和人工开发权重混成一个 P/R。
3. 两轮都报告。对每轮同一份 main 比较 main-only 与 main+critic，输出逐类增益/损失与具体 IDs。
   `.12` 相同 24 条历史结果仅作背景对照；本版本同时换契约和 fresh-main 实现，**不是仅改一句 prompt 的因果实验**。
4. 总基础请求上界 216（含预检、两轮、双阶段；实际路由通常更少）；额外 native/outer 纠错和失败 HTTP 全部记录。
   重试并集率仍须 ≤1%，不能靠最终有效率掩盖反复修复。

这 24 条是预测可见开发选择、两条 policy 应用是 AI 复核，不是新独立人工 gold。
另外 17 条边界争议不自动改标、不强制进准入计分；此轮也不裁决 cookie 口径、裸标题等其他争议。
合成样本和小开发面板不能替代 full-1,000 weighted 75/75/79、over-group≤66、分类保护与独立 holdout。
不运行 20,000 对、不发布版本、不 commit/push。

## 命令

```bash
python -m eval.dedup.analysis.composite_experiment prepare \
  --root /raid/hfang/ihb/runs/v0.6.2.29-composite-diagnostic \
  --boundary eval/dedup/analysis/v06229_boundary.json

python -m eval.dedup.analysis.composite_experiment run \
  --root /raid/hfang/ihb/runs/v0.6.2.29-composite-diagnostic
```

`prepare` 只做离线校验/冻结；`run` 才使用已配置的 NVIDIA endpoint 和凭证。
需存在专属 `RAY_ADDRESS`，不得让实验隐式占用/停止别的任务的 Ray。
