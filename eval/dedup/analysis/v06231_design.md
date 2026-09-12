# V0.6.2.31：有实际差异支撑的共享上下文证据

`.30` 第一轮正式主 Judge 的名单冲突响应正确表达了双侧损失，但短侧没有 A_ONLY，
旧编译器拒绝共享片段。保留其全部失败、原始响应、旧实现和冻结清单，不重启原 run。
H0653 另有“正文翻译完整”误写成“整对完整翻译”的问题，需明确 X 与整对输入的区别。

## 本轮改动与不变量

新 opt-in `dedup-judge-hs-v0.6.2.31` 使用内部 `dedup-retained-coverage-v5`；public v3 不变。
UNCOVERED 通常仍要求 own-unique source。仅实际矛盾允许 SHARED source：它必须与 root
hard_conflict 的本侧证据一致，counterpart 必须与 root 对侧证据一致且为对侧真实独有差异，
CONTRADICTS 与 uncovered_type 必须匹配。双方只引相同 SHARED 不足以证明不同值。
不伪造 unique 片段、不把 UNCOVERED 改为 COVERED、不修改原文或 raw response 来通过验证。
共享例外只导出冲突 no/no，不导出 containment；普通独有证据路径复用旧严格验证与语义映射。

system 替换旧“必须 own unique”的绝对指令，并加入共享上下文配对示例；pair 与 YAML
同步包含新限制。typed branch 的 5/8/4 字段规则、严格 raw schema 和 native transport 绑定保持。
COMPLETE_FAITHFUL 只适用于整对双向覆盖：翻译 X + 保留的额外 Y 必须 OTHER；不自动改写错误标记。
空锚点、普通 chrome、独立实质 Y containment、实际身份/角色/状态/闭合名单冲突保护不变。

新 main 必须实际使用 V5 prompt；critic 只消费同版本、同 payload 的 fresh main positive。
旧响应不得改版本号冒充线上新结果。离线反事实兼容探针可以在副本上显式改 contract tag，
但只诊断 adapter 表达能力，不计作模型能力提升或新版本线上输出。

## 测试、冻结与线上安排

现有环境内跑单元回归与四个真实 Ray→本地 HTTP→NDD→writer→binder 测试：主/critic × 首次/反馈。
本地边界包含两个真实双侧 UNCOVERED 名单方向与既有翻译、扩展、未决、负例、exact，
每次全套共 22 个本地 HTTP 请求，外部模型调用为零。通过后冻结全部新实现与测试摘要。

继续同模型、temperature 0、top_p 1、4096 输出、thinking off、并发16、timeout600、outer max_retries2；
原 24 条真实开发＋26 条合成镜像，标签、权重、两轮顺序不变，不能删掉难例。
预检由原8条增至11条：加入名单冲突正反方向及 H0653，以先验证本次修复。
预检主/最终主决策必须11/11正确且无 native/outer 纠错、每次请求一次 HTTP200；失败即停。
通过后运行相同50条的两轮 fresh main＋critic，真实与合成单独报告；基础请求上界222。
任一阶段不完整停止 schedule，不在观察到失败后修改冻结候选继续运行。
额外 native/outer/失败 HTTP 全量计数，≤1%纠错门槛不变；两轮均报告，不挑选最好的一轮。

沿用旧 Key 的显式 `--env-file /raid/hfang/workspaces/Curator/.env`，不打印/复制/保存密钥或其摘要。
本次使用独占 CPU Ray，结束只停自己的集群进程。旧标签、`.27`—`.30` 冻结保持；不发布 release，
不 commit/push，不扩大至20,000。这个预测可见局部面板不能证明完整1000条 weighted75/75/79门槛。

```bash
python -m eval.dedup.analysis.context_experiment prepare \
  --root /raid/hfang/ihb/runs/v0.6.2.31-context-diagnostic \
  --boundary eval/dedup/analysis/v06231_boundary.json
python -m eval.dedup.analysis.context_experiment run \
  --root /raid/hfang/ihb/runs/v0.6.2.31-context-diagnostic \
  --env-file /raid/hfang/workspaces/Curator/.env
```

运行时设置本次独占集群的 RAY_ADDRESS；已观察的 run 不可重启。
