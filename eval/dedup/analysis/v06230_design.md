# V0.6.2.30：只改分支输出说明，不放宽语义或 schema

`.29` 的线上预检 8 条最终有效 7 条；两个案例原始响应把 `COVERED` 独有字段写进 `UNCOVERED`，
其中 cookie 空锚点案例三次 outer attempt 都失败。旧 Key 已验证可用，12 次 HTTP 全为 200。
保存 `.29` 全部失败记录，不重启原 run、不修改原 source/config/prompt/adapter。

## 唯一候选改动

新 opt-in 版本 `dedup-judge-hs-v0.6.2.30` 在主 Judge、critic 的 system 中加入同一份状态优先编码说明：
先确定 COVERED/UNCOVERED/UNRESOLVED，再输出其恰好的 5/8/4 个字段，不能混用分支；空值仍是多余字段。
pair 同步加入 YAML 定义的 response_encoding 提醒。原语义 policy、rubric、V4 schema、public v3、
adapter、证据与双向保留判据完全复用 `.29`。不删非法字段，不以 NDD pruning 后的对象替代原响应。
身份/角色保护、非空正文锚点、独立 Y containment、翻译、截断规则全部不变。

新 runtime 复用旧 builder/生命周期，仅替换角色版本并增加两个固定格式块；请求对照必须能去除这两个
格式块和版本差异后还原 `.29` 的完整消息。版本化结果只有在实际新 prompt 与原始响应严格绑定后才写入，
旧 `.29` trace/main 不能改名复用。格式指令也可能影响语义行为，因此通过工程检查不等于语义提升。

## 冻结与准入

沿用 `.29` 完整的模型/生成设置：相同 27B FP8、temperature 0、top_p 1、4096 输出、thinking off、并发16，
timeout600、outer max_retries2。输入正文、证据、标签、权重、两轮顺序均保持一致，全部 fresh main。
每条实际主/critic 请求保留 `input + 4096 + 2048 <= 32768` 检查；不得截断或丢弃超长案例。

先跑本地测试与 4 个真实 Ray→本地 HTTP→NDD→writer→binder 边界（主/critic × 首次/反馈），再冻结。
使用现有安装环境，不创建环境或依赖。线上显式 `--env-file /raid/hfang/workspaces/Curator/.env` 加载旧 Key，
不复制、打印密钥或在清单中保存密钥/密钥摘要；只记录文件路径。使用本次独占 CPU Ray，结束仅停自己的进程。

1. 同样 8 条合成预检，主/最终主决策 8/8 正确且零 native/outer 纠错；每个请求一次 HTTP200。
   任一阶段不完整或预检失败，停止，不进入正式面板；不得修完同一份冻结 prompt 后继续。
2. 通过才运行两轮相同 50 对：24 条真实开发（原22条＋H0521/H0822）和26条合成镜像，分开评分。
3. 对每轮相同主输出比较 main-only 与 critic 后的结果、逐条净增损、重试与跨轮稳定性。两轮都报告，
   不挑最好的一轮。不完整 schedule 不发布语义排名。`.29` 没有完整面板成绩，不能捏造 paired semantic baseline。
4. 基础请求上界216，额外 native/outer/失败 HTTP 全量计数。≤1%纠错门槛不变；不修改旧参考，
   不以少量预测可见案例或合成样本替代完整1000条 weighted75/75/79、分类保护及独立holdout。

本轮不运行20,000对，不发布release、不commit/push，不继续裁决其他口径争议。

## 入口

```bash
python -m eval.dedup.analysis.composite_format_experiment prepare \
  --root /raid/hfang/ihb/runs/v0.6.2.30-format-diagnostic \
  --boundary eval/dedup/analysis/v06230_boundary.json

python -m eval.dedup.analysis.composite_format_experiment run \
  --root /raid/hfang/ihb/runs/v0.6.2.30-format-diagnostic \
  --env-file /raid/hfang/workspaces/Curator/.env
```

运行前设置本次独占集群的 `RAY_ADDRESS`。已观察 run 不能重启；后续修正需要新候选与新冻结根目录。
