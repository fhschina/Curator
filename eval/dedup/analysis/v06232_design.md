# V0.6.2.32：保持 V5 判据，修复方向证据提示与反馈

`.31` 预检10/11有效。反向闭合名单三次都理解冲突，但 A-in-B 的 counterpart 错选 A001，
root 又填 S001/S001。旧错误只说缺少实际差异，没有给出方向槽位与可用ID的对应关系。
本轮冻结 `.31` 的全部失败与实现，不修改已观察版本；新 opt-in 版本为 `dedup-judge-hs-v0.6.2.32`。

## 唯一诊断因素

主/critic都增加同一组方向绑定提示：A-in-B source在A、counterpart在B；B-in-A镜像；root字段按文档侧。
system包含双向镜像绑定示意，pair/Jinja直接从原payload枚举A_ONLY、B_ONLY、SHARED IDs，YAML同步规则。
这个清单只标记证据所属文档，不读标签/旧预测、不选语义正确答案、不改变原payload。
实际矛盾先定位独有差异及对侧上下文，再把同一对证据填进已有root/方向字段；普通独立Y不据此升级冲突。

失败反馈保留整个无效响应，列明错误字段、实际ID、其文档侧允许的ID和冲突证据约束。
复用现有安全反馈字段与outer重试编码，反馈不丢失；不得把错误对侧ID建议给root。
只在严格原始输入/prompt/schema/transport绑定已经通过后增强指定证据错误。原本合法的V5输出直接复用
原适配器、相同公开判决；其他错误保留。不是自动修复，不删字段，不重排/改写证据，不更改通过条件。

内部V5 schema、public v3、非空正文锚点、独立Y containment、实际冲突、完整翻译等语义规则均不变。
新增的是版本化prompt和 `dedup-directional-witness-feedback-v1`，并非新语义taxonomy。
继承binder先核实实际 `.32` prompt及原始响应，再附新版本元数据；旧 `.31` trace不能改名复用。
critic只消费同版本完整fresh main，重试反馈不跨stage继承。

## 验证与冻结安排

1. 在现有环境运行源码一对一回归，检查镜像、合法V5输出不变、非法证据不被自动接收、原标签不可泄漏，
   错误详情经过真实重试编码仍可用。使用 `.31` 保存响应离线检查，仅比较错误反馈和接受集，不称新模型结果。
2. 四个native边界：主/critic × 首次/真实outer纠错。纠错测试先返回上次同型错误，仅在收到字段级反馈后
   返回合法镜像；每次全套24个本地HTTP请求，0外部模型调用。通过后冻结实现/配置/测试/设计。
3. 同样11条预检，必须main/final全正确、零native/outer纠错、每次请求一次HTTP200；不完整或失败即停。
4. 通过后跑两轮完全相同的50对（24真实开发＋26合成），沿用原标签、权重和顺序，全部fresh main。
   每轮对比同份main-only与critic后结果，真实与合成分开，两轮都保留，不挑最好一轮。
5. 同一模型和生成设置：temperature0、top_p1、4096输出、thinking off、并发16、timeout600、outer重试2。
   原文及实际main/critic/重试prompt均须满足input+4096+2048≤32768，不能删案例或截断。
   基础请求上界222，所有额外native/outer/失败HTTP单独计数，≤1%纠错门槛不变。

任一阶段终态失败停止schedule；不修改冻结候选后继续。沿用旧Key的显式env-file，
不打印/复制/保存密钥及其摘要；本次使用独占CPU Ray，完成仅停自己的进程。
不更改参考、历史冻结与旧cache，不commit/push、不release、不扩大1000/20,000，不声称小面板达到75/75/79。

```bash
python -m eval.dedup.analysis.directional_experiment prepare \
  --root /raid/hfang/ihb/runs/v0.6.2.32-directional-diagnostic \
  --boundary eval/dedup/analysis/v06232_boundary.json
python -m eval.dedup.analysis.directional_experiment run \
  --root /raid/hfang/ihb/runs/v0.6.2.32-directional-diagnostic \
  --env-file /raid/hfang/workspaces/Curator/.env
```

运行前设置本次独占集群RAY_ADDRESS；已观察run不能重启。
