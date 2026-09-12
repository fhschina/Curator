# .33-exp1 原样全链路复现协议

用户授权一次完整原始 1,000 条复现，稳定后才进入独立验证。本次保持
.33-exp1 实验版本名；不注册新 prompt，不混入 exp2/exp3 修复，不改旧默认值。

## 冻结内容

- 主 Judge：原 .12 system/pair/YAML、原生 Data Designer 输出解析器、v6-route 适配。
- 主调用不加入 response_format；保留原 dataframe 将缺失 repair_feedback 渲染为
  nan 的行为。每条初始消息和模型参数必须重建出历史 HTTP 请求摘要。
- 主 Judge 沿用最多两次原生格式纠正、最多两次外层校验重试；不增加新语义反馈。
- 后续是原 coverage v4、subject v1 的 payload 专属 span 枚举、scope v2 介入范围和
  fixed-subject proof verifier v4。沿用 without_unique_items 输出投影，不换成
  exp3 的 portable_structure 或通用 subject schema。与已有全部阶段请求逐条对照。
- 原 payload 1,000 条全部重跑主 Judge；不新增 exact、截断或 capped packet 的前置绕过。
  后续调用只由本次新上游结果触发，绝不借用旧阶段回答。保留合法未决。
- 模型 nvidia/qwen/qwen3.8-27b、temperature=0、top_p=1、max_tokens=4096、thinking=false。
  复用已配置的旧 NVIDIA key，不保存凭据。收集器双并发、2 秒限速、最多两次 HTTP
  尝试，旧主 Judge 原为 64 并发；调度变化单独披露，不称为字节级历史时序重现。
- 全新 run root、请求/响应/结果与逐次追加进度文件。不存在旧 cache 迁移。
- 参照与权重来自已冻结的 revised reference，沿用 H0668、H0413、H0758、H0702、H0928
  五条比较排除；995 条计分，1,000 条工程统计，不扩大排除到相似样本。

## 预先声明的判读

报告加权及非加权 P/R/F1、主决策一致、全部逐案得失、阶段变化、同文一致性、
工程失败和重试。失败占位符不能获得未决匹配分，参考为正的未决仍计漏判。
不重评旧 taxonomy；当前混合开发参照不冒充独立人工 gold。

保持 P/R >=75%、主决策一致 >=79%、误合并 <=66、schema completion=100%、
样本级语义重试率 <=1%；identity、meaningful-addition、translation 各类相对旧
exp1 不下降超过 3 个百分点。同时使用预先声明的复现诊断容差：加权 P、R、
主决策一致各自比旧 exp1 下降不超过 3 个百分点。该容差是解释单次复现的点估计
标准，不是统计非劣检验、不修改既定 release 门槛、不保证模型跨遍稳定。

所有采集结束前不看中途语义分数、不改 prompt/参照或择优重跑；只监测运行质量。
原始响应绑定和逐阶段离线重放通过后生成分数。

## 独立验证

上述复现检查通过后才准备真正未用于旧评估的 400 条：200 条分层代表样本与
200 条难例，检查文档重叠而非仅 pair ID；隐藏预测与抽样原因。代表集 50 条、
难例 50 条由两位独立评审标注，分歧第三人仲裁，其余单标。助手复审不能冒充
第二名独立人类评审。评审人员或未污染数据若不足，明确报告缺口，不伪造完成。
在独立标签冻结前不查看该候选的 holdout 结果；沿用之前的单次查看和失败转开发集规则。

本次不自动启动 20k、删除数据、发布 release、修改参考、commit 或 push。
