# 完整政策＋商品：口径确认后的影响清单

2026-09-11。用户确认的规则已另存为 [composite containment policy v2](composite_containment_policy_v2.md)。
只记录口径及其逐条应用；没有改 Judge、旧参考、旧输出、旧分数或缓存，没有新增模型调用。

## 已支持的方向

完整政策 P 自身可作为共享 X。商品侧完整保留 P 并增加实质商品正文 Q 时，可以单向替代政策-only 侧。
不要求政策页先描述这个商品，也不因新商品的独立身份或板块而自动 veto。
商品新增的条件如果推翻 P，或者双方各有不同商品正文，则不能仅凭共享政策合并。

| 样本 | 本轮应用建议 | 已核对的实质内容 | 原参考 |
|---|---|---|---|
| H0108 | B 替代 A；反向不行 | Toyota 完整保修条款＋52164-52090 零件 | NO/NO |
| H0126 | A 替代 B；反向不行 | 完整购买/配送/退货政策＋Graf Bolero 冰鞋 | NO/NO |
| H0419 | A 替代 B；反向不行 | 完整配送/退款/支付政策＋背心 | NO/NO |
| H0590 | A 替代 B；反向不行 | 完整评价核验与责任说明＋Benromach 商品 | NO/NO |
| H0869 | A 替代 B；反向不行 | 完整数据/cookie 政策＋Tescoma 商品 | NO/NO |
| H0998 | B 替代 A；反向不行 | 相同 Toyota 政策＋52309-0C010 零件 | 已是 B 替代 A |

前 5 条来自原 121 条新增 over-group，权重合计 **50.225**。H0998 是原面板保护样本，不属于该 121 条。
H0108/H0998 的政策-only 侧原文完全相同；不能再一边说它是正文、一边仅凭“保修模板”否决。

以上是**用户批准规则后的预测可见 AI 实例应用，不是逐条独立人工 gold**。
5 条不再作为必须恢复旧 NO 的目标；6 条均另存为正向开发保护候选。
原参考仍保留，不能把这 50.225 的改标影响称为模型带来的 precision 改善。
以后若批准新版 reference，须让全部可比历史版本在同一参照上重算。

## 仍待核的范围

- H0008/H0756/H0876/H0964 新增的是举报流程、文章、活动或评论，不是此次明确确认的商品正文；不把它们批量判成已获批准。
- H0266/H0352/H0738 是相同的信贷代表例及 BNPL 短条款；H0812 是短保修说明。它们是否构成完整政策仍需统一，不能仅凭短、无标题或页脚位置继续强制 NO。
- 所有站点/服务简介、裸标题/编号等其他边界不在此次确认范围内。
- 普通 cookie 同意 UI 不自动成为完整政策；但完整 cookie 政策也不能仅因包含 cookie 一词被降为 UI。

因此，原 5 条负例保护中 **H0352 暂挂起强制 NO**，并未改判 YES。
其余 H0093/H0537/H0701/H0760 保留为开发负例候选；原 5 条冻结文件不修改。
原 121 条的 79/40/2 分类属于先前口径下的审计快照，不回写覆盖；本次差异保存在独立影响清单中。

## 实验面板和材料

48 条与 64 条的原冻结都保持不变。另存一份覆盖原 **全部 64 个 ID** 的政策附录，不删除样本、不改 payload、不填入独立 gold：

- H0419、H0998 附上新口径下支持 PRESENT 的开发建议；原角色/历史争议标记仍保留。
- H0266、H0352 附上短政策完整性待核状态。
- 其他成员仍按原规则等待独立标注；不能把这次确认写成整套面板 gold 已完成。

旧的已渲染请求依然属于旧冻结，**不能直接按新口径开跑**。
下一次调用前要将新口径同步进新的请求/协议版本，并冻结独立标注及执行预检；本次未启动这一步。
6 条真实正向候选另存回归池，无须为此次确认再次改变 64 条抽样名单。

新增 8 类合成材料及 A/B 镜像共 **16 对**，覆盖政策＋商品、普通 chrome、双方不同商品、政策例外、义务主体、cookie-only、等价非正文和截断。
离线测试只验证方向/契约/证据及材料完整性，不是模型已通过语义测试。

## 输出与验证

- [逐条应用与待核清单](/raid/hfang/ihb/runs/policy-only-application-v1/applications_private.json)：完整相同政策双侧范围、精确证据与字节位置、商品新增证据和旧参照。
- [当前开发保护候选](/raid/hfang/ihb/runs/policy-only-application-v1/development_guards.json)：4 条负向候选、1 条暂挂起、6 条正向候选；均非新增人工 gold。
- [64 条面板政策附录](/raid/hfang/ihb/runs/policy-only-application-v1/panel64_policy_overlay_private.json)。
- [匿名原文](/raid/hfang/ihb/runs/policy-only-application-v1/application_inputs/inputs_blind.jsonl)与[不可变摘要](/raid/hfang/ihb/runs/policy-only-application-v1/summary.json)。

定向测试 **61 项通过**，包括原复核与复合包含材料的兼容测试；Ruff 检查/格式检查和 `git diff --check` 通过。
新材料重复导出一致，并复验原 121 条复核及 48/64 条冻结摘要未变。本轮在线模型调用为 0。

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup.analysis.policy_only_review \
  --output /raid/hfang/ihb/runs/policy-only-application-v1
```
