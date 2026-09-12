# V0.6.2.15：命题对应的 critic 与保留内容一致性

## 范围

先复审 .14 critic 改错的 21 条及修好的 22 条，保持原标签，单独记录口径争议。
新版本只改 critic system/pair/rubric 和版本化仲裁，**完整保留 .14 主 Judge 的 system/pair/rubric**。
在线仍重新调用主 Judge 与 critic，不拼接历史模型响应、不复用旧 cache。
这能控制主 prompt，但不能保证服务端重复调用的主输出逐字一致；报告需要拆开这一点。
保留所有旧版本和输出，不 commit/push/release，不看 holdout，不跑 20,000。

## 两个合同

- `hs-v06215-arbitration`：.14 两个 Judge 的全部 prompt/rubric 不动，仅 v9 仲裁，使用 .14 保存响应零调用离线重放。
- `hs-v06215`：同 v9，critic 改为先重构两侧完整命题、查语义对应，再分类内容功能；没有新增模型输出字段。
- 保留 v3 public schema 和 visible payload。v3/v6/v7/v8 历史版本不得受影响。

## 有界修复

1. 独立举证的 retained identity/permission/state 等变化，不能因为 main 把文本划为 substantive 而被跳过。
   必须有本字段有效双侧与差异引用，且 critic 同意负向结论；有冲突或缺证据只能未决，不强行 no/no。
2. main 双向等价与 critic 有效确认的同记录实质扩展不能同时静默成立。对于 substantive 双侧且 scope 有效的情况，
   输出未决；现有字段没有独立扩展方向，不能从引用在哪一侧猜 direction，也不能把所有扩展直接判负。
3. 已有合法 no/no、未决、主 ledger 真实单向扩展、非正文专用边界和完整 exact 保护保持原有归属。
   不合并不同字段引用来伪造证据，不用正则关键词替代语义判断，不按 review ID 或标签路由。
4. Critic 先把共享片段与独有片段连成实际句义，再查另一侧对应句。额外 Accept 不抹去该侧“继续浏览即同意”；
   Stories/署名短指针不推测成独立文章；共享 FAQ 答案可覆盖问句标题，但不能覆盖真正新增操作。
5. 新准入条件、权限、受众变化不是普通正文追加，单侧新增也可以改变保留含义，不要求显式相反句。
   同时保留真实身份、目录、SKU、状态、列表成员与非空 containment 门槛。

## 评估冻结（在线候选输出查看前）

- 同一 258 对，标签/权重文件 `v06213_local_evaluation_labels.csv`，SHA-256
  `977eb95ba066655d03ff533ba98d5c5fe92fa3cb3f63148856159f39d09c1f2a`。
- 1,000 条原参照 SHA-256 `6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`。
  不修改标签、不排除争议条目、不以新 AI 复审结果替换原评分。不声称独立人工校准。
- 保持原准入：258/258 有效、0 终态、outer retry <=1%；12 exact 与 4 条真实 containment 方向全对；
  .12 保住的 20 负例和 3 benign 无回退；54 translation 加权主决策相对 .12 下滑不超过 3 pp。
  另完整报告全部 38 负例、5 benign、39 reference containment、非加权 translation。
- 额外保护 .14 critic 修好的全部 22 条，固定在 `v06215_critic_repaired_guards.json`，新候选必须 22/22。
  这只增加约束，不能替代或放宽既有底线。21 条回退中存在口径争议，不把“强行修回全部 21 条”设成优化目标。
- 顺序：CPU 单元测试与历史兼容 → .14 原响应 v9 离线 → 独立 20 对技术预检 → 正式 258 对全新调用。
  离线没有额外调用，不能据此让未运行的新 critic 通过。预检不注入正式 cache。
- 实际模型 `nvidia/qwen/qwen3.8-27b`，temperature=0、top_p=1、max_output_tokens=4096、max_visible_tokens=20000、
  window=4096/512、timeout=600、parallel=64、max_retries=2，与 .14 保持一致。HTTP 限流与 outer retry 分开报告。
- 在线前冻结源码、配置/资源、payload、协议与标签摘要；开始后不改推理代码或 prompt。
- local root `/raid/hfang/ihb/runs/v0.6.2.15-local-development`，Ray temp `/raid/hfang/ihb/r15dev`。
- 全部局部门槛通过才新建 full root 评估完整 1,000 条；失败就停止更大规模调用并审计，不把富集局部 P/R 当完整集指标。
  完整门槛仍为 weighted P/R >=75%、primary >=79%、over-group <=66、关键 cohort 相对 .9 不回退超过 3 pp，
  schema 100%、outer retry <=1%、全部 38/4/5 保护。即使通过也不自动授权 holdout 或 release。
