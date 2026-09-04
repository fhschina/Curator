# Qwen3.8-27B-FP8：本地单 B200 与 NVIDIA Inference Hub 推理基准

> 面向组会的实验报告 / 建议讲解时间：10–15 分钟
> 实验状态：**PASS**
> 最终代码：`experiment/inference-hosting` @ `5365eabe`
> 实验日期：2026-08-27 至 2026-08-31

## 1. 结论摘要

这次实验比较的不是两个简单的模型 API，而是同一套 Sarah NDD dedup judge workload 在两种部署方式下的端到端性能：

- Local：一张 NVIDIA B200 上的 Qwen3.8-27B-FP8，经 Ray、Dynamo 和 vLLM 提供服务。
- Hub：NVIDIA Inference Hub 上公布的 `nvidia/qwen/qwen3.8-27b` 服务。
- 两边接收相同的 canonical chat request，使用相同 prompt、rubric、structured-output contract、输入 pair、并发调度、validation 和 retry policy。
- 主指标是 **schema-valid unique pairs/s**，即每秒完成多少个最终通过结构校验的文档 pair，而不是只统计原始 API request/s。

正式实验中，每个 endpoint 完成 6,000 个 unique pairs，全部有效，0 个 terminal failure、429、5xx、timeout 或 context overflow。

在预先指定的主配置 `C=8` 下：

- Local：**0.1143 valid pairs/s**
- Hub：**1.4783 valid pairs/s**
- 三个 paired blocks 的 Hub/Local 加速比中位数：**14.40×**，范围 **10.45×–14.89×**
- 独立重跑全部三个 `C=8` blocks 后，中位加速比为 **10.65×**，范围 **9.22×–11.44×**

因此，最稳妥的结论是：

> 在本实验的 Sarah judge workload、当前单 B200 本地 serving 配置和当时的 Hub 服务状态下，Hub 的 schema-valid goodput 稳定地显著高于 Local；`C=8` 的独立验证仍观察到约 10.65× 的中位加速。具体倍数会随 Hub 服务时段和运行状态波动。

这是一项 **same-canonical-request、advertised-model-matched 的黑盒托管服务比较**，不能解读为 Hub 底层硬件与 B200 的直接硬件 benchmark。

## 2. 为什么做这个实验

Sarah 的 dedup pipeline 使用 LLM 判断两个文档是否属于同一个 duplicate group，以及一个文档能否替代另一个。每个 pair 会产生一组结构化标签，例如：

- `same_duplicate_group`
- `a_can_replace_b` / `b_can_replace_a`
- `relation_type`
- `material_difference`
- `fuzzy_scope`

我们已经可以在本地 B200 上运行 Qwen3.8-27B-FP8，也可以使用 NVIDIA Inference Hub 上的同名 Qwen3.8 服务。实际问题是：

> 如果保持输入、prompt、输出 schema 和并发方式不变，哪种部署能更快地产生可用的 judge 结果？

这里强调“可用结果”，因为只比较 HTTP request/s 会漏掉 parser correction、schema validation、retry 和 terminal failure 的成本。

## 3. 实现方式

### 3.1 总体数据流

```text
固定、分层抽样的 document pairs
              │
              ▼
      Sarah NDD / Data Designer
   相同 prompt、rubric、output schema
              │
              ▼
        logical model request
       Qwen/Qwen3.8-27B-FP8
              │
              ▼
      本机 loopback request relay
   先计算 canonical hash 和 token 数
              │
       ┌──────┴────────┐
       ▼               ▼
 Local target       Hub target
 Dynamo/vLLM        仅改写 model 名称
 1 × B200           并加入 API auth
       │               │
       └──────┬────────┘
              ▼
 相同 parser、validation、failed-subset retry
              │
              ▼
 terminal valid/error record + timing/accounting
```

### 3.2 为了让两边可比，我新增了什么

1. **Endpoint-neutral runtime**

   把 Sarah 原有 judge pipeline 与“模型部署在哪里”解耦。同一个 Ray/Data Designer pipeline 可以连接本地 Dynamo/vLLM，也可以连接 OpenAI-compatible Hub endpoint。

2. **统一 loopback relay**

   Data Designer 始终发送同一个 logical model ID 和 request body。Relay 在任何 endpoint-specific 改写之前计算 canonical request hash：

   - Local：转发给本地 Dynamo/vLLM。
   - Hub：只改写 Hub model 名称并注入认证信息。
   - Relay 记录 hash、client tokenizer token count、时间、HTTP status 和 outstanding concurrency。
   - API key 不打印、不写入产物；relay 不记录完整文档或完整请求正文。

3. **固定 workload 与 run identity**

   输入 block、source candidate、prompt、rubric、tokenizer、配置和代码 revision 都写入 digest。任何关键内容变化都必须产生新的 run ID，避免“同一个目录里悄悄换了实验条件”。

4. **端到端计时与严格 accounting**

   Block 从第一条 measured request submit 开始，到最后一个 pair 通过 validation 或成为 terminal error 为止。Retry 时间包含在主指标中。部分完成的 block 不与后续结果拼接。

5. **可恢复但不可篡改的运行方式**

   每个 endpoint/block/attempt 有独立目录。只有具有完整 marker 和 artifact hash 的 block 才能复用；中断 block 必须新建 attempt 并完整重跑。Recovery manifest 记录父 run、新 run、代码 revision、request contract 和每个导入 artifact 的 hash。

## 4. 实验配置

| 项目 | 配置 |
|---|---|
| Host | `umb-b200-218` |
| Local GPU | 1 × NVIDIA B200，约 183 GB 显存可见 |
| Local model | `Qwen/Qwen3.8-27B-FP8` |
| Local revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Local engine | Dynamo + vLLM，TP=1 |
| Local serving | `max_model_len=32768`，`max_num_seqs=8`，GPU memory utilization 0.8，`enforce_eager=true` |
| Hub endpoint | `https://inference-api.nvidia.com/v1` |
| Hub model | `nvidia/qwen/qwen3.8-27b` |
| Generation | temperature=0，top_p=1，thinking disabled |
| Output budget | 4,096 tokens |
| Timeout | 600 s |
| Prompt construction | pinned Qwen tokenizer；最多 20,000 visible tokens；window=4,096，overlap=512 |
| Context gate | `prompt_tokens + 4096 <= 32768` |
| Concurrency | closed-loop `C ∈ {1, 2, 4, 8}`，不设置客户端 RPM cap |
| Retry | 最多两次 failed-subset retry；实际正式运行没有发生 pair retry |

Local checkpoint 经固定 revision 校验，实际为 FP8、66 个 weight shards、约 30.9 GB weight files。Local cold start 为 **293.8 s**，单独记录，不计入 warm-endpoint 主指标。

Hub 不公开 remote checkpoint revision、tokenizer revision 或 serving hardware，因此只能确认服务名称与请求契约匹配，无法证明两边使用 bit-identical checkpoint 或相同 serving implementation。

## 5. Workload 和实验协议

### 5.1 Pair 采样

- Source：固定的 V0 candidate-pair population。
- Seed：`26082701`。
- 正式 workload：12 个互不重叠的 blocks，每个 block 500 pairs。
- 每个 endpoint：总计 **6,000 unique pairs**。
- 每个 concurrency：3 个 blocks，共 1,500 pairs/endpoint。
- 每个 block：
  - 250 removal-track pairs
  - 250 cross-group-track pairs
  - 每个 track 再按最终 Qwen prompt length 的五个 quintiles 等量采样
- Warm-up：额外 100 pairs，与正式 workload 不重叠，不计入结果。

### 5.2 调度和顺序控制

- 使用相同 closed-loop scheduler：最多保持 `C` 个 outstanding requests，一个完成后才补下一个。
- 12 个 blocks 采用固定交错顺序。
- 全局有 6 个 Local-first blocks 和 6 个 Hub-first blocks，降低“某一端总在更空闲时段运行”的偏差。
- 本地模型只启动一次并保持加载；Hub blocks 运行时本地服务不重启。

### 5.3 正式运行前的 gates

正式测量前完成了 checkpoint 下载与验证、测试修复、Hub model/quota/credit probe、40-pair pilot 以及 100-pair warm-up。Dynamic preflight 的关键结果为：

- Canonical request hash equality：**true**
- Pinned-tokenizer prompt token equality：**true**
- Hub 100-request quota probe：**0 个 429**
- Hub 接受 thinking-disabled 参数：**true**
- 所有 prompt 均通过 context gate

## 6. 指标如何定义

### 主指标

```text
schema-valid goodput
= terminal schema-valid unique pairs / measured block wall seconds
```

这个指标会惩罚慢响应、schema-invalid output、parser correction、retry 和最终失败，更接近实际 dedup pipeline 能消费的产出速度。

### 主 headline

预先指定 `C=8` 为 headline。对三个相同 pair block 分别计算：

```text
Hub block goodput / Local block goodput
```

然后报告三个 paired ratios 的 median 与 min–max。这样可以在相同 block 内配对，减少不同样本长度构成造成的影响。

### 次要指标

- request latency p50/p95/p99
- first-attempt valid rate
- retry、terminal error、timeout、429、5xx
- prompt/completion token accounting
- Local/Hub judge label agreement
- Local GPU utilization、memory、power 和 cold-start time

## 7. 正式实验结果

### 7.1 各并发级别

| Concurrency | Local valid pairs/s | Hub valid pairs/s | Hub/Local aggregate goodput | Local latency p50/p95 | Hub latency p50/p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0171 | 0.2597 | 15.21× | 48.07 / 52.21 s | 3.30 / 5.84 s |
| 2 | 0.0324 | 0.3808 | 11.77× | 50.28 / 54.93 s | 4.65 / 7.84 s |
| 4 | 0.0603 | 0.7554 | 12.54× | 53.27 / 58.54 s | 4.36 / 8.93 s |
| 8 | **0.1143** | **1.4783** | **12.94×** | **56.45 / 62.71 s** | **4.63 / 7.40 s** |

表中的 speedup 是“两个 endpoint 的 aggregate goodput 相除”。预先指定的正式 headline 则是“三个 block 内 paired ratio 的中位数”，因此 `C=8` headline 是 **14.40×**，不等于表中的 12.94×。两者回答的问题略有不同，不能混用。

### 7.2 C=8 主结果

> At concurrency 8, Hub/Local achieved **1.4783/0.1143 schema-valid pairs/s**, a **14.3999× median paired throughput ratio** across three blocks, with a range of **10.4508×–14.8947×**. Local p50/p95 latency was **56.45/62.71 s**; Hub p50/p95 was **4.63/7.40 s**.

### 7.3 正确性和运行健康度

| Gate | Local | Hub |
|---|---:|---:|
| Measured unique pairs | 6,000 | 6,000 |
| Schema-valid pairs | 6,000 | 6,000 |
| Valid rate | 100% | 100% |
| First-attempt valid rate | 100% | 100% |
| Pair retries | 0 | 0 |
| Terminal errors | 0 | 0 |
| HTTP 429 | 0 | 0 |
| HTTP 5xx | 0 | 0 |
| Timeout | 0 | 0 |
| Context overflow | 0 | 0 |

Local 与 Hub 在全部 6 个核心 judge fields 上完全一致的 pair 比例为 **85.25%**。单字段一致率为 **93.67%–97.43%**。这说明两边输出在大多数单项判断上高度一致，但并非完全相同。

即使 temperature=0，跨两个 serving stack 也不保证 bitwise deterministic；此外 Hub 的具体 checkpoint/tokenizer revision 不公开，所以这里不应把 85.25% 解读为严格相同模型的可重复性测试。

### 7.4 Local GPU 观测

只统计 Local measured windows：

- GPU utilization：mean **35.24%**，p95 **37%**，max 100%
- GPU memory：mean 约 **147.5 GiB**，max 约 180.4 GiB
- Power：mean **293.6 W**，p95 **471.6 W**
- SM clock：mean 约 1,960 MHz

这些数据表明，结果代表的是 **当前 Sarah + Ray + Dynamo/vLLM 配置的端到端性能**，而不是 B200 的理论吞吐上限。本地路径很可能仍有 scheduler、structured generation、CPU/Ray pipeline 或 vLLM 参数方面的优化空间。

## 8. 独立 C=8 验证

正式结果完成后，又创建了一个独立 validation run：

- 非 headline 的 9 个 blocks 仅作为已验证 artifacts 导入。
- 三个 `C=8` blocks 在最终代码 revision 下全部从头重跑。
- 因此，独立验证新增测量的是 **1,500 个 C=8 pairs/endpoint**，不是重新执行全部 6,000 pairs/endpoint。

| Run | Local pairs/s | Hub pairs/s | Paired median speedup | Paired range | Local p50/p95 | Hub p50/p95 |
|---|---:|---:|---:|---:|---:|---:|
| 正式结果 | 0.1143 | 1.4783 | 14.40× | 10.45×–14.89× | 56.45 / 62.71 s | 4.63 / 7.40 s |
| 独立 C=8 验证 | 0.1196 | 1.2386 | 10.65× | 9.22×–11.44× | 54.11 / 60.71 s | 5.72 / 9.03 s |

验证结果显示：

- Local goodput 从 0.1143 变为 0.1196 pairs/s，变化较小。
- Hub goodput 从 1.4783 变为 1.2386 pairs/s，约下降 16%。
- 加速比从 14.40× 降到 10.65×，说明 Hub 性能存在时段或服务负载波动。
- 但三个独立 validation blocks 中最低仍为 9.22×，所以“Hub 显著更快”的方向性结论得到确认。
- Validation 的全部核心字段一致率为 85.02%，与正式结果的 85.25% 接近。

推荐对外表述时同时给出两次结果，不只挑选较高的 14.40×。

## 9. 为什么只有 6,000 pairs 却跑了很久

“6,000 pairs”是每个 endpoint 的总 unique workload，但实验不是只跑一次 `C=8`：

- 4 个 concurrency levels
- 每档 3 个重复 blocks
- 每个 block 500 pairs
- Local 和 Hub 各执行一次
- 为避免相互干扰，blocks 串行交错运行

正式结果中的累计 measured endpoint wall time为：

- Local：**47.84 小时**
- Hub：**3.53 小时**
- 合计：**51.37 endpoint-hours**，还不包括 checkpoint、preflight、warm-up、服务启动、修复和中断期间的自然时间

其中 Local `C=1` 的 1,500 pairs 单独就用了约 **24.4 小时**。低并发档是刻意保留的，因为它可以展示 scaling curve，但也显著拉长了总实验时间。

这也解释了为什么之前 Hub-only 的 20K-pair V0 可能一个晚上就能完成：Hub-only/high-concurrency workload 与“两个 endpoint × 四档 concurrency × 三次重复”的完整 benchmark 不是同一个工作量；Sarah structured judge 的长输出和 validation contract 也比只统计简单 request completion 更重。

## 10. 中途遇到的主要难点和修复

### 10.1 Preflight 和 restart 必须幂等

早期版本在重跑 preflight 或 prepare 时会因为时间戳、可用空间或已有 checkpoint marker 变化而认为实验身份改变。修复后：

- 稳定字段必须一致。
- 时间戳等 volatile fields 不参与身份变化判断。
- 已验证 checkpoint provision marker 可以安全复用，但 revision 或文件不一致会立即失败。

### 10.2 断点恢复不能污染计时

如果进程在一个 block 中间退出，把前半段和重启后的后半段拼在一起会产生错误 wall time。最终方案是：

- 不完整 attempt 永不进入汇总。
- 重启时新建 attempt directory，并完整重跑该 endpoint block。
- 已完成 block 通过 completion marker 和 artifact SHA256 保持不可变。

### 10.3 Provider token accounting 不完全一致

两端 canonical request 和 pinned client tokenizer count 完全一致，但 Hub 与 Local 的 provider-reported prompt usage 在 **99/6,000 个 paired initial requests** 上不同。

这说明远端可能使用不同 tokenizer revision、chat-template accounting 或服务端内部处理。最终处理方式是：

- 公平比较统一使用 pinned client tokenizer 作为 prompt-token accounting basis。
- Provider usage 作为黑盒 telemetry 保留并披露。
- 不使用 provider-reported prompt tokens 宣称严格可比的 token throughput。

### 10.4 Transport attempt、parser correction 和 pair identity 容易混淆

一次 pair evaluation 可能对应多个底层 HTTP events。失败的 transport attempt 必须进入 wall time 和错误统计，但不能制造额外 pair，也不能破坏 Local/Hub pairing。修复后的 accounting：

- 配对使用每个 pair 的 successful initial request。
- 失败 transport attempts 仍保留在 latency、HTTP status 和 raw attempt telemetry 中。
- Parser correction 或 failed-subset retry 与初始 workload 分开计数。

### 10.5 Recovery 本身也需要可审计

正式 recovery run 使用新 run ID：导入 3 个已完成且不可变的 paired blocks，重跑其余 9 个。Recovery manifest 记录：

- 父 run 与新 run
- 旧/新 Git revision
- config、workload 和 request-contract digests
- 每个导入 marker、event 和 terminal artifact 的 SHA256
- 哪些 blocks 被导入，哪些被重跑

随后又从头重跑全部三个 `C=8` blocks，专门消除“headline 是否受恢复过程影响”的疑问。

## 11. 结果应该如何解读

### 可以得出的结论

1. 在同 canonical request、同 Sarah contract 和相同 closed-loop concurrency 下，Hub 的 schema-valid goodput 明显高于当前 Local serving stack。
2. Hub 的 request latency 也明显更低；在 `C=8` 正式结果中，p50 为 4.63 s，而 Local 为 56.45 s。
3. 该结论经过全部三个 `C=8` blocks 的独立重跑，方向保持不变。
4. 两端都通过了可靠性 gates，所以速度差异不是用更多 invalid output、429 或 terminal failure 换来的。

### 不能直接得出的结论

1. 不能说“Hub 的某一张 GPU 比 B200 快 10–14×”；Hub 硬件数量、型号、并行策略和负载不可见。
2. 不能说两端是 bit-identical model；Hub 不披露 checkpoint/tokenizer revision。
3. 不能把 14.40× 当作永远稳定的常数；独立验证为 10.65×，Hub 有明显时段波动。
4. 不能据此直接作成本结论；本实验没有将 Hub credit 消耗、GPU 占用成本或工程维护成本统一折算。
5. 不能说当前 Local 结果代表 B200 的最佳性能；本地 measured-window GPU utilization 仍显示优化空间。

## 12. 建议的后续工作

1. **以独立验证作为保守 headline**

   对组内可以说“正式运行 14.40×，独立验证 10.65×；保守地说 Hub 约有一个数量级的 goodput 优势”。

2. **单独优化 Local serving stack**

   Profile Ray/Data Designer、relay、structured generation 和 vLLM 各阶段；重点测试 `enforce_eager=false`、更高 `max_num_seqs`、continuous batching 和 scheduler/CPU bottleneck。优化后必须创建新 run ID 再做同协议比较。

3. **增加时间维度重复实验**

   在不同时间段重复 `C=8` blocks，报告 Hub goodput 的日内/日间分布，而不是单个点估计。

4. **补充 cost-normalized metric**

   在吞吐之外比较 cost per 1,000 schema-valid pairs，才能回答生产选择问题。

5. **分析 15% 全字段不完全一致的 pairs**

   按 prompt length、track 和 label 类型分层，确认差异来自 checkpoint/tokenizer/serving nondeterminism，还是某些样本对 prompt 细节更敏感。

## 13. 建议的现场讲法

### 开场（约 1 分钟）

“我想回答的不是哪个 API 返回得更快，而是在完全相同的 Sarah dedup judge workload 下，本地一张 B200 与 Inference Hub 谁能更快地产生真正可用、schema-valid 的 pair 判断。为了防止输入、prompt、并发和 retry 不一致，我实现了一个 endpoint-neutral benchmark 和统一 relay。”

### 方法（约 3–4 分钟）

依次讲：

1. 上面的数据流图。
2. 12 × 500-pair blocks、四档并发、三次重复。
3. Canonical request hash、固定 tokenizer、6 Local-first/6 Hub-first。
4. Goodput 包含 validation 和 retry 时间。
5. 不完整 block 丢弃，完整 block 由 hash 锁定。

### 结果（约 3 分钟）

先给 `C=1/2/4/8` 表，再突出 `C=8`：正式 14.40×、独立验证 10.65×。同时说明 6,000/6,000 valid、0 errors，证明速度不是以正确性失败为代价。

### 难点和边界（约 3 分钟）

重点讲三件事：

1. Restart-safe block timing。
2. Provider token telemetry drift 与 pinned-tokenizer accounting。
3. Hub 是黑盒，因此结论是服务栈比较，不是硬件比较。

### 收尾（约 1 分钟）

“结论是当前 Hub 在这个 workload 下稳定快一个数量级，但倍数随 Hub 状态波动；同时本地 GPU 利用率说明 Local 还有优化空间。下一步应做 Local tuning、跨时段重复和 cost-normalized comparison。”

## 14. 可能会被问到的问题

**Q：断点重跑会不会让时间结果无效？**
A：不会把断点前后的部分拼接。不完整 attempt 不进入汇总，整个 endpoint block 在新 attempt 中重跑；已完成 block 有 marker 和 SHA256。并且 headline 的三个 `C=8` blocks 后来又全部从头独立重跑。

**Q：为什么正式结果是 14.40×，表里 C=8 是 12.94×？**
A：14.40× 是三个相同 workload block 内 Hub/Local ratio 的中位数，是预先指定的 headline；12.94× 是两端 aggregate goodput 相除。计算顺序不同。

**Q：为什么 temperature=0 仍然只有 85% 的全部字段完全一致？**
A：temperature=0 不能保证跨不同 serving stack 的 bitwise determinism；Hub 的 checkpoint/tokenizer revision 也不公开。单字段一致率仍为 93.7%–97.4%，但这个差异需要作为限制披露。

**Q：为什么 6,000 pairs 跑了两天多？**
A：每个 endpoint 都跑 4 个并发档 × 3 个 blocks，且 blocks 串行；Local measured time 本身就是 47.84 小时，其中 C=1 用了约 24.4 小时。

**Q：能否直接用这个结果决定上 Hub？**
A：吞吐和延迟证据支持 Hub，但生产决策还需要成本、数据治理、SLA、可用性和 Local 优化后的对照。

**Q：API key 或文档内容有没有写进 Git？**
A：没有。Credential 只在运行时注入；raw payload 和 outputs 位于权限受控的私有 run directory，Git 中只保留实现和聚合结果。

## 15. 可复现性信息

- Branch：`experiment/inference-hosting`
- Final benchmark code：`5365eabec768f7191704bf00746813a3ebcc1a25`
- Local model revision：`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Formal run：`qwen38-hosting-20260829T203512Z-2d432ca900a4-recovery`
- Independent C=8 validation：`qwen38-hosting-20260831T154448Z-2d432ca900a4-c8-validation`
- Formal result：`/raid/hfang/ihb/runs/qwen38-hosting-20260829T203512Z-2d432ca900a4-recovery/RESULTS.md`
- Validation result：`/raid/hfang/ihb/runs/qwen38-hosting-20260831T154448Z-2d432ca900a4-c8-validation/RESULTS.md`
