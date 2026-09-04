# Qwen3.8-27B-FP8 Inference Benchmark: Local NVIDIA B200 vs. NVIDIA Inference Hub

> Technical report
> Experiment status: **PASS**
> Final implementation: `experiment/inference-hosting` at `5365eabe`
> Experiment period: August 27–31, 2026

## Executive Summary

This study evaluates two deployment options for the same Sarah NDD document-deduplication judge workload:

1. A local `Qwen/Qwen3.8-27B-FP8` deployment on one NVIDIA B200, served through Ray, Dynamo, and vLLM.
2. The advertised `nvidia/qwen/qwen3.8-27b` service on NVIDIA Inference Hub.

The benchmark was designed as an endpoint-neutral, paired comparison. Both endpoints received the same canonical chat requests and used the same Sarah prompts, rubric, structured-output contract, source pairs, validation logic, retry policy, and closed-loop concurrency schedule. The primary metric was **schema-valid unique pairs per second**, rather than raw HTTP request throughput.

The formal experiment evaluated 6,000 unique document pairs per endpoint across concurrency levels 1, 2, 4, and 8. Both endpoints produced 6,000 schema-valid terminal results, with no terminal failures, HTTP 429 responses, HTTP 5xx responses, timeouts, or context overflows.

At the pre-specified headline concurrency of 8:

- Local goodput: **0.1143 schema-valid pairs/s**
- Hub goodput: **1.4783 schema-valid pairs/s**
- Median paired Hub/Local goodput ratio: **14.40×**
- Paired-block range: **10.45×–14.89×**
- Local request latency p50/p95: **56.45/62.71 s**
- Hub request latency p50/p95: **4.63/7.40 s**

An independent validation run then re-executed all three frozen concurrency-8 blocks from the beginning. It measured a median Hub/Local ratio of **10.65×**, with a range of **9.22×–11.44×**. The validation confirms the direction and order of magnitude of the result while also showing that Hub performance varies with service conditions and time.

The appropriate conclusion is:

> Under this Sarah judge workload, the tested local serving configuration, and the observed Hub service conditions, NVIDIA Inference Hub delivered substantially higher schema-valid goodput and lower request latency than the current single-B200 local stack. The independent concurrency-8 validation continued to show approximately an order-of-magnitude goodput advantage.

This is a **same-canonical-request, advertised-model/precision-matched black-box hosting comparison**. It is not a direct hardware benchmark between a B200 and a known Hub GPU configuration, because the Hub checkpoint revision, tokenizer revision, hardware, replica count, and serving topology are not disclosed.

## 1. Background and Objective

Sarah's NDD deduplication pipeline uses an LLM judge to determine the relationship between two documents. For each candidate pair, the judge produces structured fields including:

- `same_duplicate_group`
- `a_can_replace_b`
- `b_can_replace_a`
- `relation_type`
- `material_difference`
- `fuzzy_scope`

The existing workflow can run Qwen3.8-27B-FP8 locally on a B200. A compatible Qwen3.8 service is also available through NVIDIA Inference Hub. The purpose of this experiment was to determine which deployment produced usable judge results faster under controlled and reproducible conditions.

The research question was:

> When the logical model, canonical request, output contract, input workload, validation behavior, retry behavior, and concurrency schedule are held constant, which warm endpoint produces more schema-valid unique pair decisions per second?

The study intentionally uses schema-valid goodput instead of raw request rate. Raw request rate would not account for parser correction, schema validation, failed-subset retries, or terminal errors and could therefore reward an endpoint for returning fast but unusable responses.

## 2. Systems Under Test

### 2.1 Local endpoint

The local endpoint used:

- Model: `Qwen/Qwen3.8-27B-FP8`
- Pinned revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Hardware: one NVIDIA B200
- Serving stack: Ray, Dynamo, and vLLM
- Tensor parallelism: 1
- Maximum model length: 32,768 tokens
- Maximum concurrent sequences: 8
- GPU memory utilization setting: 0.8
- Eager execution: enabled

The downloaded checkpoint was verified against the pinned revision. The provisioned model contained 66 FP8 weight shards with approximately 30.9 GB of weight files.

### 2.2 Hub endpoint

The Hub endpoint used:

- Base URL: `https://inference-api.nvidia.com/v1`
- Model name: `nvidia/qwen/qwen3.8-27b`
- OpenAI-compatible request interface
- No client-side requests-per-minute limiter

Hub authentication was injected only at request-forwarding time. The credential was not printed, stored in benchmark artifacts, or committed to Git.

### 2.3 Comparison boundary

The local model revision, tokenizer assets, and server configuration were pinned and inspectable. In contrast, Inference Hub did not disclose its checkpoint revision, tokenizer revision, serving hardware, number of replicas, parallelism strategy, or current service load.

Consequently, this report compares two complete serving stacks as observed by the client. It does not isolate model weights, GPU hardware, or serving software as independent causal factors.

## 3. Benchmark Implementation

### 3.1 Architecture

```text
Frozen, stratified document-pair workload
                    │
                    ▼
          Sarah NDD / Data Designer
  Same prompts, rubric, schema, and validation
                    │
                    ▼
              Logical model request
             Qwen/Qwen3.8-27B-FP8
                    │
                    ▼
             Local loopback relay
 Canonical hash and client token count computed
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
     Local target          Hub target
     Dynamo/vLLM        Rewrite model name
       1 × B200          and inject API auth
          │                    │
          └─────────┬──────────┘
                    ▼
 Same parser, validation, and failed-subset retry
                    │
                    ▼
 Terminal valid/error record, timing, and accounting
```

### 3.2 Endpoint-neutral runtime

The Sarah Ray/Data Designer pipeline was separated from model deployment. A shared judge runtime drives both the local and external OpenAI-compatible endpoints. The Local runtime remains compatible with the original Sarah behavior but is composed from the same shared pipeline plus a local Dynamo/vLLM inference server.

This design minimizes endpoint-specific code and ensures that pipeline orchestration is consistent between endpoints.

### 3.3 Canonical request relay

Data Designer always sends the same logical model identifier and request body to a local loopback relay. Before any endpoint-specific transformation, the relay:

1. Canonicalizes and hashes the request body.
2. Counts prompt tokens using the pinned client tokenizer.
3. Verifies the context budget.
4. Records submission and completion timestamps, HTTP status, safe numeric usage fields, and observed concurrency.

For the local endpoint, the relay forwards the request to Dynamo/vLLM. For the Hub endpoint, it changes only the target model name and adds authentication.

The relay does not log credentials or full document text. Raw source-derived workloads and model outputs remain in a private run directory and are not committed to the repository.

### 3.4 Immutable run identity

The benchmark identity incorporates digests of the configuration, Git revision, prompts, rubric, tokenizer assets, source candidates, payloads, and workload membership. Any material change requires a new run ID. This prevents a restarted experiment from silently changing its inputs or execution contract.

### 3.5 Timing and accounting

A measured block begins when its first measured request is submitted and ends only when its final unique pair reaches a schema-valid result or terminal error after validation. Retry time is included.

HTTP events and pair evaluations are accounted for separately. A failed transport attempt contributes to elapsed time and HTTP telemetry but cannot create an additional pair or alter Local/Hub workload identity.

## 4. Experimental Configuration

| Category | Configuration |
|---|---|
| Host | `umb-b200-218` |
| Local hardware | 1 × NVIDIA B200, approximately 183 GB visible memory |
| Local model | `Qwen/Qwen3.8-27B-FP8` |
| Local revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Local engine | Dynamo + vLLM, TP=1 |
| Local serving settings | `max_model_len=32768`, `max_num_seqs=8`, GPU memory utilization 0.8, eager execution enabled |
| Hub model | `nvidia/qwen/qwen3.8-27b` |
| Generation parameters | temperature=0, top_p=1, thinking disabled |
| Maximum output | 4,096 tokens |
| Request timeout | 600 s |
| Prompt tokenizer | Pinned Qwen tokenizer at the local model revision |
| Visible input limit | 20,000 tokens |
| Long-document window | 4,096 tokens with 512-token overlap |
| Context gate | `prompt_tokens + 4096 <= 32768` |
| Concurrency levels | Closed-loop concurrency 1, 2, 4, and 8 |
| Retry policy | Up to two failed-subset retries |

The local model cold-start time was **293.8 seconds**. Startup was reported separately and excluded from the warm-endpoint primary metric.

## 5. Workload Design and Execution Protocol

### 5.1 Workload construction

The benchmark used a frozen candidate-pair population from the source V0 run and sampling seed `26082701`.

The measured workload consisted of:

- 12 mutually disjoint blocks
- 500 unique pairs per block
- 6,000 unique pairs per endpoint
- Three blocks, or 1,500 pairs per endpoint, at each concurrency level
- An additional disjoint 100-pair warm-up set excluded from measured results

Every block contained 250 removal-track pairs and 250 cross-group-track pairs, with equal representation from five final prompt-length quintiles within each track. This stratification prevents the comparison from being dominated by one candidate type or prompt-length region.

### 5.2 Closed-loop scheduling

Both endpoints used the same closed-loop scheduler. At concurrency `C`, the client maintained at most `C` outstanding requests and submitted a replacement only after one completed.

The schedule was frozen and interleaved across endpoints, with six Local-first blocks and six Hub-first blocks. This reduces bias from always running one endpoint earlier or later. The local model was loaded once and remained active across both Local and Hub blocks.

### 5.3 Pre-measurement validation

Before formal measurement, the workflow completed implementation tests, checkpoint provisioning, static preflight, Hub model/quota/credit probes, a 40-pair Hub pilot, and a 100-pair warm-up.

The dynamic preflight established:

- Canonical request-hash equality: **true**
- Pinned-tokenizer prompt-count equality: **true**
- Hub quota probe at concurrency 8: **100 requests, 0 HTTP 429 responses**
- Hub acceptance of thinking-disabled generation: **true**
- Context-budget validation: **pass**

No measured speed conclusion would have been emitted if the Hub service had been quota limited or if either endpoint failed the schema-valid completion gate.

## 6. Evaluation Metrics

The primary metric was:

```text
schema-valid goodput
= terminal schema-valid unique pairs / measured block wall time
```

Concurrency 8 was pre-specified as the headline configuration. For each of the three concurrency-8 blocks, the benchmark computed:

```text
Hub block goodput / Local block goodput
```

The report uses the median and minimum-to-maximum range of those paired ratios. Secondary metrics include request latency, first-attempt validity, retries, terminal errors, HTTP status, timeouts, token accounting, judge-label agreement, and Local GPU telemetry.

## 7. Formal Results

### 7.1 Results by concurrency

| Concurrency | Local valid pairs/s | Hub valid pairs/s | Aggregate Hub/Local ratio | Local latency p50/p95 | Hub latency p50/p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0171 | 0.2597 | 15.21× | 48.07/52.21 s | 3.30/5.84 s |
| 2 | 0.0324 | 0.3808 | 11.77× | 50.28/54.93 s | 4.65/7.84 s |
| 4 | 0.0603 | 0.7554 | 12.54× | 53.27/58.54 s | 4.36/8.93 s |
| 8 | **0.1143** | **1.4783** | **12.94×** | **56.45/62.71 s** | **4.63/7.40 s** |

The aggregate ratio in this table divides each endpoint's aggregate goodput. The formal concurrency-8 headline instead takes the median of three within-block paired ratios. The two computations are not mathematically equivalent; therefore, the table's 12.94× aggregate ratio should not be substituted for the pre-specified 14.40× paired median.

### 7.2 Headline result

> At concurrency 8, Hub and Local achieved **1.4783** and **0.1143 schema-valid pairs/s**, respectively. The median paired Hub/Local throughput ratio was **14.3999×**, with a range of **10.4508×–14.8947×**. Local request latency p50/p95 was **56.45/62.71 seconds**, while Hub request latency p50/p95 was **4.63/7.40 seconds**.

### 7.3 Reliability and acceptance results

| Acceptance measure | Local | Hub |
|---|---:|---:|
| Measured unique pairs | 6,000 | 6,000 |
| Schema-valid terminal pairs | 6,000 | 6,000 |
| Valid rate | 100% | 100% |
| First-attempt valid rate | 100% | 100% |
| Pair retries | 0 | 0 |
| Terminal errors | 0 | 0 |
| HTTP 429 | 0 | 0 |
| HTTP 5xx | 0 | 0 |
| Timeouts | 0 | 0 |
| Context overflows | 0 | 0 |

The performance difference was not obtained by accepting more invalid responses, dropping difficult pairs, or operating under a quota-limited condition.

### 7.4 Judge-output agreement

Across 6,000 pairs valid on both endpoints:

- All six core fields matched exactly on **85.25%** of pairs.
- Individual field agreement ranged from **93.67% to 97.43%**.

The outputs are highly aligned at the field level but are not identical. Temperature zero does not guarantee bitwise determinism across different serving systems. The undisclosed Hub checkpoint and tokenizer revisions also prevent this result from being interpreted as a strict same-checkpoint reproducibility test.

### 7.5 Local GPU telemetry

During Local measured intervals only:

- GPU utilization: mean **35.24%**, p95 **37%**, maximum 100%
- GPU memory usage: mean approximately **147.5 GiB**, maximum approximately 180.4 GiB
- Power draw: mean **293.6 W**, p95 **471.6 W**
- SM clock: mean approximately 1,960 MHz

These measurements indicate that the benchmark represents the current Sarah, Ray, Dynamo, and vLLM stack rather than a theoretical B200 throughput limit. The telemetry suggests potential headroom but does not identify the dominant bottleneck by itself.

## 8. Independent Concurrency-8 Validation

After the formal run passed, a separate validation run was created under the final implementation revision. It imported the nine non-headline blocks as verified, immutable artifacts and re-executed all three frozen concurrency-8 blocks from the beginning.

The validation therefore added **1,500 fresh measured pairs per endpoint at concurrency 8**. It was not a fresh execution of all 6,000 pairs per endpoint; its purpose was specifically to reproduce the headline blocks.

| Run | Local pairs/s | Hub pairs/s | Median paired ratio | Paired range | Local p50/p95 | Hub p50/p95 |
|---|---:|---:|---:|---:|---:|---:|
| Formal run | 0.1143 | 1.4783 | 14.40× | 10.45×–14.89× | 56.45/62.71 s | 4.63/7.40 s |
| Independent C=8 validation | 0.1196 | 1.2386 | 10.65× | 9.22×–11.44× | 54.11/60.71 s | 5.72/9.03 s |

Local goodput changed only moderately. Hub goodput declined by approximately 16%, reducing the paired median ratio from 14.40× to 10.65×. This variation indicates that Hub performance is sensitive to service conditions or time. However, all three independent validation blocks still showed at least a 9.22× Hub advantage, confirming the direction and approximate order of magnitude of the formal result.

All-core-field agreement in the validation summary was **85.02%**, close to the formal run's 85.25%.

## 9. Recovery and Experimental Validity

The formal benchmark was completed through a controlled recovery process after an interrupted earlier run. The recovery run imported three completed, immutable paired blocks and replayed the remaining nine blocks.

Validity was preserved through the following rules:

1. A partial endpoint block has no valid completion marker and is excluded from summaries.
2. An interrupted endpoint block is replayed in full in a new attempt directory.
3. Partial terminal rows are never merged with a later attempt.
4. A completed block is immutable and referenced by hashes of its marker, request events, and terminal records.
5. Recovery uses a new run ID and records the parent and recovery Git revisions, configuration digest, workload digest, request-contract digest, and imported/replayed block membership.

The independent concurrency-8 validation provides an additional safeguard because all three headline blocks were later re-executed under the final source revision.

## 10. Prompt-Token Accounting

Canonical request bodies and pinned-client-tokenizer prompt counts matched for all 6,000 paired initial requests. However, Local and Hub provider-reported prompt usage differed on **99 of 6,000 paired initial requests**.

Possible causes include different internal tokenizer revisions, chat-template accounting, or provider-side request handling. Since the Hub implementation is not inspectable, these values are retained as black-box telemetry.

The final accounting policy was to:

- Use the pinned client tokenizer as the common prompt-token basis.
- Require canonical request and pinned-token equality for paired comparisons.
- Preserve and disclose provider-reported token drift.
- Avoid using provider-reported prompt counts to claim strictly comparable token throughput.

The primary pairs-per-second metric is unaffected because it is based on terminal unique pairs and measured wall time.

## 11. Runtime Analysis

The experiment did not consist of one 6,000-pair concurrency-8 run. It covered four concurrency levels, three repeated blocks per level, both endpoints, and sequential interleaved execution.

The accumulated measured endpoint wall time was:

- Local: **47.84 hours**
- Hub: **3.53 hours**
- Combined: **51.37 endpoint-hours**

This excludes checkpoint provisioning, preflight, warm-up, model startup, debugging, and interruption time. The Local concurrency-1 configuration alone required approximately **24.4 hours** for 1,500 pairs.

This explains why the benchmark took substantially longer than a prior Hub-only 20,000-pair V0 run. A Hub-only, higher-concurrency execution is not equivalent to a two-endpoint benchmark covering four concurrency levels and three repetitions. The Sarah judge also produces long structured outputs and performs validation included in the measured workflow.

## 12. Engineering Challenges and Resolutions

### 12.1 Idempotent preflight and provisioning

Early implementations treated volatile observations such as timestamps or available storage as evidence that a repeated preflight had changed. The final implementation separates stable identity fields from volatile observations. Existing reports and model-provision markers can be reused only when their stable content matches the frozen run.

### 12.2 Restart-safe block timing

Combining partial work from before and after a restart would invalidate wall-time measurements. Each attempt is now immutable, and any attempt without a completion marker is excluded. An interrupted block must be replayed in full.

### 12.3 Exact retry and transport accounting

A pair evaluation and an HTTP event are not interchangeable. The implementation pairs endpoints using the successful initial request for each immutable pair. Failed transport attempts remain visible in wall-time, latency, and status telemetry but cannot create additional workload rows.

### 12.4 Provider token-usage drift

The provider-reported token mismatch made a strict same-provider-count requirement unsuitable for a black-box endpoint. The benchmark therefore uses the pinned client tokenizer for comparable accounting while explicitly reporting provider drift.

### 12.5 Auditable recovery

Simple file reuse would not demonstrate that recovered blocks came from the same workload and request contract. The recovery mechanism hashes every imported artifact, records old and new source provenance, and identifies every imported and replayed block.

## 13. Discussion

Hub achieved higher schema-valid goodput at every tested concurrency and also delivered substantially lower request latency. Both endpoints scaled as concurrency rose, although neither scaled perfectly linearly.

The formal and validation runs agree on the direction and order of magnitude of the result but not on a single permanent speedup value. The most transparent internal statement is:

> The formal concurrency-8 run measured a 14.40× median paired advantage, while the independent validation measured 10.65×. A conservative summary is that Hub delivered approximately an order-of-magnitude schema-valid goodput advantage under the tested conditions.

The result should not be treated as a final measure of what a B200 can achieve. The local configuration used eager execution and a maximum of eight sequences, and the measured GPU telemetry indicates potential optimization headroom.

The high individual-field agreement suggests that both endpoints generally implemented the intended judge behavior. Nevertheless, the approximately 85% exact all-field agreement warrants targeted analysis before assuming semantic interchangeability for every pair.

## 14. Threats to Validity and Limitations

1. **Hub infrastructure is opaque.** The remote checkpoint, tokenizer, GPU type, replica count, and serving configuration are not disclosed.
2. **This is a serving-stack comparison.** The result cannot be attributed solely to hardware or model implementation.
3. **Hub performance varies over time.** The independent validation measured lower Hub goodput than the formal run.
4. **Only one local configuration was tested.** The result does not represent an optimized B200 performance ceiling.
5. **Provider token telemetry differs.** Client-side canonical accounting is consistent, but provider-reported prompt usage differs on 99 paired requests.
6. **Outputs are not identical.** Exact agreement across all six core fields is approximately 85%, although individual-field agreement exceeds 93%.
7. **Cost was not measured.** No cost-per-valid-pair conclusion can be made from throughput alone.
8. **The workload is source-specific.** Results apply to the sampled Sarah dedup workload and may not generalize to unrelated LLM tasks.

## 15. Conclusion

The benchmark established an auditable, endpoint-neutral method for comparing the Sarah NDD judge on a local B200 deployment and NVIDIA Inference Hub. It controlled the canonical request, prompt resources, tokenizer accounting, source workload, concurrency schedule, validation logic, retry policy, and block timing.

All 6,000 measured pairs per endpoint completed with schema-valid results and without quota, HTTP 5xx, timeout, or context-overflow failures. At concurrency 8, the formal run measured 1.4783 valid pairs/s on Hub and 0.1143 valid pairs/s locally, corresponding to a 14.40× median paired advantage. A complete independent replay of the three concurrency-8 blocks measured a 10.65× median paired advantage.

The independent run confirms that Hub was substantially faster under the tested conditions while demonstrating that the exact speedup is sensitive to Hub service conditions. The strongest supportable conclusion is that the observed Hub service delivered approximately an order-of-magnitude higher schema-valid goodput than the current single-B200 Sarah serving stack.

This conclusion must be framed as a comparison of the tested end-to-end systems. It should not be generalized into a direct hardware comparison or a claim about an optimally tuned local deployment.

## 16. Recommended Follow-up Work

1. **Report both runs.** Present 14.40× as the formal result and 10.65× as the independent validation rather than selecting only the higher estimate.
2. **Optimize the local serving path.** Profile Ray/Data Designer, relay overhead, structured generation, CPU processing, and vLLM scheduling. Evaluate non-eager execution, higher sequence limits, and batching changes under a new run identity.
3. **Repeat concurrency-8 blocks across time windows.** Quantify Hub variability at different times of day.
4. **Add cost-normalized metrics.** Compare cost per 1,000 schema-valid pairs in addition to throughput and latency.
5. **Analyze output disagreements.** Stratify mismatched pairs by track, prompt length, and judge field.
6. **Re-evaluate after local tuning.** A production decision should compare Hub against the best practical local configuration, not only the current baseline.

## Appendix A. Reproducibility Information

- Branch: `experiment/inference-hosting`
- Final benchmark commit: `5365eabec768f7191704bf00746813a3ebcc1a25`
- Local model revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Formal run ID: `qwen38-hosting-20260829T203512Z-2d432ca900a4-recovery`
- Independent validation run ID: `qwen38-hosting-20260831T154448Z-2d432ca900a4-c8-validation`
- Formal result: `/raid/hfang/ihb/runs/qwen38-hosting-20260829T203512Z-2d432ca900a4-recovery/RESULTS.md`
- Validation result: `/raid/hfang/ihb/runs/qwen38-hosting-20260831T154448Z-2d432ca900a4-c8-validation/RESULTS.md`

## Appendix B. Anticipated Review Questions

### Does recovery invalidate the timing comparison?

No partial measurements were merged. An interrupted block was excluded and replayed in full in a new immutable attempt. Completed blocks were accepted only with validated markers and artifact hashes. All three headline concurrency-8 blocks were also independently re-executed under the final revision.

### Why is the formal concurrency-8 result 14.40× when the concurrency table shows 12.94×?

The 14.40× headline is the median of three within-block Hub/Local ratios. The 12.94× figure divides the endpoints' aggregate concurrency-8 goodputs. The benchmark protocol pre-specified the paired median as the headline metric.

### Why are the outputs not identical when temperature is zero?

Temperature zero does not guarantee cross-system bitwise determinism. Hub also does not disclose whether its checkpoint, tokenizer revision, or serving implementation is identical to the local system.

### Why did 6,000 pairs require more than two days?

Each endpoint executed four concurrency levels with three 500-pair repetitions at each level, and endpoint blocks ran sequentially. The Local measured intervals alone totaled 47.84 hours, including approximately 24.4 hours for concurrency 1.

### Does this result justify an immediate production migration to Hub?

The throughput and latency evidence favors Hub for this workload, but a production decision also requires cost, data-governance, reliability, availability, and SLA analysis, plus comparison against an optimized local serving configuration.
