# Qwen3.8 hosting benchmark

This package runs a paired, endpoint-neutral comparison of Sarah's dedup LLM Judge on one local B200 and NVIDIA Inference Hub. It answers one question: under the same logical model, rendered request, output contract, input workload, and closed-loop concurrency, which warm endpoint produces more schema-valid unique pairs per second?

The comparison is an advertised model/precision matched black-box hosting benchmark. The local side is pinned to `Qwen/Qwen3.8-27B-FP8` revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; the Hub service does not disclose its checkpoint revision or serving hardware.

## Controls

- Both endpoints use the same Sarah NDD system prompt, pair prompt, rubric, structured output, parser correction, and at most two failed-subset retries.
- Data Designer always addresses the logical model `Qwen/Qwen3.8-27B-FP8`. A loopback relay hashes the canonical request before it rewrites only the target model name and adds Hub authentication.
- There is no client RPM limiter. Each block uses the same closed-loop `max_parallel_requests` value: 1, 2, 4, or 8.
- The relay rejects prompts whose locally counted Qwen tokens plus 4,096 output tokens exceed 32,768. It records hashes, timings, status, safe numeric usage fields, and concurrency—not prompts, documents, responses, or credentials.
- The 100-pair warm-up is also the no-cap Hub quota probe. Any 429 stops the run before measured results can support a speed conclusion.
- The main block duration begins at the first measured request submission and ends when the final unique pair reaches a schema-valid or terminal-error result after validation. Retry time is included.

## Host and checkout

Run only on `umb-b200-218.cl1u1.colossus.nvidia.com`, after independently verifying its ED25519 host-key fingerprint through a trusted NVIDIA channel. Use the isolated checkout and the already-locked environment:

```bash
git clone --branch experiment/inference-hosting \
  https://github.com/fhschina/Curator.git \
  /raid/hfang/Curator-inference-hosting
cd /raid/hfang/Curator-inference-hosting

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/raid/hfang/dedup_eval_cache/huggingface
export PATH=/raid/hfang/llm_judge_tools/bin:$PATH
read -rsp "NVIDIA API key: " NVIDIA_API_KEY && echo
export NVIDIA_API_KEY
```

Do not create or re-resolve another Python environment. Use `/raid/hfang/llm_judge_env_pr2324_latest/bin/python`.

## Run protocol

`prepare` rebuilds the Qwen-tokenized visible payload, allocates 100 disjoint warm-up pairs and twelve disjoint 500-pair blocks, and materializes the pinned FP8 checkpoint. The run identity covers the config, Git commit, prompt, rubric, tokenizer, source artifacts, payload hashes, and block membership.

```bash
BENCH_PYTHON=/raid/hfang/llm_judge_env_pr2324_latest/bin/python
BENCH_CONFIG=eval/dedup/hosting_benchmark/qwen38_fp8.json

$BENCH_PYTHON -m eval.dedup.hosting_benchmark prepare --config "$BENCH_CONFIG"
$BENCH_PYTHON -m eval.dedup.hosting_benchmark preflight --run-root /raid/hfang/ihb/runs/<run_id>
$BENCH_PYTHON -m eval.dedup.hosting_benchmark run --run-root /raid/hfang/ihb/runs/<run_id>
$BENCH_PYTHON -m eval.dedup.hosting_benchmark summarize --run-root /raid/hfang/ihb/runs/<run_id>
```

`preflight` performs static host, GPU, credential-presence, checkpoint, resource-digest, workload-digest, and local-engine checks. To avoid loading the local model twice, `run` starts it once and performs the dynamic smoke/quota/request-equality checks during the unmeasured warm-up. It writes `dynamic_preflight.json` before any measured block.

The fixed schedule contains three 500-pair blocks per concurrency. Every block has 250 removal-track and 250 cross-group-track pairs, with 50 rows from each track/length quintile. Endpoint order alternates globally, yielding six Local-first and six Hub-first blocks.

An interrupted block has no completion marker. Re-running `run` creates a new attempt directory and reruns that whole endpoint block; partial terminal rows are never merged. A completed block is immutable and is reused after restart.

## Acceptance and artifacts

`summarize` produces the headline only if both endpoints have 6,000 terminal unique-pair records, zero 429s, zero context overflows, and at least 99% schema-valid completions. It reports per-concurrency goodput, paired C=8 ratios, request latency, raw request rate, token rate, retries, terminal errors, completion length, label agreement, local GPU telemetry, and cold-start time.

The private run directory contains source-derived payloads, terminal records, request events, checkpoints, and GPU samples. It is created with mode 0700 under `/raid/hfang/ihb/runs`; never copy those raw artifacts into Git. Only the aggregate `RESULTS.md`, `summary_by_concurrency.csv`, `summary_by_block.csv`, and `summary.json` are safe result candidates after checking that they contain no document text or credential material.

Do not push, merge, or force-update `dedup-eval`. Implementation and aggregate results belong only on `experiment/inference-hosting`.
