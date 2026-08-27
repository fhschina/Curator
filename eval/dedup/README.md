# NeMo Curator Fuzzy Dedup Evaluation

## Introduction

This package implements a reproducible, ten-stage evaluation of a completed fuzzy-deduplication system over the frozen
CC-MAIN-2025-26 10M handoff. It evaluates existing SUT outputs; it does not rerun exact deduplication or physically remove rows.

It answers four practical questions:

- Were keeper-to-removed decisions supported by the pair evidence?
- How often do sampled cross-group candidates appear to be missed duplicates?
- Which languages, group sizes, and relation types account for failures?
- Where do the SUT, fuzzy-dedup outcome, LLM Judge, and human reviewers disagree?

The automated benchmark uses a schema-constrained LLM Judge as its reference, not as human ground truth. Human QA is an
independent calibration layer: the random blind sample supports the primary agreement estimate, while the disagreement-focused
diagnostic set is a separate challenge set for regression analysis and debugging. Do not combine them into one headline metric.

The design source is the [V0.3 operational proposal](docs/proposals/v0.3/NeMo_Curator_Dedup_Evaluation_V0_3_Operational_Proposal.docx).
Earlier revisions are in the [proposal archive](docs/proposals/archive/). The latest technical presentation is
[NeMo Curator Dedup Evaluation — Current Status](docs/presentations/NeMo_Curator_Dedup_Evaluation_Current_Status.pptx).

## Current results and dashboards

These stable links require access to the NVIDIA internal network:

| Result | Link | Contents |
|---|---|---|
| Latest automated result | [Pair Explorer](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/) | Result-linked removal errors, cross-group positives, Judge decisions, provenance, and group context. |
| Latest Human QA set | [Human QA Dashboard](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/human-qa/) | The 200-pair blind sample and 200-pair diagnostic set, review progress, and CSV export. |

The shared Human QA URL always serves the dashboard from the latest run with a completed Step 8. Every run keeps its own
immutable HTML file, so automatic updates do not overwrite historical results. The current reference run is
`dedup-full-20260813T220949Z-d4c37bb483`.

## Runtime profiles

- `smoke`: all 10,008,061 handoff documents, 20 anchors, 50 removal decisions, up to 50 cross-group pairs, and at most 100 judge requests. Its report is explicitly non-V0.
- `full`: 1,000 anchors, 10,000 removal decisions, up to 10,000 cross-group pairs, a 200-pair blind Human QA sample, and a separate 200-pair diagnostic set.

The default smoke and full configs now use Sarah's MinHash/surface-overlap NDD Judge contract,
`dedup-judge-sarah-minhash-v1`, with the local `Qwen/Qwen3.8-27B` model on one B200. Both Sarah-backed profiles are
explicitly non-formal-V0.

The original NVIDIA API setup is unchanged in
`v0_config.legacy_nvidia.example.json` and `v0_config.legacy_nvidia.full.json`. The legacy full profile freezes
`nvidia/deepseek-ai/deepseek-v4-pro` and remains the formal V0 configuration.

The first V0 implementation deliberately uses the proposal's allowed path of skipping the optional minimum-diff challenge slice. The full 10,000 Step 5a budget is a uniform sample of actual keeper-to-removed decisions.

## Evaluation contract

The relaxed lexical grid is frozen as `(5,1), (6,1), (7,1), (8,1)` after real-corpus calibration. The pilot still applies the proposal's hard rule: select a configuration with median cross-group candidates in `[20,50]` closest to 35, or fail. The per-anchor safety limit is 250,000 and every trial is written to `retrieval_config.json`.

All backends are validated locally. The legacy JSON-mode path retains its evidence realignment policy. The Sarah path runs one
Ray/Dynamo/vLLM/Qwen service across the initial batch and at most two failed-subset retries. Missing rows, duplicate rows,
malformed rubrics, invalid enums, and cross-field inconsistencies are retried with safe structured validation feedback only.
Every pair ends with either a schema-valid result or an explicit terminal error; valid records are fsynced individually to the
existing Judge cache so an interrupted run submits only pending pairs on resume.

The tracked Sarah YAML disables Data Designer's batch-level early shutdown, sets deterministic conversation restarts to zero,
and allows two parser-correction turns per row. This prevents a handful of malformed fenced-JSON replies from abandoning the
rest of a blind batch; at temperature zero, correction feedback is useful while a fresh restart would repeat the same reply.
The correction trace and NDD reasoning remain runtime-only and are never written to formal dedup artifacts.

The Sarah YAML, Jinja prompts, and compatibility shim are content-hashed into the Judge contract, run manifest, and cache key.
Changing any of them creates a different execution contract.

### Judge contract versions

- `dedup-judge-sarah-minhash-v1` uses Sarah's NDD surface-overlap policy and `judge-visible-payload-v2`, but adapts the
  output to the existing `dedup-judge-output-v0` artifact schema. NDD enums are uppercased, boolean reason rubrics become the
  existing flat reason-code array, confidence uses a discrete rubric, and `evidence=[]`. NDD reasoning is not stored; only a
  canonical response SHA-256 is retained.
- `dedup-judge-v0` with `dedup-judge-output-v0` preserves the original prompt, flat reason-code array, and
  `judge-visible-payload-v1` neutral document metadata.
- `dedup-judge-v1` with `dedup-judge-output-v1` uses the polished multilingual policy, structured V2 reasons, stronger
  cross-field consistency validation, and `judge-visible-payload-v2`. The v2 payload exposes only cleaned text plus explicit
  long-document truncation windows; it excludes URL, hostname, timestamps, language tags, character/token counts, and all SUT
  provenance.

Select v1 by changing both version fields together:

```json
{"prompt_version":"dedup-judge-v1","schema_version":"dedup-judge-output-v1"}
```

The blind Judge records observable dedup risk factors, not hidden SUT error direction. Reporting may combine those factors with
the SUT result later to derive overmerge or undermatch categories without leaking the SUT decision into the Judge prompt.

## How to run the evaluation

Run every command below from the repository root.

### 1. Prepare the environment

You need Linux with CUDA 12, access to the frozen 10M handoff, the local model at
`/raid/hfang/hf_cache/Qwen3.8-27B`, and sufficient `/raid` space. The project requires uv `>=0.12.0`; the validated
installation used uv 0.12.3 and Python 3.11.

Keep the Sarah-capable environment separate from the existing dedup environment:

```bash
export UV_PROJECT_ENVIRONMENT=/raid/hfang/llm_judge_env_pr2324_latest
export UV_CACHE_DIR=/raid/hfang/dedup_eval_cache/uv
export HF_HOME=/raid/hfang/dedup_eval_cache/huggingface
export PATH=/raid/hfang/llm_judge_tools/bin:$PATH
/raid/hfang/dedup_eval_tools/uv sync --locked --python 3.11 \
  --extra deduplication_cuda12 --extra text_cuda12 --extra sdg_cuda12
```

`text_cuda12` supplies the text/runner stack, `sdg_cuda12` supplies Data Designer and Dynamo, and
`deduplication_cuda12` supplies the RAPIDS retrieval path. The lock fixes `data-designer==0.9.1` and applies
`pyarrow>=19,<24` because RAPIDS/cuDF does not accept Data Designer's upstream `pyarrow>=24` requirement; the validated
environment resolves PyArrow 23.0.1. Do not replace the locked install with an unconstrained `uv pip install`. Because uv
environments normally omit pip but Ray 2.57's uv actor bootstrap needs it, the local Judge runner seeds pip from Python's
bundled `ensurepip` wheel before Ray starts when necessary.

The local YAML expects `etcd` and `nats-server` on `PATH`, writes Ray/uv/checkpoint state under
`/raid/hfang/dedup_eval_cache`, allows 1800 seconds for runtime setup, and explicitly loads
`local_ndd/cutlass_compat/sitecustomize.py` for the verified CUTLASS compatibility aliases. Gemma is not started by the
default dedup backend. Keep `ray_temp_dir` short because Ray's dashboard Unix socket has a 107-byte path limit after its
session suffix is added; `preflight` and `LocalJudgeRuntime` reject paths that exceed the conservative budget.

### 2. Configure data and credentials

Review the absolute handoff, output, cache, model, Ray, and checkpoint paths before running:

- Sarah/Qwen default smoke: `eval/dedup/resources/v0_config.example.json`
- Sarah/Qwen default full: `eval/dedup/resources/v0_config.full.json`
- Legacy NVIDIA smoke: `eval/dedup/resources/v0_config.legacy_nvidia.example.json`
- Legacy NVIDIA formal V0 full: `eval/dedup/resources/v0_config.legacy_nvidia.full.json`

Sarah's local backend needs no API key. Only the legacy NVIDIA configs read `NVIDIA_API_KEY`; place it in a repository-root
`.env` or export it before invoking the CLI:

```dotenv
NVIDIA_API_KEY=replace_with_your_nvidia_api_key
```


The CLI does not override an exported value. `.env` is Git-ignored, and credentials are never written to run artifacts.

### 3. Run preflight, then the evaluation

Preflight validates the handoff, storage, tokenizer, GPU retrieval prerequisites, and Judge contract without creating a run.

Smoke:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.example.json \
  --profile smoke

/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.example.json \
  --profile smoke
```

Full:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.full.json \
  --profile full

/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.full.json \
  --profile full
```

Legacy formal V0:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.legacy_nvidia.full.json \
  --profile full

/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.legacy_nvidia.full.json \
  --profile full
```

The `STRUCTURED_OUTPUT_FALLBACK_REQUIRED` preflight error and `json_object_plus_local_schema` fallback apply only to the
legacy NVIDIA API backend.

Each `run` command prints the immutable `run_root` used by all later commands.

### 4. Resume, inspect, and validate

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup resume --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup status --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run

/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup validate --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
```

Run creation freezes the `eval/dedup` Python, YAML, and Jinja source digest. `run` and `resume` stop with
`RESUME_SOURCE_MISMATCH` if the implementation changes. Resume an older run only with the source revision that created it;
start a new immutable run after changing Judge code or resources.

### 5. Read or regenerate the automated report

Step 10 publishes the automated report without waiting for Human QA. For an older immutable run whose Step 9 artifacts are complete, render a versioned derived report without altering its stage markers:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup report --run-root <v0_run>
```

By default, derived reports are exported to
`/home/nfs/hfang/dedup_eval/dedup_eval_runs/<evaluation_run_id>/v0_run/reports/`.
Large run artifacts remain under the frozen run root on `/raid`. Use `--output-root`
to select a different run-scoped report export root.

Derived renders default to the `automated_v9` output label and also write a matching self-contained
`pair_explorer[.<output-label>].html` plus `report_generation_manifest[.<output-label>].json`. The report retains the detailed
headline, accounting, and `By ...` slice tables, but routes pair-level examples, document excerpts, and identifiers to the Pair
Explorer instead of embedding them in the report body. Stage-level operations, reproducibility data, machine paths, and the
artifact inventory remain in the appendices.

After a derived report is written, the renderer copies everything before the first appendix verbatim to
`eval/dedup/RESULTS.md`. This keeps the repository-visible results synchronized without recomputing or separately summarizing
the metrics. The Pair Explorer separates the observed SUT grouping/action, the Judge group and directional-replacement verdicts,
and the derived evaluation outcome. It also includes endpoint group/action context, Judge evidence-repair coverage, 5b
evaluation-retrieval scores, bounded deterministic group-member summaries, and explicit SUT-provenance availability. Human
review annotations are stored in browser `localStorage` and can be imported or exported as CSV/JSON; they do not modify frozen
run artifacts.

### 6. Complete and import Human QA

Step 8 writes the self-contained `reports/human_qa_dashboard.html`. Its selector switches between the blind sample and
diagnostic set while keeping their progress and CSV exports separate. The reviewer-blind UI omits Judge decisions, payload
hashes, SUT outcomes, and sampling strata. The labels CSV stays compact and directly importable by `qa-import`; the packet JSON
is the self-contained sharing artifact, with each review next to its two Judge-visible documents. The Sarah default and Judge
v1 packets contain only cleaned text and explicit long-document windows. Legacy Judge v0 packets retain their neutral metadata
contract.

When the blind Human QA budget exhausts the candidate population (as it can in the smoke profile), the independent diagnostic
set is empty by construction. Step 8 still writes its empty packet and labels artifacts and records zero counts; the dashboard
then exposes only the non-empty blind sample, and the remaining stages continue normally.

Only three decisions are required:

- `same_duplicate_group`
- `a_can_replace_b`
- `b_can_replace_a`

Relation type, material difference, fuzzy scope, reason codes, and notes are optional. For CSV compatibility, optional Human QA
reason labels retain the legacy flat taxonomy and are independent of the automated Judge's versioned reason structure. Reviews
live in browser `localStorage`, so export the CSV before changing browsers or clearing site data.

For a full run, export the completed **blind-sample** CSV and import it. `qa-import` validates IDs, fields, enums, and reason codes:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup qa-import \
  --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run \
  --labels /path/to/completed_blind_labels.csv
```

The import writes `reports/human_qa_report.md` and `reports/human_qa_metrics.json` without changing the automated report.
Keep the diagnostic export as a separate challenge/regression result; do not merge it into the blind-sample headline metric.

`dashboard_server.py` serves the latest Human QA dashboard through the stable URL above. It switches only after Step 8 has a
complete marker and never copies, overwrites, or deletes a run-scoped dashboard. Historical run files remain immutable.

## Key outputs

All canonical artifacts live below the immutable `run_root`:

| Stage | Artifact | Purpose |
|---|---|---|
| 6 | `data/judge_results.jsonl` | Schema-valid automated Judge decisions. |
| 6 | `data/human_qa_packet.jsonl`, `data/human_qa_labels.csv` | Frozen blind Human QA packet and template. |
| 8 | `data/pair_comparisons.parquet` | Pair-level SUT-versus-Judge comparisons. |
| 8 | `data/human_qa_diagnostic_packet.jsonl`, `data/human_qa_diagnostic_labels.csv` | Disagreement challenge set. |
| 8 | `reports/human_qa_dashboard.html` | Self-contained Human QA reviewer UI. |
| 9 | `reports/metrics.json`, `reports/metrics_by_slice.csv` | Canonical metrics and slice results. |
| 9 | `reports/pipeline_accounting.csv` | Stage and population accounting. |
| 10 | `reports/final_report.md` | Automated evaluation report. |
| 10 | `reports/pair_explorer.html` | Self-contained result explorer. |

Formal V0 reporting requires at least 99% schema-valid Judge completion. Step 10 remains independent of Human QA completion.

## Module and artifact map

- `handoff/`: validates and wraps Steps 1–2 artifacts.
- `pair_construction/`: writes document outcomes, anchors, the two pair tracks, canonical queue, and provenance.
- `judging/`: constructs blind payloads, handles long documents, calls a provider, validates the schema, retries, and resumes.
- `analysis/`: creates the partial graph, pair comparison, and frame-valid metrics.
- `run.py`: stage transactions and resume validation; domain logic remains in the packages above.
- `report.py`: deterministic fact reporting and example selection, bounded DeepSeek recommendations, report publication,
  and the independent blind human-QA exchange.
- `dashboard.py`: Pair Explorer data joins, bounded group context, static HTML rendering, and local human-review tooling.
- `human_qa_dashboard.py`: reviewer-blind QA packet rendering, browser-local progress, and contract-compatible CSV export.
- `dashboard_server.py`: stable internal URLs for Pair Explorer and the latest completed, immutable Human QA dashboard.

Large generated data remains beneath the configured external run root. Derived automated and Human-QA reports are exported beneath `/home/nfs/hfang/dedup_eval/dedup_eval_runs/<evaluation_run_id>/v0_run/reports/` by default; `eval/dedup/docs/` is reserved for design documents, proposals, and templates rather than run-specific results. Imports are side-effect free.
