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

The smoke config freezes `nvidia/deepseek-ai/deepseek-v4-flash` on `https://inference-api.nvidia.com/v1`. The full config freezes
`nvidia/deepseek-ai/deepseek-v4-pro`, which is required for formal V0 run creation.

The first V0 implementation deliberately uses the proposal's allowed path of skipping the optional minimum-diff challenge slice. The full 10,000 Step 5a budget is a uniform sample of actual keeper-to-removed decisions.

## Evaluation contract

The relaxed lexical grid is frozen as `(5,1), (6,1), (7,1), (8,1)` after real-corpus calibration. The pilot still applies the proposal's hard rule: select a configuration with median cross-group candidates in `[20,50]` closest to 35, or fail. The per-anchor safety limit is 250,000 and every trial is written to `retrieval_config.json`.

JSON-mode results are parsed strictly and validated locally. Evidence quotes may be deterministically realigned only by exact search in judge-visible text; unalignable evidence is dropped without changing the decision fields. Results record the provider-response SHA-256 and versioned repair events. Validation failures receive up to the frozen retry budget with safe structured feedback, including missing and extra field names but never raw provider responses or document text.

## How to run the evaluation

Run every command below from the repository root.

### 1. Prepare the environment

You need Linux with CUDA 12 support, access to the frozen 10M corpus/SUT handoff, sufficient `/raid` space, and an NVIDIA API
key for the live Judge backend.

Create the CUDA dedup environment while keeping both the environment and cache on `/raid`:

```bash
export UV_PROJECT_ENVIRONMENT=/raid/hfang/dedup_eval_env
export UV_CACHE_DIR=/raid/hfang/dedup_eval_cache/uv
export HF_HOME=/raid/hfang/dedup_eval_cache/huggingface
/raid/hfang/dedup_eval_tools/uv sync --frozen --extra deduplication_cuda12
```

### 2. Configure data and credentials

Review the absolute `handoff_root`, `output_root`, and `cache_root` paths before running:

- Smoke: `eval/dedup/resources/v0_config.example.json`
- Full: `eval/dedup/resources/v0_config.full.json`

Both configs contain machine-specific paths and may need local edits for another host.

Put `NVIDIA_API_KEY` in a repository-root `.env` file for the live backend:

```dotenv
NVIDIA_API_KEY=replace_with_your_nvidia_api_key
```

The CLI does not override an already exported value. `.env` is Git-ignored, and the key is never written to run artifacts.

### 3. Run preflight, then the evaluation

Preflight validates the handoff, storage, tokenizer, GPU retrieval prerequisites, and Judge contract without creating a run.

Smoke:

```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.example.json \
  --profile smoke

/raid/hfang/dedup_eval_env/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.example.json \
  --profile smoke
```

Full:

```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.full.json \
  --profile full

/raid/hfang/dedup_eval_env/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.full.json \
  --profile full
```

If the provider rejects `json_schema` but passes JSON mode, preflight stops with `STRUCTURED_OUTPUT_FALLBACK_REQUIRED`; update the config to `json_object_plus_local_schema` and rerun so the fallback is explicit and frozen.

Each `run` command prints the immutable `run_root` used by all later commands.

### 4. Resume, inspect, and validate

```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup resume --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup status --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup validate --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
```

Run creation also freezes the `eval/dedup` source-tree digest. `run` and `resume` stop with `RESUME_SOURCE_MISMATCH` if the implementation changes, preventing mixed-code artifacts under one evaluation run ID.

### 5. Read or regenerate the automated report

Step 10 publishes the automated report without waiting for Human QA. For an older immutable run whose Step 9 artifacts are complete, render a versioned derived report without altering its stage markers:

```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup report --run-root <v0_run>
```

By default, derived reports are exported to
`/home/nfs/hfang/dedup_eval/dedup_eval_runs/<evaluation_run_id>/v0_run/reports/`.
Large run artifacts remain under the frozen run root on `/raid`. Use `--output-root`
to select a different run-scoped report export root.

Every automated render also writes a matching self-contained `pair_explorer[.<output-label>].html`. The report embeds
deterministically selected removal-disagreement and cross-group-positive examples, links each example to a focused review queue,
and records the dashboard checksum plus selected pair IDs in `report_generation_manifest[.<output-label>].json`. The Pair
Explorer separates the observed SUT grouping/action, the Judge group and directional-replacement verdicts, and the derived
evaluation outcome. It also includes endpoint group/action context, Judge evidence-repair coverage, 5b evaluation-retrieval
scores, bounded deterministic group-member summaries, and explicit SUT-provenance availability. Human review annotations are
stored in browser `localStorage` and can be imported or exported as CSV/JSON; they do not modify frozen run artifacts.

### 6. Complete and import Human QA

Step 8 writes the self-contained `reports/human_qa_dashboard.html`. Its selector switches between the blind sample and
diagnostic set while keeping their progress and CSV exports separate. The reviewer-blind UI omits Judge decisions, payload
hashes, SUT outcomes, and sampling strata.

Only three decisions are required:

- `same_duplicate_group`
- `a_can_replace_b`
- `b_can_replace_a`

Relation type, material difference, fuzzy scope, reason codes, and notes are optional. Reviews live in browser `localStorage`,
so export the CSV before changing browsers or clearing site data.

For a full run, export the completed **blind-sample** CSV and import it. `qa-import` validates IDs, fields, enums, and reason codes:

```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup qa-import \
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
