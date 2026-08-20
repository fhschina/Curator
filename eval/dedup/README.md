# NeMo Curator Fuzzy Dedup Evaluation

This package implements the ten-step V0.3 operational proposal over the frozen CC-MAIN-2025-26 10M handoff. It evaluates a completed fuzzy-dedup SUT; it does not rerun exact deduplication or physically remove corpus rows.

The current design source is the [V0.3 operational proposal](docs/proposals/v0.3/NeMo_Curator_Dedup_Evaluation_V0_3_Operational_Proposal.docx). Earlier proposal revisions are retained in the [proposal archive](docs/proposals/archive/), and the technical presentation is under [docs/presentations](docs/presentations/).

## Runtime profiles

- `smoke`: all 10,008,061 handoff documents, 20 anchors, 50 removal decisions, up to 50 cross-group pairs, and at most 100 judge requests. Its report is explicitly non-V0.
- `full`: 1,000 anchors, 10,000 removal decisions, up to 10,000 cross-group pairs, and a separate 200-pair human-QA double-check.

The checked-in operational config temporarily freezes `nvidia/deepseek-ai/deepseek-v4-flash` on `https://inference-api.nvidia.com/v1` for the NON-V0 smoke run. Formal `full` run creation remains blocked unless the judge is restored to `nvidia/deepseek-ai/deepseek-v4-pro`.

The first V0 implementation deliberately uses the proposal's allowed path of skipping the optional minimum-diff challenge slice. The full 10,000 Step 5a budget is a uniform sample of actual keeper-to-removed decisions.

## Environment and commands
The relaxed lexical grid is frozen as `(5,1), (6,1), (7,1), (8,1)` after real-corpus calibration. The pilot still applies the proposal's hard rule: select a configuration with median cross-group candidates in `[20,50]` closest to 35, or fail. The per-anchor safety limit is 250,000 and every trial is written to `retrieval_config.json`.

JSON-mode results are parsed strictly and validated locally. Evidence quotes may be deterministically realigned only by exact search in judge-visible text; unalignable evidence is dropped without changing the decision fields. Results record the provider-response SHA-256 and versioned repair events. Validation failures receive up to the frozen retry budget with safe structured feedback, including missing and extra field names but never raw provider responses or document text.


Create the CUDA dedup environment from the repository root while keeping both the environment and cache on `/raid`:

```bash
export UV_PROJECT_ENVIRONMENT=/raid/hfang/dedup_eval_env
export UV_CACHE_DIR=/raid/hfang/dedup_eval_cache/uv
export HF_HOME=/raid/hfang/dedup_eval_cache/huggingface
/raid/hfang/dedup_eval_tools/uv sync --frozen --extra deduplication_cuda12
```

Then run:

```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.example.json \
  --profile smoke

/raid/hfang/dedup_eval_env/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.example.json \
  --profile smoke
```

Put `NVIDIA_API_KEY` in a repository-root `.env` file for the live backend:

```dotenv
NVIDIA_API_KEY=replace_with_your_nvidia_api_key
```

The CLI loads that exact file without overriding a value already exported in the process environment. `.env` is Git-ignored, and the key value is never written to arguments, manifests, caches, logs, or reports.

If the provider rejects `json_schema` but passes JSON mode, preflight stops with `STRUCTURED_OUTPUT_FALLBACK_REQUIRED`; update the config to `json_object_plus_local_schema` and rerun so the fallback is explicit and frozen.

Resume and audit an existing run:

```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup resume --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup status --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup validate --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
```

Run creation also freezes the `eval/dedup` source-tree digest. `run` and `resume` stop with `RESUME_SOURCE_MISMATCH` if the implementation changes, preventing mixed-code artifacts under one evaluation run ID.

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

For a full run, complete the frozen `data/human_qa_labels.csv` template outside the immutable bundle and import it. `qa-import` validates the labels and writes the independent `reports/human_qa_report.md` and `reports/human_qa_metrics.json`; it does not rewrite the automated report:

```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup qa-import --run-root <v0_run> --labels <completed_labels.csv>
```

Step 8 also writes the self-contained reviewer interface `reports/human_qa_dashboard.html`. A selector switches between the
blind sample and diagnostic set while keeping their progress and CSV exports separate. The dashboard is reviewer-blind: it
includes the visible document payloads but omits Judge verdicts, payload hashes, SUT outcomes, and sampling strata. Reviews are
saved in browser `localStorage`; export the complete labels CSV before moving browsers or clearing site data. Only
`same_duplicate_group`, `a_can_replace_b`, and `b_can_replace_a` are required. The exported CSV also supports the three optional
categorical fields, a JSON-array `reason_codes` column using the Judge schema's exact vocabulary, reviewer status, and notes.
`qa-import` accepts this schema and remains backward compatible with the earlier template that omitted `reason_codes`.

For a stable shared URL, `dashboard_server.py` discovers dashboards beneath the configured runs root. It only considers a Human
QA dashboard after Step 8 has a complete stage marker, selects the artifact with the latest stage completion time, and falls
back to an explicitly configured original file when no completed run contains one. The server reads run-scoped immutable
artifacts directly; it does not copy, overwrite, or delete current or historical dashboards.

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
