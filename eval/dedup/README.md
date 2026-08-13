# NeMo Curator Fuzzy Dedup Evaluation

This package implements the ten-step V0.3 operational proposal over the frozen CC-MAIN-2025-26 10M handoff. It evaluates a completed fuzzy-dedup SUT; it does not rerun exact deduplication or physically remove corpus rows.

The current design source is the [V0.3 operational proposal](docs/proposals/v0.3/NeMo_Curator_Dedup_Evaluation_V0_3_Operational_Proposal.docx). Earlier proposal revisions are retained in the [proposal archive](docs/proposals/archive/), and the technical presentation is under [docs/presentations](docs/presentations/).

## Runtime profiles

- `smoke`: all 10,008,061 handoff documents, 20 anchors, 50 removal decisions, up to 50 cross-group pairs, and at most 100 judge requests. Its report is explicitly non-V0.
- `full`: 1,000 anchors, 10,000 removal decisions, up to 10,000 cross-group pairs, and the required 200-pair human-QA gate.

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

For a full run, complete the frozen `data/human_qa_labels.csv` template outside the immutable bundle, import it; `qa-import` validates the labels and resumes Step 10 automatically:
Run creation also freezes the `eval/dedup` source-tree digest. `run` and `resume` stop with `RESUME_SOURCE_MISMATCH` if the implementation changes, preventing mixed-code artifacts under one evaluation run ID.


```bash
/raid/hfang/dedup_eval_env/bin/python -m eval.dedup qa-import --run-root <v0_run> --labels <completed_labels.csv>
```

## Module and artifact map

- `handoff/`: validates and wraps Steps 1–2 artifacts.
- `pair_construction/`: writes document outcomes, anchors, the two pair tracks, canonical queue, and provenance.
- `judging/`: constructs blind payloads, handles long documents, calls a provider, validates the schema, retries, and resumes.
- `analysis/`: creates the partial graph, pair comparison, and frame-valid metrics.
- `run.py`: stage transactions and resume validation; domain logic remains in the packages above.
- `report.py`: blind human-QA exchange and the bounded-claim report.

All generated data is written beneath the configured external run root. Imports are side-effect free.
