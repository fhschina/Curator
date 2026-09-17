# NeMo Curator Fuzzy Dedup Evaluation

> **Current internal release:** v0.7.1 on
> [`fhschina/Curator:dedup-eval`](https://github.com/fhschina/Curator/tree/dedup-eval).
> This is not an NVIDIA-NeMo/Curator upstream release.

This package evaluates the decisions produced by a completed fuzzy-deduplication
run. It applies a schema-constrained LLM Judge to a frozen population of
keeper-to-removed and cross-group pairs, records every request and response, and
replays the saved calls offline before declaring a run complete.

It does **not** rerun exact or fuzzy deduplication, change the source corpus, or
physically remove documents. Use it when dedup outputs already exist and you want
to measure whether the pair-level decisions are supported by document evidence.

## Current release

Use the stable Dedup Eval command for new runs:

```bash
python -m eval.dedup --version
```

The command prints the installed tool version. Each run manifest records that
version together with Judge contract `v0.7`, the source digest, model settings,
and backend configuration.

The same Judge contract can run against either:

- **NVIDIA Inference Hub**, the default backend; or
- **a local Qwen FP8 deployment**, with one tensor-parallel-1 replica per
  selected B200.

Backend selection is frozen when a run is prepared. Hub and local evaluations
must use different fresh run roots.

## Prerequisites

Start with the normal NeMo Curator source-checkout requirements. Dedup Eval also
needs the following evaluation-specific resources.

### Required for both backends

- Python 3.11 and `uv >= 0.12.0`; use the checked-in lockfile.
- The `dedup_eval` optional dependency set. It includes the CUDA 12 SDG, Data
  Designer, and Curator inference-server dependencies required by the Judge.
- A readable copy of the frozen v0.7 20K input bundle. It contains
  `manifest.json`, `complete.json`, `panel_index.json`, `inputs/`, and
  `main_requests/`.
- A new writable run directory with enough space for the selected population,
  requests, responses, results, recovery logs, and the final audit.

The input bundle contains internal evaluation data and is intentionally not
committed to Git. Obtain it from the project owner or approved private storage.
Dedup Eval resolves files relative to the supplied bundle and verifies their
content hashes. The fixed 24-pair smoke panel ships with the tool, so no prior
result bundle is needed.

Configure the input location once per shell:

```bash
export CURATOR_V07_SOURCE_RUN=/path/to/frozen-v07-inputs
```

The equivalent `prepare` option is `--source-run`. An explicit option takes
precedence over the environment variable.

### Additional requirements for NVIDIA Inference Hub

- Network access to `https://inference-api.nvidia.com/v1`.
- A valid `NVIDIA_API_KEY` in a private local environment file. Do not commit
  that file.

Example:

```dotenv
NVIDIA_API_KEY=replace_with_your_key
```

### Additional requirements for the local backend

- Linux, a CUDA 12-compatible NVIDIA driver, and one or more NVIDIA B200 GPUs.
- The pinned `Qwen/Qwen3.8-27B-FP8` checkpoint at revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. The validated model bundle
  includes `.hosting-benchmark-revision.json`; preparation rejects a different
  revision or a non-FP8 checkpoint.
- `etcd` and `nats-server` executables in one directory.
- A short writable local runtime path. The default is derived under `/tmp` to
  stay below Ray's Unix-socket path limit.

Configure machine-local resources without editing repository files:

```bash
export CURATOR_V07_LOCAL_MODEL_PATH=/path/to/Qwen3.8-27B-FP8
export CURATOR_V07_LOCAL_TOOLS_DIR=/path/to/directory-containing-etcd-and-nats-server
```

The equivalent options are `--local-model-path` and `--local-tools-dir`.

## Installation

Clone the internal branch and create an environment on a suitable machine:

```bash
git clone --branch dedup-eval https://github.com/fhschina/Curator.git
cd Curator

uv sync --locked --python 3.11 --extra dedup_eval
source .venv/bin/activate

python -m eval.dedup --version
```

Do not reuse an environment resolved from unconstrained package versions. The
lockfile preserves the validated Data Designer, Ray, Dynamo, vLLM, PyArrow, and
RAPIDS compatibility set.

## Quick start

Run all commands from the repository root. `prepare` copies and verifies the
frozen Judge inputs; it does not rerun deduplication or reuse old model answers.

The recommended release flow prepares the complete 20K root, evaluates only the
24-pair smoke gate first, and then resumes the same immutable root over the
remaining population.

### Option A: NVIDIA Inference Hub

Choose a new run root and private credential file:

```bash
export V07_RUN_ROOT=/path/to/runs/v071-hub-001
export V07_ENV_FILE=/path/to/private/nvidia.env

python -m eval.dedup prepare \
  --root "$V07_RUN_ROOT"

python -m eval.dedup launch \
  --root "$V07_RUN_ROOT" \
  --env-file "$V07_ENV_FILE" \
  --smoke-only

python -m eval.dedup status --root "$V07_RUN_ROOT"
```

After status reports a passed smoke gate, continue the same immutable run over
the remaining population:

```bash
python -m eval.dedup launch \
  --root "$V07_RUN_ROOT" \
  --env-file "$V07_ENV_FILE"
```

### Option B: local model on one B200

GPU 0 with one local replica is the explicit single-GPU configuration:

```bash
export V07_RUN_ROOT=/path/to/runs/v071-local-1gpu-001

python -m eval.dedup prepare \
  --root "$V07_RUN_ROOT" \
  --backend local \
  --local-device 0

python -m eval.dedup launch \
  --root "$V07_RUN_ROOT" \
  --smoke-only

python -m eval.dedup status --root "$V07_RUN_ROOT"
```

After `SMOKE_PASSED`, continue without changing the run configuration:

```bash
python -m eval.dedup launch --root "$V07_RUN_ROOT"
```

The local backend does not read or persist an NVIDIA API key.

### Option C: local model on multiple B200s

Use a comma-separated device list and the same number of independent replicas:

```bash
export V07_RUN_ROOT=/path/to/runs/v071-local-2gpu-001

python -m eval.dedup prepare \
  --root "$V07_RUN_ROOT" \
  --backend local \
  --local-devices 0,1 \
  --local-replicas 2

python -m eval.dedup launch \
  --root "$V07_RUN_ROOT" \
  --smoke-only
```

Each B200 hosts one tensor-parallel-1 replica. Aggregate client concurrency is
eight requests per replica. `--local-replicas` must equal the number of unique
devices in `--local-devices`; the runner does not split one model across GPUs.

### Compact smoke-only root

For a quick acceptance test that does not copy all 20,000 input files, add
`--smoke-only` during both preparation and execution:

```bash
export V07_SMOKE_ROOT=/path/to/runs/v071-smoke-001

python -m eval.dedup prepare \
  --root "$V07_SMOKE_ROOT" \
  --smoke-only

python -m eval.dedup launch \
  --root "$V07_SMOKE_ROOT" \
  --env-file "$V07_ENV_FILE" \
  --smoke-only
```

This root contains only the 24 smoke pairs and intentionally cannot be resumed
as a full 20K run. Omit `--smoke-only` from `prepare` when you intend to continue
after the smoke gate.

### Monitor and audit a run

`launch` starts a detached process and records its session log below the run
root. These commands are safe to rerun:

```bash
python -m eval.dedup status --root "$V07_RUN_ROOT"
python -m eval.dedup audit --root "$V07_RUN_ROOT"
```

Run `audit` after full completion. It verifies the frozen manifest, accounts for
all outputs and transport attempts, and replays every saved model call without
making new requests. Existing v0.7 run roots can also be audited read-only; the
tool does not rewrite their manifests or completion markers.

A completed run is immutable. Use a new root for another backend or
configuration.

## Current results

The validated v0.7 run completed 20,000/20,000 schema-valid outputs and replayed
all 24,105 saved calls identically offline. On the shared 994-pair development
reference:

| Metric | v0.7 contract |
|---|---:|
| Weighted precision | 81.82% |
| Weighted recall | 80.04% |
| Weighted primary-decision agreement | 88.88% |

These are development-reference results, not independent holdout accuracy. See
the [full current results](RESULTS.md) for comparison tables, engineering
accounting, SUT diagnostics, and methodological limits.

## How a run works

```text
Frozen 20K pairs
    -> immutable backend manifest
    -> 24-pair smoke gate
    -> remaining Judge population
    -> saved requests, responses, and decisions
    -> offline replay and completion audit
```

The smoke gate covers the main, coverage, subject, and verifier paths. A
smoke-only launch on a full prepared root can be resumed into the full
population without recalling completed pairs. Backend, model, topology,
generation, concurrency, runtime paths, source digest, and Judge contract are
frozen in `manifest.json` at preparation time.

## Key outputs

All runtime artifacts stay below the user-selected `V07_RUN_ROOT`:

| Path | Purpose |
|---|---|
| `manifest.json` | Immutable tool, contract, backend, input, code, model, and topology binding. |
| `smoke_panel.json` | Frozen 24-pair smoke population. |
| `smoke_complete.json` | Smoke-gate decision and replay accounting. |
| `requests/`, `responses/` | Hash-bound model-call ledger. |
| `results/` | One validated public Judge result per canonical pair. |
| `recovery/sessions/` | Detached launch records, heartbeats, exits, and logs. |
| `complete.json` | Final population accounting and offline replay proof. |
| `provenance/` | Copied source manifests needed to reproduce the input binding. |

Commands do not write runtime files into `eval/dedup`.

## Repository layout

- `core/`: contracts, validation, configuration, and immutable I/O
- `handoff/`: corpus and system-under-test handoff manifests
- `pair_construction/`: canonical pairs and retrieval
- `judging/`: the selected v0.7 schema, critics, recovery, and relays
- `runtime/`: prepare, Hub/local execution, status, replay, and audit
- `analysis/`: reusable metrics, comparisons, and constraint graphs
- `reporting/`: reports, Pair Explorer, and checksum-bound serving
- `release/`: prompts, schema, local config, and smoke panel

## Development

Run the CPU-scoped suite before submitting changes:

```bash
pytest -q -m "not gpu" \
  tests/eval/dedup \
  tests/eval/llm_judge \
  tests/stages/synthetic/nemo_data_designer/test_data_designer.py

pre-commit run --all-files
```

Use Conventional Commits. Changes to prompts, schemas, adapters, source
bindings, model revision, generation settings, or retry behavior create a new
Judge contract and must not silently reuse an existing run root.

## Scope and limitations

- This is an internal fork tool release, not an upstream
  NVIDIA-NeMo/Curator production release.
- The LLM Judge is an automated development reference, not independent human
  ground truth.
- Git contains the runner and current summary, but not the private frozen 20K
  data, model weights, credentials, raw runs, or system binaries.
- A successful checkout does not grant data, model, GPU, API, or internal
  service access; those resources must be arranged separately.
