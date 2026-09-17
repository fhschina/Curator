# Dedup Eval v0.7.1

Dedup Eval is a source-checkout evaluation tool for NeMo Curator fuzzy
deduplication. Tool release `v0.7.1` preserves the immutable Judge contract
`v0.7`; the release changes organization and dependency boundaries, not Judge
semantics.

## Commands

```bash
python -m eval.dedup --version
python -m eval.dedup prepare --root RUN_ROOT --source-run SOURCE_RUN
python -m eval.dedup launch --root RUN_ROOT --env-file ENV_FILE
python -m eval.dedup run --root RUN_ROOT --env-file ENV_FILE --session SESSION
python -m eval.dedup status --root RUN_ROOT
python -m eval.dedup audit --root RUN_ROOT
```

Choose `--backend local` during `prepare` to freeze a local Qwen/B200 backend;
otherwise the Hub backend is used. Existing `CURATOR_V07_SOURCE_RUN`,
`CURATOR_V07_LOCAL_MODEL_PATH`, and `CURATOR_V07_LOCAL_TOOLS_DIR` environment
variables remain supported. `python -m eval.dedup.recommended` is a deprecated
one-release alias.

Every generated request, response, result, report, heartbeat, and log is written
under the explicit run root. Commands do not write into `eval/dedup`.

## Layout

- `core/`: contracts, configuration validation, and immutable I/O
- `handoff/`: corpus and system-under-test handoff manifests
- `pair_construction/`: canonical pairs and retrieval
- `judging/`: the selected v0.7 schema, critics, recovery, and relays
- `runtime/`: prepare, Hub/local execution, status, replay, and audit
- `analysis/`: reusable metrics, comparisons, and constraint graphs
- `reporting/`: reports, Pair Explorer, and checksum-bound serving
- `release/`: selected prompt, schema metadata, local config, and smoke panel

Historical experiments and raw runs are intentionally absent. They are retained
in the private archive repository and approved object storage, not in this tree.

## Dependencies

Install the optional `dedup_eval` extra for execution. The tool remains a
source-checkout utility and is not included in the NeMo Curator wheel package.
