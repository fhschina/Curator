# Qwen retention calibration v2, round 022

**The task acceptance conditions were not met.** This is the best available
frozen version after six semantic candidates, delivered with all failed checks.
D121 remains pending. The fixed baseline is
`6715574e4e553f7b36f263d755f6cebe198e7b11`; the selected variant is `candidate-02`.

candidate-02 was selected from the frozen alternatives because it preserved complete technical validity and zero observed reference-required wrong allows at every layer across all 258 hard presentations. Among variants meeting those two conditions, it had the greatest correct directional coverage. Other required-Yes mistakes still fail the hard gate; no version is promoted by majority vote. The ranking uses hard evidence only; development-reference improvement cannot excuse hard failures.

Baseline and selected variant each completed all 150 discussion inputs and all
100 development inputs three times. Invalid and unresolved judgments remain
incorrect in their fixed eligible denominators. These are exposed development
references, not independent human accuracy or corpus deletion-risk estimates.
Allocated GPU use was 3.58 B200 hours on one GPU;
the owned service was stopped. The six-candidate search limit was reached; the
ten-hour wall-clock maximum was not exhausted.

| Version | Reference | Final direction mean | Direction worst | Final pair mean | Pair worst |
| --- | --- | --- | --- | --- | --- |
| baseline | personal120 | 75.42% | 71.67% | 70.00% | 65.83% |
| baseline | development99_original | 94.61% | 93.94% | 93.60% | 92.93% |
| baseline | development99_reviewed | 95.62% | 94.95% | 94.61% | 93.94% |
| baseline | development97_strict | 95.88% | 95.88% | 94.85% | 94.85% |
| candidate-02 | personal120 | 80.28% | 79.58% | 75.83% | 75.83% |
| candidate-02 | development99_original | 94.78% | 94.44% | 93.94% | 93.94% |
| candidate-02 | development99_reviewed | 95.79% | 95.45% | 94.95% | 94.95% |
| candidate-02 | development97_strict | 95.70% | 95.36% | 94.85% | 94.85% |

The personal denominator is 120 pairs / 240 directions. The 100-input panel has
99 complete original references (H039 has an unresolved reference direction),
99 reviewed references accepting H029/H052 whole-pair alternatives, and 97 strict
reviewed references excluding those two ambiguous cases. All 100 inputs remain
in technical-validity counts. The reports distinguish the mean 80%/75% target
from the additional each-repeat stability check. Every hard-regression layer and
repetition must pass; no majority-vote acceptance is used.

## Reports and evidence

- [English report](../../../../../fern/versions/main/pages/curate-text/process-data/deduplication/retention-policy-v2-report-en.mdx)
  and [direct Chinese translation](../../../../../fern/versions/main/pages/curate-text/process-data/deduplication/retention-policy-v2-report-zh.mdx).
- Standalone [English HTML](report_en.html) and [Chinese HTML](report_zh.html).
- [Machine summary](summary.json), [per-run comparison CSV](comparison.csv),
  [machine HTML](comparison.html), and [shared report tables](report_tables.json).
- [Frozen manifest and hashes](manifest.json), [run provenance](run-provenance.json),
  [evaluation protocol](frozen-evaluation-protocol.json),
  [selection](final-selection.json), [engineering changes](engineering-changes.json),
  [verification](verification-final.json), and
  [translation verification](translation_validation.json).
- The ZIP archive(s) in this directory contain all six frozen prompt versions,
  the separate candidate-5 engineering repair, per-run JSON/CSV/HTML, original
  run manifests, review provenance, input indexes, and reproduction tools.
  The full comparison JSON is under `delivery/comparison.json` in the archive.

Full immutable input texts, native Data Designer outputs, HTTP attempt traces,
source snapshots and checkpoints remain at:

```text
/raid/hfang/retention_policy_eval/v1/experiments/round-022-v2-20261027T214234Z
```

Per-row CSVs preserve historical reference columns even for unscored rows;
scoring eligibility is defined by the frozen evaluation lists and aggregate
groups, not by the mere presence of a historical reference.

Raw corpus text and HTTP bodies are not committed. No new 20K or Deciffusion
inference was run. Their frozen historical input hashes do not indicate new
inference. The original round-021 reports and branch remain intact.

## Recompute the comparison without inference

Use the existing environment and this checkout. Choose a new output directory;
the evaluator refuses to overwrite an existing comparison or inference run.

```bash
cd /raid/hfang/workspaces/Curator-retention-policy-v2
export PATH=/raid/hfang/llm_judge_env_pr2324_latest/bin:$PATH
export PYTHONPATH=/raid/hfang/workspaces/Curator-retention-policy-v2/tutorials/eval/dedup:/raid/hfang/workspaces/Curator-retention-policy-v2
python tutorials/eval/dedup/policy_tuning.py compare \
  --session /raid/hfang/retention_policy_eval/v1/experiments/round-022-v2-20261027T214234Z \
  --best 2 --best-root candidate-02 --output NEW_UNIQUE_COMPARISON
```

To rebuild both human reports from that comparison, extract the ZIP archive(s)
into one new directory and run its shared renderer, supplying the original
session explicitly:

```bash
python EXTRACTED/tools/render_bilingual.py \
  /raid/hfang/retention_policy_eval/v1/experiments/round-022-v2-20261027T214234Z/NEW_UNIQUE_COMPARISON/comparison.json \
  NEW_UNIQUE_HUMAN_REPORT_DIRECTORY \
  /raid/hfang/retention_policy_eval/v1/experiments/round-022-v2-20261027T214234Z
```

The renderer checks common table data, section order and numeric equality across
languages. It requires three completed baseline and selected-variant diagnostic
and development repetitions, plus the frozen selection and verification receipts.

## Replay inference with frozen prompts

First provide one caller-owned GPU0 service using the exact profile in
[service.json](service.json): the pinned image and Qwen revision, FP8, TP=1,
context 32768, max sequences 8, memory fraction 0.8, and the read-only checkpoint
mount. Use a unique container name, loopback port and cache directory, and an
absolute shutdown deadline. The recorded supervisor and deadline-guard sources
are included as evidence; their original names and expired deadline must be
adapted for a new replay. The native evaluator verifies the live service profile.

With that service ready, run the archived replay script in the existing
`/raid/hfang/llm_judge_env_pr2324_latest` environment:

```bash
python EXTRACTED/tools/replay_frozen.py \
  --source-session /raid/hfang/retention_policy_eval/v1/experiments/round-022-v2-20261027T214234Z \
  --destination /absolute/path/to/NEW_UNIQUE_REPLAY \
  --checkout /raid/hfang/workspaces/Curator-retention-policy-v2 \
  --endpoint http://127.0.0.1:NEW_PORT/v1 --container NEW_OWNED_CONTAINER \
  --hours 8 --repeats 1 2 3
```

This creates fresh run directories and checkpoints, links/copies only immutable
frozen inputs, and calls `policy_tuning.py execute` followed by the original
`policy_eval.py run`. It performs no prompt search and does not create or sync an
environment. Native parameters remain max_tokens=6144, temperature=0.2,
concurrency=8, timeout=600, thinking disabled, and the fixed 6000-character window.
Keep the caller-owned service's lifecycle receipt and stop it when replay ends.
Recompute its comparison using the command above with the new session path.

## Validation

CPU tests cover fixed denominators, coherent accepted pairs, retained failures,
raw unsafe signals, actual template compilation, evidence/window rendering, the
native Data Designer stage and workflow. Registered GPU checks exercise actual
structured generation/SkipConfig, untrusted document instructions, coverage
proof and subject verification. Only the single-GPU checks were run; the
registered eight-replica check was excluded. Exact results and log hashes are in
`verification-final.json`; final hook evidence accompanies the delivery.
