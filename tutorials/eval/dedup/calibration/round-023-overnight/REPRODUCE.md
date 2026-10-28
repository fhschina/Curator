# Reproduce the frozen overnight comparison

The evaluator uses the existing `/raid/hfang/llm_judge_env_pr2324_latest`
environment and native Data Designer 0.9.1. Do not create or synchronize another
environment. The source session and its raw inputs remain outside Git:

`/raid/hfang/retention_policy_eval/v1/experiments/round-023-overnight-20261028T025929Z`

`replay_frozen_v3.py` verifies the frozen files and native source hashes, copies
inputs independently, declares three run identities per panel, and calls the
existing `policy_tuning.py execute` → `policy_eval.py run` chain. It then retains
the native comparison and applies the shared, read-only engineering 04 ledger
audit. It does not alter generated decisions, references or denominators. Every
failed or unresolved judgment retains its fixed denominator and earns no credit.

The replay has three versions: baseline `6715574e`, starting `39714b47`, and the
single selected frozen candidate. Each version runs gate, coverage-regression,
preflight, diagnostic and holdout panels three times. The source session's
`final-selection.json` supplies the selected root by default; during tuning,
`--best-root` explicitly selects an already frozen candidate. Failed run names
cannot be replaced within the same root.

## Service and deadline prerequisites

Supply one caller-owned local service and install its independent absolute
shutdown guard **before allocating the GPU**. The replay entry does not start or
stop that service. Preserve this measured profile:

| Setting | Value |
| --- | --- |
| Hardware | One NVIDIA B200, TP=1 |
| Model | Qwen/Qwen3.8-27B-FP8 |
| Revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Image | `ghcr.io/pst2154/nemotron-deciffusion@sha256:494aac2d7af43b7a70234792623a510bf29045a45b3aa893bc3170108ded9cea` |
| Context / output | 32768 / 6144 tokens |
| Temperature / concurrency / timeout | 0.2 / 8 / 600 seconds |
| Thinking / visible window | Disabled / 6000 characters |
| Service limits | `max_num_seqs=8`, GPU memory utilization 0.8, prefix caching |

The original service command, model/image verification, process registry and
tested task-specific guard remain in the external session. The guard's cleanup
is limited to exact task-owned process identities, recorded Ray temporary paths
and the owned container label. A future replay must use its own ownership
identity and explicit future deadline; the original deadline never moves.

Choose a new absolute UTC delivery deadline ending in `Z`. The replay's GPU
cutoff is fifteen minutes earlier. Allow enough time for all 45 panel runs plus
cleanup. Its admission check refuses late work rather than extending the cutoff.
Keep the destination in an external experiment directory so raw text, complete
model exchanges and HTTP bodies stay outside Git.

## Entry point

From the checkout, run the existing environment's Python with:

```text
python tutorials/eval/dedup/calibration/round-023-overnight/replay_frozen_v3.py
  --source-session /absolute/frozen-session
  --destination /absolute/new-experiment-directory
  --checkout /absolute/Curator-checkout
  --endpoint http://127.0.0.1:PORT/v1
  --container YOUR_OWNED_CONTAINER
  --deadline-utc YOUR_EXPLICIT_FUTURE_UTC_TIMESTAMP
  --audit-tools tutorials/eval/dedup/calibration/round-023-overnight/engineering-04-ledger-completeness
```

The arguments above are one invocation; join the lines or use shell continuation
characters. The source session must contain the frozen raw panels, labels,
manifests and the selected root, not just the public prompt archives.

Use `--plan-only` to verify and print the plan without creating a destination.
Use `--prepare-only` to independently copy and verify all frozen inputs and
declare the run group without any model call. These modes do not establish an
evaluation result or acceptance. The checked-in preparation receipt covers 117
files and 45 declared slots with zero model calls; full inference replay is not
claimed by that check.

## Acceptance and evidence

The final comparison reports personal120, original99, reviewed99 and strict97,
each repetition, means and worst results, main/coverage/final snapshots, technical
failures and raw wrong-allow/block signals. The exact final thresholds are
648/720 directions and 306/360 pairs; the milestone is 612/720 and 288/360. The
same frozen candidate must satisfy both rates. Every required hard presentation
must pass in every layer and repetition, and each development directional mean
must meet the fixed baseline. D121 stays pending.

The engineering 03 artifacts and the complete engineering 04 audit remain
separate. Current comparison and snapshot pointers identify the applicable audit
hash. A later audit or failed candidate does not erase earlier raw evidence.
Public `frozen-candidates` archives contain prompt/configuration files and
provenance; raw corpora and full HTTP exchanges remain external. The paired Fern
reports share their data tables and check numeric and section consistency.

These are exposed development references, not independent human accuracy or a
measured corpus deletion rate. Independent human acceptance, a new 20K run and
Deciffusion comparison are follow-up work.
