# V0.6.2.29 implementation checkpoint — online diagnosis not started

Date: 2026-09-10 UTC. Status: opt-in experimental candidate, **not a release**.

The [approved composite-containment policy](composite_containment_policy_v1.md) is implemented by
[the new V4 coverage adapter](../judging/composite_coverage.py),
[fresh-main/critic runtime](../judging/composite_runtime.py), and
[component diagnostic runner](composite_experiment.py). See the [frozen design](v06229_design.md)
and [versioned configuration](../resources/local_ndd/hs_v06229_composite_qwen_c16.yaml).

## Implemented boundary

- Substantive X + independent Y can replace X in one direction. A same-record extension is no longer required.
- Nonempty shared substantive X, full directional retention, and actual paired conflict evidence remain required.
  Independent Y's title or identity is not itself an identity conflict in X. Template-only overlap is not sufficient.
- Main and critic share the same policy and strict `dedup-retained-coverage-v4` response contract, exporting public v3.
  Main is generated afresh; no historical main verdict or cache is reused. Critic only reviews this main's positives.
- Raw responses, transported values, original-text evidence, input hashes, prompts and accepted attempts are bound and
  replayed separately. Legacy same-record adapters do not decide the new verdicts.
- Main-only and main-plus-critic results will be compared on the same main outputs. This is not a policy-only ablation:
  both the main implementation and internal response contract differ from historical `.12`.

## Completed offline verification

The [boundary report](v06229_boundary.json) hashes the implementation, prompts, tests and test XML artifacts.

| Check | Result |
| --- | --- |
| CPU evaluation suite | 900 passed, 34 skipped, 2 GPU tests deselected |
| New native Ray → local HTTP → NDD → writer → binder integration | 4 passed: main/critic × first/feedback |
| Local HTTP responses in the final native boundary run | 14 successful responses |
| External model calls this iteration | **0** |
| Frozen 50-pair main input budget | All passed; 46 model-routed, 4 deterministic input branches |
| Largest rendered main prompt | 19,794 tokens; 25,938 including output allowance and safety margin, below 32,768 |
| Historical `.27` and `.28` source/input/config freezes | Validated unchanged |
| Ruff checks, formatting and whitespace checks | Passed for the new Python implementation/tests |

The four integration cases use scripted local HTTP responses, not model predictions. They establish engineering
compatibility, not semantic accuracy. The 34 CPU skips include the four new native cases, which were run separately;
the other historical optional native cases were not rerun. Critic input length depends on the new main response and
is checked before each actual critic request; only main prompt sizes can be checked before fresh inference.

## Frozen diagnostic, awaiting credentials

Run root: `/raid/hfang/ihb/runs/v0.6.2.29-composite-diagnostic`.

Contract digest: `67376d0367224af13927ab29ee2f71fde758f7495560ecb9a308ce96070cb788`.

The manifest freezes 171 source/reference assets plus the diagnostic inputs. The schedule is:

1. Eight synthetic preflight cases. All eight main and final primary decisions must be correct with zero corrections.
2. Only after preflight passes, two fresh repeats of 50 pairs: 24 real development cases and 26 synthetic policy cases.
3. Report the two cohorts separately, including same-main critic gains/losses and all retries. Retain both repeats.

The real panel retains original labels and sampling weights. H0521/H0822 are included as applications of the approved
policy, not newly independent human annotations. The other 17 disputed boundary cases are not automatically relabeled.
All old reference labels, results, prompts, adapters and caches remain unchanged. No historical score was recomputed
against a selectively revised reference.

At this checkpoint, `NVIDIA_API_KEY` is not configured in the execution environment. No `started.json` or online
cell exists. The owned local test Ray cluster is shut down after testing; a dedicated live `RAY_ADDRESS` is also
required when resuming. Configure credentials in the execution environment, not in source files or chat, then run:

```bash
python -m eval.dedup.analysis.composite_experiment run \
  --root /raid/hfang/ihb/runs/v0.6.2.29-composite-diagnostic
```

Use the existing installed environment. The runner rechecks all frozen files and refuses to restart an observed run.
Do not repeat `prepare` against this already-frozen root. If preflight fails, retain the failure and stop before the
formal panel; do not edit the frozen candidate and continue under the same digest.

There is **no new online precision/recall result or release-gate pass**. This selected development panel and synthetic
suite do not replace the full-1,000 weighted 75% precision / 75% recall / 79% primary agreement thresholds, class
protections, retry gate or independent holdout. No 20,000-pair evaluation, release, commit or push was performed.
