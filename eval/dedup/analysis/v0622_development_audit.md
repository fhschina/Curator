# V0.6.2.2 LLM Judge development audit

Date: 2026-09-09

## Decision

`dedup-judge-hs-v0.6.2.2` does **not** advance to the frozen holdout. Its semantic ledger fixes enough false containment
decisions to pass the weighted precision, primary-decision, and over-group gates, but it rejects too many human-labeled true
containments. Weighted recall is 63.67%, below the 75% gate, and the weighted meaningful-addition cohort accuracy regresses
by 54.77 percentage points from the equal-contract development baseline.

The run itself is operationally valid: 1,000/1,000 rows completed, there were no terminal errors, schema completion was
100%, and 8 rows required a Judge retry (0.8%). Its pair membership and visible-payload digests exactly match the other
development variants.

## Equal-contract development comparison

All rows below use the same 1,000 human-labeled pairs, model, temperature, sampling parameters, and visible payloads.

| Variant | Weighted precision | Weighted recall | Weighted primary exact | Over-group | Containment over-group | Under-group | Development gates |
|---|---:|---:|---:|---:|---:|---:|---|
| V0.6.2 dev baseline | 63.60% | 77.39% | 75.65% | 120 | 111 | 63 | Baseline only |
| V0.6.2 | 65.30% | 82.74% | 78.74% | 122 | 95 | 49 | Failed |
| V0.6.2.1 | 67.10% | 77.94% | 78.69% | 106 | 95 | 58 | Failed |
| V0.6.2.2 | **76.06%** | **63.67%** | **79.61%** | **59** | **27** | **95** | **Failed** |

V0.6.2.2 reduces over-group by 50.8% relative to the equal-contract baseline and by 44.3% relative to V0.6.2.1. That gain
comes with 77 missed human-labeled containments, so it is not a viable precision/recall trade by the frozen selection rule.

## Gate result

| Gate | Threshold | V0.6.2.2 | Result |
|---|---:|---:|---|
| Weighted duplicate precision | at least 75% | 76.06% | Pass |
| Weighted duplicate recall | at least 75% | 63.67% | **Fail** |
| Weighted primary exact | at least 79% | 79.61% | Pass |
| Over-group | at most 66 | 59 | Pass |
| Identity/slot regression | no worse than -3 pp | +0.41 pp | Pass |
| Meaningful-addition regression | no worse than -3 pp | **-54.77 pp** | **Fail** |
| Translation regression | no worse than -3 pp | +3.20 pp | Pass |
| Schema completion | 100% | 100% | Pass |
| Judge retry rate | at most 1% | 0.8% | Pass |

Unweighted precision, recall, and primary exact are respectively 79.58%, 70.77%, and 79.60%. These are reported separately
and do not replace the weighted development gates.

## Semantic-ledger findings

- `HARD_CONFLICT_VETO` handled 601 rows and reached 91.84% weighted primary accuracy. It produced no over-group cases, but
  accounts for 50 under-group cases.
- `NONEMPTY_SUBSTANTIVE_CONTAINMENT` handled 64 rows and reached only 55.12% weighted primary accuracy. It accounts for all
  27 remaining containment over-group cases.
- `SUBSTANTIVE_NON_MAIN_PROFILE_MISMATCH`, `NON_MAIN_MESSAGE_NOT_EQUIVALENT`, and `NO_SHARED_SUBSTANTIVE_ANCHOR` account for
  40 under-group cases. Thirty-five of these are old human `meaningful_addition`/`CONTAINMENT` labels and are explicit
  policy-review candidates.
- The meaningful-addition cohort falls from 75.58% to 20.81% weighted primary exact. This is the dominant release blocker;
  identity/slot and translation are not the source of the aggregate failure.

## Required review before another prompt version

[`v0622_policy_review_candidates.csv`](v0622_policy_review_candidates.csv) contains all 204 primary-tuple disagreements,
with the old label, V0.6.2.1 result, V0.6.2.2 result, semantic-ledger values, exact evidence, visible text excerpts, and blank
human disposition fields. [`v0622_policy_review_summary.json`](v0622_policy_review_summary.json) records the reproducible
counts.

The highest-priority queues are:

| Queue | Rows | Weighted rows | Why it must be reviewed |
|---|---:|---:|---|
| Potential policy-label conflict | 35 | 369.32 | Old containment labels may rely only on boilerplate/non-main overlap, which V0.6.2 explicitly forbids. |
| Potential model classification error | 50 | 549.49 | The model may be inventing identity/state/role conflicts or two-sided divergence. |
| Remaining over-containment | 27 | 257.54 | The same-record/non-empty gate still accepts false one-sided additions. |
| Translation completeness review | 5 | 29.89 | Reviewers must distinguish faithful complete translation from genuine added/omitted facts. |

These are candidates for human review, not proven annotation errors. No development label is changed automatically. The next
candidate should be designed only after this queue is adjudicated: improve content-profile and same-record classification
without weakening the non-empty-containment rule. The 400-pair holdout remains untouched until a candidate passes every
development gate.
