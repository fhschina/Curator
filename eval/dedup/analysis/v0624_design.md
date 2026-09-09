# V0.6.2.4 verified-identity and surface-delta design

> Final status: rejected on the 127-row residual smoke. The immutable run returned 126/127 valid rows, retried 7/127,
> reduced over-group from 46 to 25 and containment over-group from 22 to zero, but increased under-group from five to 21.
> See `v0624_residual_smoke.md` and `v0624_residual_smoke_summary.json`. No full-development or holdout run is authorized.

## Decision context

V0.6.2.3 is preserved as a rejected immutable experiment. Its 127-row residual smoke was originally evaluated before the
newly exposed labels were reviewed. The follow-up review is diagnostic rather than blind because V0.6.2.3 predictions were
already visible.

The diagnostic review made two policy corrections without changing the semantic contract:

- all 15 newly surfaced containment misses and all five apparent translation misses are `NO/NO`; the non-main messages
  differ materially, or the translations change identity, quantities, claims, or page purpose;
- three additional old `YES/YES` labels are `NO/NO`: `H0807` adds preference-based advertising, `H0802` presents different
  cookie/GDPR/ad-blocker messages, and `H0921` adds a multi-sentence forum description to an authentication notice.

The original 85-row review, the 20-row diagnostic follow-up, and the three-row revision remain separate immutable ledgers.
The cumulative development labels are `v0624_policy_reconciled_labels_1000.csv`; they preserve the pre-review fields and
per-row review protocol.

## Reconciled baseline

Against the cumulative labels, the frozen V0.6.2.2 full 1,000-pair result has:

| Metric | Weighted | Unweighted |
|---|---:|---:|
| Duplicate precision | 69.49% | 74.39% |
| Duplicate recall | 95.01% | 95.98% |
| Primary-decision exact | 88.67% | 88.10% |
| Taxonomy exact | 65.00% | 63.20% |

It has 74 over-group errors, including 27 containment over-group errors, and nine under-group errors.

On the same 127 residual IDs, V0.6.2.3 improves over-group from 74 to 46 and containment over-group from 27 to 22 while
reducing under-group from nine to five. Its weighted precision is 44.42%, recall 87.74%, and primary-decision exact 33.58%
on this deliberately error-enriched slice. The 46 over-group errors comprise 24 near-surface equivalences and 22
containments.

The five remaining true duplicate misses are `H0038`, `H0347`, `H0453`, `H0679`, and `H0748`. They are narrow benign
surface cases: redundant placeholder repetition; wishlist/learn-more/accept labels around the same notice; a `Payment`
heading; and a stories/byline header around the same membership pitch.

## Immutable candidate

V0.6.2.4 uses prompt version `dedup-judge-hs-v0.6.2.4` with the unchanged published
`dedup-judge-output-v3` schema. It copies V0.6.2.3 and adds three diagnostic semantic-ledger fields:

1. `record_identity_support` requires either a verbatim shared identifier, at least two distinctive facts per side, a
   complete matching non-main message, or an explicit negative/unresolved classification.
2. `overlap_scope` separates document-wide same-record evidence from a complete non-main message, a local passage, or a
   generic family/template.
3. `surface_delta_type` separates formatting, short navigation/byline labels, and redundant repetition from any material
   proposition or record-bearing delta.

Positive substantive identity is locally verifiable. `VERBATIM_SHARED_IDENTIFIER` requires matching normalized exact A/B
quotes in that field's reasoning; `DISTINCTIVE_MULTI_FACT_IDENTITY` requires at least two aligned exact quotes from each
side. General quote evidence cannot satisfy this field-specific contract.

The adapter then applies these gates before the V0.6.2.3 resolution order:

- local-passage or generic-family/template overlap is always `NO/NO`;
- a material proposition or record delta is always `NO/NO`;
- substantive equivalence or containment requires verified identity and document-wide same-record scope;
- non-main equivalence requires the same complete message plus a benign surface delta;
- isolated navigation/category/byline/button labels and redundant repetition may remain `YES/YES`;
- unresolved new gates produce the fully unresolved, low-confidence result.

The system prompt, pair prompt, YAML rubric, adapter, reason-code reporting, policy-review routing, calibration reporting,
version registry, comparison runner, and one-to-one tests are updated together. V0.6.2.3 resources and results are not
modified or reused as V0.6.2.4 cache entries.

## Admission sequence

### Residual smoke

Run the immutable candidate on the same 127 residual IDs and score it against the cumulative labels. Advance only if every
check passes:

- total over-group is at most 32, a reduction of at least 30% from V0.6.2.3's 46;
- containment over-group is at most 15, a reduction of at least 30% from 22;
- under-group is at most five and all five named benign duplicates are retained;
- all 23 labels changed in the diagnostic follow-up and preflight revision remain `NO`;
- schema completion is 100%, terminal errors are zero, and retry rate is at most 2% on the small smoke batch.

Failure freezes V0.6.2.4 as rejected and prevents a full development or holdout run.

### Full reconciled development set

If the smoke passes, run all 1,000 cumulative development labels with identical model, decoding, and payload settings. The
candidate may advance only if:

- weighted duplicate precision and recall are each at least 75%;
- weighted primary-decision exact is at least 87%, while remaining above the original 79% release target;
- total over-group is at most 50 and containment over-group is at most 18;
- identity, meaningful-addition, and translation cohort accuracy do not regress by more than three percentage points from
  V0.6.2.2;
- schema completion is 100%, terminal errors are zero, and retry rate is at most 1%.

Only an all-gates pass may freeze and expose a new 400-pair holdout. The holdout remains unseen throughout the residual and
full-development decisions.
