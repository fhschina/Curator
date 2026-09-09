# V0.6.2.5 residual-smoke result

## Outcome

V0.6.2.5 is rejected at the frozen residual-smoke gate and must not advance to the full 1,000-pair development run or the
holdout. The immutable run is `/raid/hfang/ihb/runs/v0.6.2.5-residual-smoke`; the machine-readable result is
`v0625_residual_smoke_summary.json`.

Operationally the candidate succeeded: 127/127 valid, zero terminal errors, and one pair-level retry (0.79%). The preflight
returned 11 successful model responses. The formal run returned 144 successful responses, including internal corrections,
with no HTTP 500 response. Removing V0.6.2.4's second identity-evidence mini-contract fixed its completion and retry failure.

Semantically the candidate did not find the required balance. It restored 34 containment predictions, but 25 grouped human
negatives. It left 21 human duplicates separated, retained only two of five protected benign duplicates, and violated five
of the 23 newly reviewed negative labels.

## Frozen gate result

| Check | V0.6.2.3 | V0.6.2.4 | V0.6.2.5 | Required | Result |
|---|---:|---:|---:|---:|---|
| total over-group | 46 | 25 | 31 | at most 32 | pass |
| containment over-group | 22 | 0 | 25 | at most 15 | fail |
| total under-group | 5 | 21 | 21 | at most 5 | fail |
| protected benign duplicates | 5/5 | 2/5 | 2/5 | 5/5 | fail |
| diagnostic negatives preserved | n/a | 21/23 | 18/23 | 23/23 | fail |
| true containment primary tuple | 1/30 | 0/30 | 6/30 | at least 8/30 | fail |
| schema completion | 100% | 99.21% | 100% | 100% | pass |
| terminal errors | 0 | 1 | 0 | 0 | pass |
| retry rate | 1.57% | 5.51% | 0.79% | at most 2% | pass |

The six exactly directed true containments are `H0017`, `H0432`, `H0612`, `H0723`, `H0828`, and `H0906`. The protected
duplicates retained are `H0347` and `H0679`; `H0038`, `H0453`, and `H0748` were split. The reviewed negative violations are
`H0053`, `H0716`, `H0721`, `H0835`, and `H0921`.

On this deliberately error-enriched slice, unweighted duplicate precision is 39.22%, recall 48.78%, and primary-tuple exact
agreement 48.82%. Weighted precision is 39.65%, recall 46.81%, and primary-tuple exact agreement 50.59%. These are diagnostic
slice metrics, not full-population estimates.

## Diagnosis

The added class made the failure measurable but did not make the model's same-record claim reliable:

- `SAME_RECORD_CONTENT_EXTENSION` was selected for 35 rows. It generated all 25 containment over-groups and only a small
  number of exact true containments. In many cases the shorter text is literally contained in the longer text, but the
  uncovered title, product/category, profile, page target, policy section, forum description, date, or state changes which
  record or role the page represents. The model still treats textual superset as semantic same-record superset.
- `MATERIAL_NON_MAIN_MESSAGE_CHANGE` was selected for 32 rows and created no over-group, but rejected 14 human positives.
  This exposes a remaining policy-label boundary: several legacy labels permit containment or equivalence between non-main
  messages that the reconciled prompt treats as materially different. Those rows need independent policy review before
  they are used as recall pressure.
- `RECORD_IDENTITY_ROLE_OR_STATE_CHANGE` was much healthier: 34 rows, no over-group, four under-group, and 85.3% primary
  exact. `TWO_SIDED_CONTENT_CHANGE` was similarly useful at 85.7% primary exact.
- `UNIVERSAL_UI_OR_REDUNDANT_REPETITION` still confuses record-bearing short labels with universal controls: 17 rows yielded
  four over-groups and two under-groups.
- Compared with V0.6.2.4 on their 126 common valid rows, V0.6.2.5 fixed 24 duplicate-group decisions and broke 29. The
  restored positive path primarily recovered false positives, not the missing true positives.

Confidence remains uncalibrated on this packet. The sole `HIGH` row was wrong; the 126 `MEDIUM` rows achieved only 49.2%
unweighted primary exact. These tiers must continue to be reported empirically rather than read as probabilities.

## Next design boundary

Do not create V0.6.2.6 by adding another self-reported categorical field to the same monolithic prompt. Two successive
smokes show that the model can repeat the rule vocabulary while failing to ground “same record” and “complete message” in
the actual uncovered spans.

The next experiment should change architecture:

1. deterministically build a blind coverage packet containing aligned shared blocks and explicit A-only/B-only spans, with
   stable span IDs and no MinHash/SUT metadata;
2. ask a smaller LLM rubric to classify each uncovered span as universal UI/repetition, same-record extension,
   identity/role/state, material non-main message, or two-sided content, citing only supplied span IDs;
3. let the adapter verify span IDs and compute directions. Containment requires a verified non-empty same-record anchor,
   uncovered content on exactly one side, and no uncovered identity/role/state or material-message span;
4. keep the existing byte-aligned bilateral evidence contract, but derive evidence from the cited packet instead of asking
   for duplicate free-form quotes;
5. independently review the 14 human-positive rows that the new policy calls material non-main-message changes. Because
   V0.6.2.5 predictions are now visible, that review must be marked diagnostic or assigned to a fresh reviewer; it is not a
   blind holdout result.

This architecture separates observation from policy resolution and gives the adapter something locally verifiable. Until
that change and the policy audit are complete, no new full-development or holdout run is justified.
