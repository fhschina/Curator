# V0.6.2.6 residual-smoke result

## Outcome

V0.6.2.6 is operationally sound but is rejected on the reconciled 127-pair residual set. It produced 127/127 valid results,
zero terminal errors, and zero pair-level retries. Its deterministic span packet and cited-span adapter therefore solved the
completion and evidence-grounding problem, but not the record-binding problem.

The label baseline includes two explicitly diagnostic audits performed after predictions were visible. Fourteen legacy
non-main containment labels were reviewed first; 13 primary tuples changed. The remaining 17 containment labels in the
residual queue were then exhausted; only `H0017` and `H0333` remained true containment. A final review of the candidate's
over-containment rows corrected `H0878` from version-related no/no to A-only containment. These are development-label repairs,
not blind holdout evidence. All previous values remain in the `pre_policy_review_human_*` fields.

## Reconciled result

| Check | V0.6.2.6 | Residual requirement | Result |
|---|---:|---:|---|
| total over-group | 35 | at most 32 | fail |
| containment over-group | 17 | at most 15 | fail |
| total under-group | 3 | at most 5 | pass |
| protected benign duplicates | 5/5 | 5/5 | pass |
| true containment primary tuple | 3/3 | 3/3 | pass |
| schema completion | 127/127 | 100% | pass |
| pair-level retries | 0 | at most 2% | pass |
| terminal errors | 0 | 0 | pass |

On this deliberately error-enriched slice, unweighted duplicate precision is 33.96%, recall 85.71%, primary-tuple exact is
66.14%, and taxonomy exact is 52.76%. Weighted precision is 31.71%, recall 84.72%, primary-tuple exact is 66.09%, and
taxonomy exact is 54.45%. These are diagnostic slice metrics, not full-population estimates.

## Diagnosis

The model cites real spans but repeatedly calls a reusable shared block the atomic record. The 17 false containment rows have
the same structural failure: a site description plus an article, author bio plus an article, legal/warranty/store policy plus
a product, checkout state plus product attributes, collection description plus list members, or cookie policy plus a page
identity. The shared text is often long and literal, so citation validity alone cannot prove that the unique content predicates
the same record.

V0.6.2.6 should remain frozen. Its span packet is a useful observation layer for the next candidate, but containment needs an
independent record-binding decision rather than another assertion inside the same semantic answer.
