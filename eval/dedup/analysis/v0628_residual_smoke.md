# V0.6.2.8 residual-smoke result

## Outcome

V0.6.2.8 is rejected and must not advance to the full development set or holdout. It completed cleanly with 127/127 valid
results, zero pair-level retries, and zero terminal errors. It reduced total over-group from 25 to 19, reduced false
containment from four to one, restored protected sample `H0748`, and preserved all 23 original diagnostic negatives.

Those precision-side gains came from excessive rejection. Under-group increased from five to 13, duplicate recall fell from
76.19% to 38.10%, and only three of five protected benign duplicates survived. None of the four reconciled true containment
directions was exact.

## Reconciled result

| Check | V0.6.2.7 | V0.6.2.8 | Residual requirement | Result |
|---|---:|---:|---:|---|
| total over-group | 25 | 19 | at most 32 | pass |
| containment over-group | 4 | 1 | at most 15 | pass |
| total under-group | 5 | 13 | at most 5 | fail |
| protected benign duplicates | 4/5 | 3/5 | 5/5 | fail |
| true containment primary tuple | 3/4 | 0/4 | 4/4 | fail |
| original diagnostic negatives preserved | 22/23 | 23/23 | 23/23 | pass |
| schema completion | 100% | 100% | 100% | pass |
| pair-level retries | 0 | 0 | at most 2% | pass |
| terminal errors | 0 | 0 | 0 | pass |

The true-containment denominator is four after diagnostic review corrected `H0653`: its English side translates the German
Videvo record and adds contact/request information, so it is additive translation rather than complete translation. This
review occurred after predictions were visible and is development-only evidence.

On the error-enriched slice, unweighted precision is 29.63%, recall 38.10%, primary-tuple exact is 71.65%, and taxonomy exact
is 51.97%. Weighted precision is 28.11%, recall 37.34%, primary-tuple exact is 72.49%, and taxonomy exact is 54.37%.

## Failure analysis

The new critic selected `NON_MAIN_POLICY_OR_STATE_CHANGE` 40 times: 32 human negatives and eight human positives. Treating
that imperfect verdict as an adapter veto directly caused eight false negatives: equivalent cookie/chrome cases `H0089`,
`H0214`, `H0226`, `H0347`, `H0453`, `H0604`, `H0741`, and `H0792`. The implementation therefore reopened a boundary that
was intended to remain governed by the main non-main ledger.

The critic also called the genuine additions in `H0333` and `H0878` benign, so the adapter flattened them to yes/yes. It was
unresolved on additive translation `H0017` and called `H0653` two-sided/conflicting. The remaining false containment is
`H0723`, where a cookie-policy inventory plus a page identity is still treated as one atomic extension.

## Next boundary

V0.6.2.8 should remain frozen. A future candidate should make the critic asymmetric and conservative:

1. never let the secondary critic veto when both main profiles are `NON_MAIN_ONLY`; the already calibrated main ledger owns
   complete-message equivalence and material non-main differences;
2. never let `BENIGN_NON_RECORD_DELTA` erase a main-ledger `SAME_RECORD_CONTENT_EXTENSION`; benign critic evidence may retain
   the duplicate group but cannot rewrite a cited material addition;
3. keep explicit substantive `SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT` and `TWO_SIDED_OR_CONFLICTING` vetoes;
4. handle partial/additive translation separately from lexical record binding, and keep unresolved critic evidence from
   silently becoming a confident semantic decision;
5. target `H0723` as the remaining policy/page-role containment case.

No additional label tuning, full-development run, or holdout access is justified from this candidate.
