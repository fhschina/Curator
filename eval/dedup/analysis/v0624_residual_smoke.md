# V0.6.2.4 residual-smoke result

## Outcome

V0.6.2.4 is rejected at the residual-smoke gate. It must not advance to the full 1,000-pair development run or the
holdout. The immutable partial run is `/raid/hfang/ihb/runs/v0.6.2.4-residual-smoke`; the machine-readable gate result is
`v0624_residual_smoke_summary.json`.

The run produced 126 valid V3 results for 127 requested pairs. `H0278` failed closed after three attempts because the
model repeatedly selected positive record-identity support without supplying the separate field-specific bilateral exact
quotes required by V0.6.2.4. Six other rows needed two or three attempts before passing the same family of checks. Counting
the failed row, seven pairs were retried (5.51%). The formal relay recorded 162 successful responses and 14 recovered HTTP
500 responses; the preflight recorded 11 successful responses.

For semantic scoring, the failed row is treated as fully unresolved. This preserves fail-closed accounting instead of
silently dropping it.

## Frozen gate result

| Check | V0.6.2.3 on cumulative labels | V0.6.2.4 | Required | Result |
|---|---:|---:|---:|---|
| total over-group | 46 | 25 | at most 32 | pass |
| containment over-group | 22 | 0 | at most 15 | pass |
| total under-group | 5 | 21 | at most 5 | fail |
| named benign duplicates retained | 5/5 | 2/5 | 5/5 | fail |
| diagnostic negative labels preserved | n/a | 21/23 | 23/23 | fail |
| schema completion | 100% | 99.21% | 100% | fail |
| terminal errors | 0 | 1 | 0 | fail |
| retry rate | 1.57% | 5.51% | at most 2% | fail |

On this deliberately error-enriched slice, fail-closed unweighted duplicate precision is 44.44%, recall 48.78%, and
primary-tuple exact agreement 49.61%. Weighted precision is 44.86%, recall 48.67%, and primary-tuple exact agreement
50.04%. These slice metrics are diagnostic and are not estimates of full-development prevalence.

## What improved

- Total false grouping fell from 46 to 25, a 45.7% reduction.
- False containment fell from 22 to zero.
- The hard-conflict branches were reliable on this packet: all identity/slot, legal-context, list-membership, and
  state/version conflicts that the ledger actually identified were classified correctly.
- The new overlap-scope and surface-delta fields expose the decision failure much more clearly than V0.6.2.3's single
  same-record classification.

## Why the candidate failed

The adapter made `MATERIAL_PROPOSITION_OR_RECORD` an unconditional no/no veto. That is correct for conflicting cookie
purposes, changed states, and different records, but wrong for a real one-sided addition to the same record. The candidate
therefore emitted no `CONTAINMENT` result at all. Fourteen true containments and four other true duplicates were rejected
through this veto, while three more benign duplicates were rejected for lack of a complete non-main-message proof.

The opposite boundary remained too permissive. The model labeled 45 rows as bidirectionally equivalent; only 20 were true
duplicates. Eighteen false positives flowed through `SAME_NON_MAIN_MESSAGE` plus `COMPLETE_NON_MAIN_MESSAGE`, usually after
the model called a page-specific title, product/category slot, page role, count, date, or extra legal/cookie proposition a
mere navigation/formatting delta. `H0276` and `H0983` are also two of the 23 newly reviewed negative labels that this route
incorrectly regrouped.

The five protected benign duplicates were not handled consistently. `H0679` and `H0748` were retained, but `H0038`,
`H0347`, and `H0453` were split even though their only accepted deltas are redundant placeholder repetition or the frozen
wishlist/learn-more/accept boundary.

Finally, the separate evidence mini-contract inside `record_identity_support` was operationally too brittle. General
bilateral quote evidence was usually available, but asking the model to repeat aligned quotes inside a second rubric field
created six recovered retries and one terminal failure without improving calibration.

## Next immutable boundary

A successor must not mutate V0.6.2.4. It should:

1. remove the separate field-specific identity-quote requirement and keep the existing general bilateral exact-evidence
   contract;
2. distinguish a verified one-sided same-record content extension from a conflicting/material non-main message instead of
   treating every material proposition as no/no;
3. split universal UI/format/repetition deltas from page-specific title, product/category, profile, count, date, role, and
   state deltas;
4. permit containment only for the explicit same-record-extension class, while forcing record/role/state and material
   message changes to no/no;
5. add boundary examples drawn from the observed error families, without exposing review IDs or expected labels in the
   evaluation payload.

The next smoke must use a fresh prompt version, contract digest, run root, and cache. V0.6.2.4's incomplete cache is
diagnostic only and must not be resumed as a release candidate.
