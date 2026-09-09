# V0.6.2.3 reviewed-boundary design

## Decision

V0.6.2.2 is a valid development baseline after the 85-row policy reconciliation, but its 127 residual primary/taxonomy
disagreements expose two opposite boundary failures. V0.6.2.3 keeps the published `dedup-judge-output-v3` result contract and
adds three prompt-only semantic-ledger fields whose values are resolved deterministically by the adapter.

No V0.6.2.2 cache is reusable. V0.6.2.3 has an immutable prompt version, resource digest, run root, and cache namespace.

## Residual evidence

V0.6.2.2 has 59 over-group and 17 under-group errors after reconciliation. Twenty-seven over-groups are containment and 32
are bidirectional equivalence. Of the bidirectional false positives, 24 are pairs classified as non-main-only with an
apparently equivalent message even though the visible controller, target, legal proposition, state, list membership, or page
role differs. The containment false positives repeatedly claim a substantive anchor without proving the same record; common
cases are distinct articles sharing a local passage, generic product-family copy paired with one SKU, and reusable policy or
template text.

The under-group set contains true same-record pages rejected for harmless headers, navigation, repetition, captions, or field
ordering, plus five old-label translation candidates requiring a dedicated completeness review. Seven manually confirmed
same-record duplicates must be protected:
`H0038`, `H0361`, `H0394`, `H0488`, `H0748`, `H0811`, and `H0921`.

## New semantic-ledger fields

The model completes the existing five gates, then these three fields:

1. `record_alignment`
   - `SAME_SUBSTANTIVE_RECORD`
   - `SAME_NON_MAIN_MESSAGE`
   - `GENERIC_OR_TEMPLATE_OVERLAP_ONLY`
   - `DIFFERENT_RECORD_OR_ROLE`
   - `UNRESOLVED`
2. `non_main_difference`
   - `NONE_OR_IGNORABLE_CHROME`
   - `MATERIAL_MESSAGE_OR_STATE`
   - `NOT_APPLICABLE`
   - `UNRESOLVED`
3. `translation_status`
   - `COMPLETE_FAITHFUL`
   - `PARTIAL_OR_ADDITIVE`
   - `NOT_TRANSLATION`
   - `UNRESOLVED`

The fields remain diagnostic prompt output rather than additions to the stored V3 schema. Their selected options are retained
as reason codes, so calibration can report accuracy and error concentration for every gate without a schema migration.

## Deterministic resolution order

1. Any unreadable or unresolved gate produces the fully unresolved/low result.
2. Any visible hard conflict produces `NO/NO`.
3. `COMPLETE_FAITHFUL` produces `YES/YES`, `NEAR_SURFACE`, and material `NONE` only when both sides are substantive and a
   substantive anchor plus `SAME_SUBSTANTIVE_RECORD` are present.
4. A substantive/non-main profile mismatch produces `NO/NO`.
5. Substantive pages require both a substantive anchor and `SAME_SUBSTANTIVE_RECORD`; otherwise they are `NO/NO`.
6. Non-main pages require an equivalent message plus `SAME_NON_MAIN_MESSAGE`; any `MATERIAL_MESSAGE_OR_STATE` delta produces
   `NO/NO`.
7. Two-sided main divergence produces `NO/NO`.
8. `NONE` or ignorable non-main differences produce `YES/YES`.
9. Containment requires two substantive pages, a substantive anchor, `SAME_SUBSTANTIVE_RECORD`, no conflict, and exactly one
   main-addition side.

The prompt and YAML rubric explicitly distinguish missing from conflicting values, order changes from membership changes,
complete from partial translation, specific legal policies from generic footer text, and record identity from generic product
or template overlap.

## Evaluation sequence

### Residual smoke

Run the immutable candidate on all 127 reconciled residual rows. Advance to the full development run only if:

- containment over-group falls from 27 to at most 18;
- total over-group does not exceed 59 and total under-group does not exceed 17;
- at least six of the seven manually confirmed duplicates remain or become `YES`;
- at least four of the five residual translation candidates become `YES`;
- schema completion is 100% and retry rate is at most 1%.

### Full reconciled development set

Run all 1,000 pairs with identical model, decoding, and payload settings. The candidate may be selected only if:

- weighted duplicate precision and recall are each at least 75%;
- weighted primary-decision exact agreement is at least 87.00%, a one-point non-inferiority floor relative to V0.6.2.2's
  88.00%, and remains above the original 79% release target;
- over-group is at most 59 and containment over-group is at most 18;
- all seven reviewed true duplicates are preserved;
- identity, meaningful-addition, and translation cohort accuracy do not regress by more than three percentage points from
  V0.6.2.2;
- schema completion is 100% and retry rate is at most 1%.

Only after this pass is the 400-pair holdout frozen and reviewed. The holdout is viewed once under the previously defined
representative non-inferiority, precision/recall, difficult-over-group, and containment-miss gates. Failure converts that
holdout into development data and requires a newly sampled holdout.

## Executed status

The immutable residual smoke completed with 127/127 valid results and no terminal errors, but V0.6.2.3 failed five of seven
admission checks. Containment over-group fell only from 27 to 22, under-group rose from 17 to 28, only four of seven protected
duplicates were retained, none of the five targeted translation misses was recovered, and two retries produced a 1.57% rate.
The full 1,000-pair development run and holdout were therefore not started. See `v0623_residual_smoke.md` for the result and
diagnosis. This version is frozen as a rejected experiment; further prompt changes require a new version.
