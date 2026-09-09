# V0.6.2.7 residual-smoke result

## Outcome

V0.6.2.7 is rejected, despite a large precision-side improvement. The independent critic reduced total over-group from 35
to 25 and false containment from 17 to 4. Primary-tuple exact increased from 66.14% to 74.02%. The run was operationally
clean: 127/127 valid, zero pair-level retries, and zero terminal errors.

It cannot advance because it retained only four of five protected benign duplicates and exactly directed only two of the
three reconciled true containments. `H0748` was incorrectly vetoed as a separate record, while `H0878` was incorrectly
flattened from a genuine one-sided FAQ extension to bidirectional equivalence. `H0394`, an archive profile with reordered
fields plus one added field, was also incorrectly vetoed and accounts for the other newly exposed under-group.

## Reconciled result

| Check | V0.6.2.6 | V0.6.2.7 | Residual requirement | Result |
|---|---:|---:|---:|---|
| total over-group | 35 | 25 | at most 32 | pass |
| containment over-group | 17 | 4 | at most 15 | pass |
| total under-group | 3 | 5 | at most 5 | pass at boundary |
| protected benign duplicates | 5/5 | 4/5 | 5/5 | fail |
| true containment primary tuple | 3/3 | 2/3 | 3/3 | fail |
| schema completion | 100% | 100% | 100% | pass |
| pair-level retries | 0 | 0 | at most 2% | pass |
| terminal errors | 0 | 0 | 0 | pass |

Unweighted precision is 39.02%, recall 76.19%, primary-tuple exact is 74.02%, and taxonomy exact is 55.91%. Weighted
precision is 37.59%, recall 75.58%, primary-tuple exact is 75.10%, and taxonomy exact is 58.05%. This is an error-enriched
development slice, not a population estimate.

## What the critic fixed and what remains

The critic correctly vetoed most article/bio, product/policy, seller-profile/legal, and page-role attachments. Four false
containments remain: `H0634`, `H0723`, `H0267`, and `H0253`. They expose four missing critic boundaries: forum banner versus
search page; cookie-policy inventory versus page identity; store checkout/shipping block versus product attributes; and
collection description versus list membership.

The next immutable candidate should keep the independent critic but make a minimal, testable revision:

1. treat a real unique FAQ instruction as an atomic extension even when other unique tokens are spelling variants;
2. treat reordered same-record fields plus one missing field as an atomic extension;
3. treat short trailing story/navigation teasers as benign when the complete shared record is already present;
4. explicitly reject the four remaining attachment patterns above;
5. let a critic veto a positive substantive-main decision, not only containment, while leaving non-main equivalence under
   the main span ledger until the critic is separately calibrated for complete-message equivalence.

No holdout or full 1,000-pair development run is authorized from this result.
