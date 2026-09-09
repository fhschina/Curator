# V0.6.2.3 residual-smoke result

## Outcome

V0.6.2.3 is rejected at the residual-smoke gate and must not advance to the full 1,000-pair development run or the holdout.
The immutable run is `/raid/hfang/ihb/runs/v0.6.2.3-residual-smoke`.
The machine-readable gate result is `v0623_residual_smoke_summary.json`.

All 127 requested rows produced valid V3 results with no terminal errors. Two rows required a retry, a 1.57% retry rate.
The run used a new V0.6.2.3 contract digest and did not reuse V0.6.2.2 Judge cache.

## Gate result

| Check | V0.6.2.2 residual baseline | V0.6.2.3 | Required | Result |
|---|---:|---:|---:|---|
| containment over-group | 27 | 22 | at most 18 | fail |
| total over-group | 59 | 46 | at most 59 | pass |
| total under-group | 17 | 28 | at most 17 | fail |
| reviewed true duplicates retained | 7 | 4 | at least 6 | fail |
| five residual translation misses recovered | 0 | 0 | at least 4 | fail |
| schema completion | 100% | 100% | 100% | pass |
| retry rate | 0.8% on full baseline | 1.57% | at most 1% | fail |

V0.6.2.3 corrected 19 rows whose duplicate-group decision was wrong under V0.6.2.2, but introduced 15 new
duplicate-group disagreements among rows that V0.6.2.2 had classified correctly. Primary-tuple exact agreement on this
deliberately error-only packet rose from zero to 18/127, but that gain is not sufficient to offset the new false negatives.

## Diagnosis

The new ledger is directionally useful but too coarse in two places:

- It reduced containment false positives by only five. Twenty-two false containments still received
  `SAME_SUBSTANTIVE_RECORD`; the model continues to treat a shared local passage, product-family copy, or reusable page text
  as record identity. A categorical self-report does not reliably prove record alignment.
- It rejected three of the seven manually confirmed duplicates (`H0038`, `H0748`, and `H0921`) by treating short template,
  navigation, category, forum-description, or membership text as a distinct substantive record instead of ignorable chrome.
- None of the five residual translation misses (`H0104`, `H0397`, `H0815`, `H0900`, `H0920`) became duplicate. Inspection
  shows that several contain visible number, organization, slot, or page-purpose differences, so they require dedicated
  policy adjudication rather than a prompt rule that blindly forces translation equivalence.
- Fifteen newly reported containment under-groups include many cookie, forum-permission, blocked-page, and disclaimer pairs
  that were not part of the authorized 85-row review. Several appear to be remaining old-policy label conflicts, so they
  cannot be used as clean evidence that the stricter Judge is wrong without a second adjudication pass.

## Next version boundary

Do not mutate V0.6.2.3 after this run. A subsequent candidate should use a new version and should replace the single
`SAME_SUBSTANTIVE_RECORD` self-classification with verifiable identity-anchor fields: an A identity quote, a B identity quote,
and an explicit anchor kind. It should separately classify `shared local passage` and `generic family copy`, and should add a
small-page rule that treats short category/navigation/site-description text as chrome when the complete retained message is
otherwise the same.

Before that candidate is scored for recall, adjudicate the five translation rows and the 15 newly surfaced containment-label
conflicts. Those rows were outside the 85-row review requested for V0.6.2.3 and should remain unchanged until explicitly
reviewed.
