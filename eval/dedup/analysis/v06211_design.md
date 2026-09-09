# V0.6.2.11: scoped supplements over the V0.6.2.9 record critic

## Freeze before evaluation

The reference remains `v0628_policy_reconciled_labels_1000.csv` (SHA-256
`6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`), restricted to the same 127 IDs in
`v0622_reconciled_residual_candidates.csv` (SHA-256
`ab1ca4efcd6618abc702521486fd2b201631be18bb1b80ff712693a983dfadea`). It includes previous prediction-aware AI
adjudications and is development-only, not an independent human holdout. No labels are edited in this iteration.

The comparator is the frozen `/raid/hfang/ihb/runs/v0.6.2.9-residual-smoke`. Model, temperature, output budget,
concurrency and visible payload stay fixed. Public output remains v3. No historical cache is migrated or reused.

## Planned sequence

1. Offline replay V0.6.2.10's saved raw scores through the proposed scoped adapter. This tests range parsing, active-branch
   evidence checks and restoration of V0.6.2.9 ownership without new inference. It is explicitly a counterfactual replay,
   not a fresh candidate run, and is never eligible for promotion.
2. `dedup-judge-hs-v0.6.2.11-dev-policy`: reuse the V0.6.2.9 main Judge verbatim. Keep the legacy record-binding critic
   instructions and score first, then add only the non-main policy subtype supplement. Independent run root:
   `/raid/hfang/ihb/runs/v0.6.2.11-policy-residual-smoke`.
3. `dedup-judge-hs-v0.6.2.11`: add the translation supplement to the policy candidate, with the same main Judge and
   legacy record-binding rubric. Independent run root: `/raid/hfang/ihb/runs/v0.6.2.11-residual-smoke`.

The policy-only variant is an ablation: it need not fix the additive-translation guard to permit the predeclared final
variant on the SAME residual. Neither variant may advance to a larger development set or holdout unless all final gates
pass. Prompts and adapter are frozen before live runs; no result-driven edits to these versions follow the evaluation.
Live calls may vary despite temperature zero, so paired outcomes do not isolate every stochastic effect.

## Scoped ownership

- Unresolved/invalid main evidence is never rescued. A valid main no/no survives unrelated supplemental citation failures.
- A material policy/inventory/page-context claim with valid own bilateral and delta evidence vetoes a positive decision.
  An unsupported material warning remains unresolved; it is not silently ignored.
- NOT_APPLICABLE is inactive. For two non-main-only sides, the main's independently verified complete-message basis remains
  decisive unless the focused policy subtype identifies a material difference. A redundant equivalence score is not the
  source of that existing decision's evidence.
- Translation directions apply only when the main establishes the same substantive record and marks complete/partial
  translation. Same-language additions and localized widgets stay under ordinary V0.6.2.9 record/delta arbitration.
  Active translation needs its own A-only and B-only citations; a redundant generic binding citation failure need not
  cancel that independent proof. Direction disagreements remain unresolved; full translation has material NONE.
- Generic BENIGN_NON_RECORD_DELTA cannot erase a cited main addition. Flattening containment additionally requires a
  focused equivalent-wrapper assessment accounting for EVERY unique span. Genuine instructions remain additions.
- Explicit same-side ID ranges expand only if every member exists. Unknown IDs, gaps, reverse or cross-side ranges fail.

The V0.6.2.9 record arbitration is reused in code, not rewritten. The main prompt, its pair prompt, and main rubric remain
identical. Supplements are synchronized between new critic system/pair instructions and YAML descriptions.

## Frozen final residual gates

- Weighted precision, recall and primary exact are each no worse than the frozen V0.6.2.9 results (43.6974%, 77.6514%,
  76.3912%, respectively; comparisons use unrounded values from the same evaluator).
- Over-group <=20; false containment <=1; under-group <=4; unresolved <=5.
- Protected primary tuples 5/5: H0038, H0347, H0453, H0679, H0748.
- True-containment primary tuples 4/4: H0878, H0333, H0653, H0017.
- Original 23 diagnostic negatives (V0.6.2.3 follow-up + V0.6.2.4 preflight) all resolved NO.
- Schema completion 100%, terminal errors zero, pair retry rate <=1%.

UNRESOLVED reference duplicates remain recall misses. Report confidence-tier empirical accuracy and all error IDs;
schema validity does not imply semantic resolution. Historical taxonomy/translation MINOR is diagnostic only. The old
V0.6.2.10 gate profile stays reproducible; a separate `v06211` audit profile enforces the tighter limits.

Only a passing final residual candidate proceeds to the 1,000-row development set and original precision/recall >=75%,
primary exact >=79%, cohort/over-group/operational gates. The untouched holdout and full 20,000-pair run remain gated.
No model agreement or MinHash guess substitutes for semantic reference evaluation or a missing SUT contract.
