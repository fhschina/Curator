# V0.6.2.2 policy reconciliation

## Scope and review protocol

The original 1,000-pair development labels remain immutable in
`hs_blind_adjudication_1000.csv`. This reconciliation overlays explicit decisions for the 35 highest-priority
`POTENTIAL_POLICY_LABEL_CONFLICT` rows and 50 highest-priority `POTENTIAL_MODEL_CLASSIFICATION_ERROR` rows. The review used
only the visible A/B payload text in a deterministic shuffled order and applied the V0.6.2 semantic-retention policy. It did
not use the old human decision or either model prediction while deciding each row.

This is a prediction-isolated single-reviewer adjudication performed by the assistant at the user's request. It is not a
statistically never-exposed, dual-independent human review. The overlay therefore qualifies the development set for prompt
iteration, but it does not replace the planned independent holdout review.

## Label-policy resolution

The governing unit is retained semantic meaning, not shared bytes or page ancestry:

- containment requires visible substantive content from the same underlying record on both sides and exactly one strict
  main-content superset;
- generic cookie, legal, navigation, authentication, error, paywall, template, product-family, or local-passage overlap
  cannot prove a shared record or create containment;
- a short header, footer, banner, or about link does not make an otherwise non-main-only page substantive;
- two non-main-only pages are duplicates only when the complete message, function, identity/controller, target, and state
  agree; legal and cookie text can carry material distinctions;
- a hard conflict requires two visible incompatible values or roles. A value missing on one side can be a one-sided addition,
  and a reordered copy of the same values is not a conflict;
- the same main record with only redundant chrome, navigation, caption, repetition, ordering, or formatting differences is
  bidirectionally replaceable;
- a complete faithful translation is bidirectionally replaceable with material difference `NONE`; partial or additive
  translation follows ordinary coverage rules.

## Adjudication outcome

All 85 requested rows received a decision and rationale. Seventy-seven are `NO/NO`, seven are `YES/YES`, and one is
`UNRESOLVED` because the visible payload does not support a safe decision. The duplicate-group label changed on 78 rows and
the three-field primary tuple changed on 82 rows. The seven reviewed duplicates are `H0038`, `H0361`, `H0394`, `H0488`,
`H0748`, `H0811`, and `H0921`.

The most frequent resolved causes were boilerplate-only overlap (27), identity/slot conflicts (15), non-equivalent non-main
messages (10), legal-context conflicts (7), and two-sided main divergence (6). The adjudication CSV contains all row-level
decisions; the reconciled CSV preserves the five prior label values under `pre_policy_review_human_*` columns.

## Recomputed V0.6.2.2 result

Against the reconciled 1,000-pair development labels, V0.6.2.2 has weighted duplicate precision 76.06%, recall 92.11%, and
primary-decision exact agreement 88.00%. Unweighted precision is 79.58%, recall 93.12%, and primary exact agreement 87.30%.
It has 59 over-group errors, including 27 containment over-groups, and 17 under-group errors. Schema completion is 100% and
the retry rate is 0.8%. Every original development gate passes under the reconciled policy.

These figures supersede the original-label gate interpretation only for subsequent V0.6.2.x development. The original
comparison remains the historical measurement of agreement with the pre-reconciliation taxonomy.

## Artifacts

- `v0622_priority_review_adjudications.csv`: 85 row-level review decisions and rationales.
- `v0622_priority_review_summary.json`: overlay counts and protocol.
- `v0622_policy_reconciled_labels_1000.csv`: development labels with original values retained.
- `v0622_policy_reconciled_development_comparison.json`: metrics and gates after reconciliation.
- `v0622_reconciled_residual_candidates.csv`: 127 remaining primary/taxonomy disagreements with visible payload snippets.
- `v0622_reconciled_residual_summary.json`: residual-cluster counts.

