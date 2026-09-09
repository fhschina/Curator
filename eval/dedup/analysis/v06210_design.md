# V0.6.2.10: evidence-scoped boundary review

## Frozen development protocol

This candidate implements the follow-up from the V0.6.2.9 residual audit. It is a development experiment, not a release
and not an independent human-blind evaluation. The reference labels include prior prediction-aware AI diagnostic
adjudications. No labels are changed for this iteration and no holdout is accessed.

- Labels: `v0628_policy_reconciled_labels_1000.csv`, SHA-256
  `6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`.
- Residual membership: `v0622_reconciled_residual_candidates.csv`, SHA-256
  `ab1ca4efcd6618abc702521486fd2b201631be18bb1b80ff712693a983dfadea`, 127 pairs.
- Comparator: frozen `/raid/hfang/ihb/runs/v0.6.2.9-residual-smoke` results, not re-adapted or re-judged.
- New isolated run/cache: `/raid/hfang/ihb/runs/v0.6.2.10-residual-smoke`.
- Same Qwen model, temperature 0, output budget 4096, concurrency 64 and visible-payload-v3 membership as V0.6.2.9.
- Public output remains `dedup-judge-output-v3`; the new critic and source are included in implementation provenance.
  Two independent LLM calls remain; neither sees the other's answer or review IDs.

## Changes

System prompt, pair prompt and YAML descriptions jointly distinguish:

1. A repeated non-main wrapper from new policy propositions, permissions, cookie inventory or page context. Inventory
   changes veto containment even when the main Judge calls the policy substantive. Dismissible chrome around the same
   complete substantive article does not redefine the article.
2. Cross-language subject alignment from fact coverage. Translation review must cite A-only and B-only spans in its own
   reasoning, identify extra facts on the owning side, and not demand a lexical shared subject. Complete faithful
   translation remains bidirectional with material difference NONE.
3. Exhaustively cited chrome/redundancy from semantic equivalence. Only the former may flatten a cited main extension;
   the latter preserves that extension at LOW confidence. Small real instructions still count as content.

The critic has separate `non_main_delta_subtype`, `translation_delta_direction`, and `record_binding_verdict` fields.
Evidence is validated per field; a citation elsewhere in the critic cannot satisfy a translation or policy claim.
Invalid evidence cannot rescue an unresolved main ledger or override explicit hard conflicts. Translation direction
disagreement is unresolved; a semantic translation review cannot upgrade a negative main decision into a duplicate.
Reason codes expose all three internal scores and the arbitration rule; evidence stays aligned to the visible payload.

## Gates fixed before the run

Keep the prior residual gates: over-group <=32, containment over-group <=15, under-group <=5, protected duplicate groups
5/5 (`H0038`, `H0347`, `H0453`, `H0679`, `H0748`), exact true-containment primary tuples 4/4 (`H0878`, `H0333`, `H0653`,
`H0017`), and the original 23 diagnostic negatives all NO. Require 100% schema completion and zero terminal errors.
Tighten pair retry rate from the prior smoke's 2% to the original plan's 1%; additionally require all five protected
primary tuples to match, so a false containment such as H0748 cannot silently pass as a correct group.

Report weighted and unweighted precision/recall, primary exact, historical taxonomy, error IDs, confidence-tier accuracy,
and boundary/cohort outcomes. The residual is error-enriched; original sampling weights do not make it representative.
UNRESOLVED on a reference duplicate counts as a miss in recall and containment gates. This corrects the shared evaluator's
previous resolved-only denominator; historical saved results/reports are not rewritten. Taxonomy is diagnostic only and
legacy translation MINOR does not participate in primary selection.

Any failed residual gate stops progression to the 1,000-row development set or holdout. Passing this smoke is only
permission to evaluate the full development set, not a release decision. Full 20,000-pair execution still requires the
original development and once-only holdout gates. MinHash remains unavailable without the actual SUT contract; no
retriever parameters or LLM predictions substitute for it.
