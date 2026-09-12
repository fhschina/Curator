# v0.6.2.33-exp5 — Exp1 + evidence-only repair

Named by the user on 2026-09-12. Short name: **Exp5**.

`v0.6.2.33-exp5` names the existing, validated checkpoint
`v0.6.2.33-exp1+conflict-evidence-fix1`. This is a naming alias, not another
implementation change or a formal release.

The implementation remains `exp1_conflict_evidence_fix.execute_case`.
Its SHA-256 is
`572a446c30b8fb21f1a90cf9759e99f812ea0d6b7d07d518aa2c60656b4b2197`.
The scope is defined in `exp1_conflict_evidence_fix.md`: original Exp1 plus
one evidence-only repair opportunity after `CRITIC_SCOPE_CONFLICT`.
No Exp4 component, semantic change, main prompt/parser change, or transport
change is included.

Historical source files, result versions, run directories and manifests retain
their original names and hashes. The frozen runtime's `version` field therefore
still emits the original checkpoint identifier. Future experiment metadata and
reports should identify this checkpoint as Exp5 and retain the original runtime
identifier as provenance; do not rewrite historical artifacts to rename them.

Existing validation is in
`/raid/hfang/ihb/runs/v06233-exp1-conflict-evidence-fix1-check-v1`.
It replayed 1,000 saved cases with one new live evidence-repair call: H0537
became valid and the other 999 results were unchanged. This was not a fresh
1,000-case evaluation. H0074/H0075 remain historical transport failures.

Naming Exp5 does not claim that independent holdout validation or release gates
have passed, and does not itself launch or authorize a 20,000-pair run.
