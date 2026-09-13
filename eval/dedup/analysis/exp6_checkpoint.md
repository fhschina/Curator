# v0.6.2.33-exp6 — Exp5 + shared-anchor numbering tolerance

Named by the user on 2026-09-13. Short name: **Exp6**.

`v0.6.2.33-exp6` names the validated checkpoint
`v0.6.2.33-exp5+anchor-id-fix1`. This is a naming alias, not another semantic
change, a fresh benchmark result, or a formal release.

The implementation is `exp5_anchor_id_fix.execute_case`; its SHA-256 is
`9388dc6ff3c54bd7e92ab6a060e6da3718f2c3eb91e122e850a61bd46cc5d923`.
The scope is defined in `exp5_anchor_id_fix.md`: frozen Exp5 plus deterministic
zero-padding of the initial coverage critic's shared-anchor references.
Only uniquely existing, fully aligned shared spans can be selected by the
corrected spelling. Unknown, ambiguous, wrong-kind and duplicate references
remain invalid. No evidence is guessed, removed or deduplicated.

Main/critic prompts, semantic fields, model settings, routing and retry limits
are unchanged. Exp5's original conflict-evidence repair remains unchanged.
No Exp4 component or generation/output-limit repair is included.

As with the Exp5 naming alias, historical result versions, source files, run
directories and manifests retain their original names and hashes. The runtime
still emits `v0.6.2.33-exp5+anchor-id-fix1` for reproducibility. Future experiment
metadata and reports should identify this checkpoint as Exp6 while retaining
that runtime identifier as provenance. Do not rewrite the completed Exp5 20K
run or its frozen entry point to enable this patch.

## Validation

Saved verification is in
`/raid/hfang/ihb/runs/v06233-exp5-anchor-id-fix1-check-v1`.
Its patch contract digest is
`f01ef67f7cf27307d4170ead2884024393836f8b36c76196306c4180325fa3c0`.

- 100 related tests passed; repository pre-commit checks passed.
- All 2,293 saved coverage responses were checked offline. The other 2,291
  valid coverage outputs and decisions were unchanged.
- P16061: `S21` → `S021`; the full pipeline replayed its two saved calls and
  completed with a valid output, preserving the saved main decision.
- P09549: `S41` → `S041`; coverage validation passed without changing the main
  decision. The existing routing still requires a subject review that the
  failed historical run never collected. Its full result remains incomplete.
- Zero new model calls. P01329's generation failure remains out of scope.

The old 20K report still records its original three engineering failures.
An offline proof repair is not an independent semantic correctness adjudication
or a fresh benchmark. Naming and publishing Exp6 does not authorize another
online evaluation or claim that release gates and holdout validation passed.
