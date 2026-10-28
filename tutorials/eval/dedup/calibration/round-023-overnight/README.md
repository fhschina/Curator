# Qwen retention-policy overnight calibration

This is the active 2026-10-28 overnight development experiment on
`codex/dedup-retention-policy-v3-overnight`. The delivery deadline is
2026-10-28 16:00 UTC (09:00 PDT); semantic search stops at 14:30 UTC and the
task-owned GPU service must stop by 15:45 UTC.

The fixed protocol requires the same frozen candidate to reach 648/720
directions and 306/360 pairs across its three declared personal-reference
runs. The intermediate milestone is 612/720 and 288/360. Hard regressions,
technical validity, and all three development nonregression views are separate
requirements. These exposed references measure development agreement.

The 06:43 UTC checkpoint contains the complete same-condition baseline,
starting-version and candidate-02 comparison. Candidate-02 has 588/720 correct
directions and 279/360 complete pairs. Its development directional totals are
564/594 original, 570/594 reviewed and 558/582 strict; all exceed the fixed
baseline totals. Its personal milestone, target, latest-review and controlled
hard groups, and technical validity fail. No candidate is accepted.

- `checkpoints/0643Z/`: complete machine comparison, CSV, bilingual readable
  reports, shared tables, translation validation, and saved current pointers.
- `frozen-candidates/`: immutable prompt/config snapshots and provenance.
  Archive notes describe their state when first saved; later checkpoint notes
  and run evidence preserve subsequent outcomes.
- `candidate-diffs/`: changes against each declared parent.
- `run-evidence/`: strict per-layer JSON and CSV, including failed outputs.
- `engineering-04-ledger-completeness/`: uniform literal-coherence audit and
  its CPU verification; it removes invalid credit without rewriting answers.
- `engineering-06-pointer-refresh/`: actual integration evidence that a newer
  repeat refreshes its own candidate's current mean while preserving historical
  snapshots. The operational controller is preserved as exact source text.
- [REPRODUCE.md](REPRODUCE.md): frozen native replay entry and prerequisites.

Raw source corpora and full HTTP receipts remain outside Git at
`/raid/hfang/retention_policy_eval/v1/experiments/round-023-overnight-20261028T025929Z`.
The final selection, resource cleanup and delivery receipts will record the
observed final result; an archived or committed candidate is not acceptance.
