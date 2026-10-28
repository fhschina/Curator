# Qwen retention-policy overnight calibration

This is the completed 2026-10-28 overnight development experiment on
`codex/dedup-retention-policy-v3-overnight`. The delivery deadline is
2026-10-28 16:00 UTC (09:00 PDT); semantic search stops at 14:30 UTC and the
task-owned GPU service must stop by 15:45 UTC.

See [the final evidence index](delivery/README.md) and the bilingual reports
for the delivered result, unmet requirements and timing record.

The fixed protocol requires the same frozen candidate to reach 648/720
directions and 306/360 pairs across its three declared personal-reference
runs. The intermediate milestone is 612/720 and 288/360. Hard regressions,
technical validity, and all three development nonregression views are separate
requirements. These exposed references measure development agreement.

The confirmed 08:35 UTC checkpoint contains the complete same-condition
baseline, starting-version and candidate-06-languagefirst1 comparison. This
frozen candidate confirms the **85% / 80% milestone**, with 612/720 directions
and 288/360 pairs across its original three runs. The final 90% / 85% target,
latest-review and controlled hard groups, and technical validity fail. All
three 100-pair nonregression views pass. No candidate is accepted.

- `checkpoints/progress-101731Z/`: refreshed bilingual reports, full machine comparison and retained snapshot history.
- `engineering-11-attempt-signal-trace/`: read-only raw-attempt signal summaries and safe row projections; no scoring changes.
- `milestones/85_80-languagefirst1-confirmed-0822Z/`: confirmed snapshot,
  immutable evidence, exact counts, and bilingual qualification summary.
- `checkpoints/0835Z-confirmed-stage/`: complete comparison, bilingual reports,
  shared tables, translation validation and current pointers.
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
The final selection and resource cleanup are recorded in `delivery/`. Delivery
was late; the final metric target was not reached and no candidate passed
acceptance. An archived or committed candidate is not acceptance.
