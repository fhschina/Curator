# Exp5 exploratory full 20k

The user authorized this run on 2026-09-12 for a Monday demonstration, explicitly
before independent holdout validation. This is not release approval; the original
release gates remain unchanged. All 20,000 original pairs are included, with
10,000 from each track. Previously evaluated pairs are not independent holdout.

The runtime is the unchanged `exp1_conflict_evidence_fix.execute_case`, publicly
named `v0.6.2.33-exp5`. Its historical runtime version remains in raw results.
No Exp4 prompt, parser, structured-main output, or semantic rule is enabled.
The existing deterministic span builder adds Exp1's span packet to the exact
v0.6.1 visible texts/windows. All 1,000 development payloads must match Exp1
byte-for-byte at the structured-payload level before any new call.

All main and routed critic responses are fresh. Only same-run, request-bound
receipts may be reused on interruption recovery. A run-scoped lock prevents
concurrent collectors; process identity and 30-second heartbeats permit live
verification. Original Exp1 rate-limit continuation waits on temporary HTTP 429
with Retry-After respected. The original two workers, two-second pacing, model,
generation, parser corrections and transport timeouts are retained. A 120,000
external-request ceiling covers the full population and bounded retries; it is
a ceiling, not an intended request count. Prolonged limits leave pairs pending.

All terminal failures and semantic unresolved results remain visible and in the
denominator. No intermediate semantic score is used to alter the experiment.
Completion requires all 20,000 results and exact offline replay of every stage,
including the failure-only evidence repair. Source/request/response bindings are
preserved; raw results are never overwritten with revised judgments.

Comparison uses the original v0.5 results on the same pair IDs. Version agreement
is descriptive, not accuracy. Development calibration and proxy challenge must
remain separate; no proxy or DeepSeek agreement decides the winning version.
SUT observations are separate from Judge outputs. MinHash diagnostics remain
`UNAVAILABLE_MISSING_CONTRACT` until actual resolved SUT configuration exists.

Commands: `python -m eval.dedup.analysis.exp5_full20k prepare|launch|status|audit
--root <new-run-root>`. Launch additionally takes `--env-file <existing-env-file>`.
Secrets are read at execution and never included in the manifest or command line.
