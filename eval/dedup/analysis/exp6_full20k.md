# Exp6 repetition-fix2: fresh 20K with an engineering smoke gate

Entry point: `python -m eval.dedup.analysis.exp6_full20k`.
Runtime: `v0.6.2.33-exp6+main-repetition-fix2` (commit `34ee5fde` patch).
This is an exploratory full-population experiment, not release approval.
Frozen Exp5/Exp6 implementations, old reports and old run roots remain unchanged.

Preparation uses a new directory, copying only the exact 20,000 frozen inputs
and initial requests from the original Exp5 run. It verifies every payload/request
binding against that run, including the unchanged 1,000 development payloads,
and checks every initial request against the latest runtime's unchanged request
builder. Frozen message text is preserved, including serialized field order. No historical
answer or critic receipt is copied. Code, protocol, tests, inputs, requests and
sampling panel are hashed before any new model call.

The fixed 24-pair smoke panel includes P01329, P09549, P16061, P06727, P14187
and normal-path controls selected by historical execution mechanism. Sampling
reasons and old predictions are not sent to the Judge. Main and all routed
critic calls are fresh. The smoke pairs remain part of the same 20,000 pairs.

The gate requires 24 VALID pipeline results with original schema/evidence
validation, exact offline replay of every request and complete result, and
observed main, coverage, subject and verifier execution. Legitimate semantic
UNRESOLVED is reported separately and is not an engineering failure. The gate
does not demand agreement with old predictions or claim precision/recall accuracy.
Missing stage coverage or any terminal engineering failure stops continuation;
the failed panel and receipts remain immutable. No automatic prompt adjustment,
replacement sample or retry-until-pass is performed.

`prepare --root <new-root>` freezes the experiment. `launch --root <root>
--env-file <existing-env>` uses nohup with disconnected stdin and a new session,
logs to `recovery/sessions/<timestamp>/run.log`, runs the smoke panel, and only
after a passing gate continues the remaining population. `--smoke-only` stops
after the gate; a later ordinary launch resumes the same run. Only exact
same-run receipts can be reused after interruption. `status` is read-only.
No recurring assistant monitor or automatic notification is installed.

The original model, generation, two workers, two-second pacing, bounded retries,
600-second transport timeout, 429 waiting behavior and 120,000 external-attempt
ceiling are retained. The repetition patch allows one logical format recovery,
not unlimited native retries. The service's requested-versus-reported token
limit discrepancy remains undiagnosed. Checkpoint integrity violations and
transport circuit failures stop work without fabricating semantic decisions.

Completion requires all 20,000 results, including terminal failures, and exact
offline replay into a disposable directory without rewriting collected results.
The report separates engineering status, semantic unresolved, main retries,
anchor/evidence/repetition repairs and transport responses. Subsequent human
calibration uses the same frozen reference/exclusions; proxy agreement is not
accuracy and does not decide release. Independent holdout gates and the 75%
precision/recall targets are unchanged. Actual SUT outcomes remain separate;
MinHash diagnostics stay `UNAVAILABLE_MISSING_CONTRACT`.
