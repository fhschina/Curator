# Exp6: bounded database-busy 401 recovery

This opt-in transport extension leaves the original Exp6 full-20K manifest,
Judge runtime, main/critic prompts, schema, model, generation, historical
requests, receipts and results unchanged. The same original API key is used.
The semantic version remains `v0.6.2.33-exp6+main-repetition-fix2`; transport
provenance is separately identified as `dedup-auth-database-busy-retry-v1`.

Only HTTP 401 with the structured `auth_error` / code `401` and the exact
observed connector/database message ending `FATAL: sorry, too many clients
already` is retryable (case/whitespace normalization only). It shares the
existing 5xx budget: at most two total upstream attempts per submission,
10-second initial backoff, Retry-After respected, unchanged deadline and
global attempt budget. Real invalid-key, generic or ambiguous 401 and all
403 errors still stop immediately. A repeated database-busy failure exhausts
the existing budget and opens the circuit. No unbounded retry or semantic
feedback change is introduced. Raw HTTP 401 remains visible in transport logs;
it is not relabeled as 200/429/503. Existing 429 handoff stays compatible.

Before continuation, one tiny health check must succeed on the same model and
endpoint using the existing key. The patch manifest binds that check, all
already collected request/response/result files and the original transport-log
prefix. Original successes AND failures remain immutable. The continuation
reuses the original pending-pair collector, rate-limit waits and full audit,
with an in-process transport override that restores the original class on exit.
New sources and a separate completion report make the transport change explicit.

`prepare --root <original-run>` freezes the extension. `launch --root <run>
--env-file <existing-env>` uses nohup and collects only unfinished pairs. No
assistant monitoring or notification is installed. The original completion
report still includes previously saved failures; it is not a zero-error rerun.

`recover-case --root <run> --target <new-directory> --env-file <existing-env>`
independently reissues the affected initial main request and unchanged routed
stages, with a 32-attempt external ceiling. It cannot overwrite the original
P18614 failure or select between multiple successful outputs. This isolated
recovery has its own manifest, receipts, replay audit and result, and is not
silently merged into the original 20K. P14546 and P16717 remain separate
non-authentication issues, outside this transport patch.

Precision/recall, reference exclusions, independent holdout and release gates
are unchanged. The health probe and isolated recovery are diagnostic calls,
reported separately from the resumed original full-population experiment.
