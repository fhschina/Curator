# Exp6 service-unavailability continuation

Opt-in entry point: `python -m eval.dedup.analysis.exp6_service_recovery`.
Semantic runtime stays `v0.6.2.33-exp6+main-repetition-fix2`. The separate
transport contract is `dedup-service-unavailable-wait-v1`. Original full-run,
authentication-recovery sources, manifests, requests, responses and results
remain immutable; the transport log is only appended to.

HTTP 500/502/503/504, upstream connection/timeouts and the previously validated
exact database-busy 401 receive the existing two inner transport attempts.
If both fail, the relay releases the request as a WAIT control message, starts
a shared service cooldown, and the collector resubmits the identical request
after 60/120/240/300/300 seconds. Retry-After is never shortened. Each stage
allows at most six service-failing submissions per execution session (at most
12 inner attempts if all failures are service failures), and at most 1,800
seconds of service backoff. Individual in-flight calls keep their original
640-second deadline and 600-second timeout. The backoff budget is not a claim
that the entire stage including inference time finishes in 30 minutes.

429 retains a separate bounded wait allowance: at most 60 rate-limited
submissions and 21,600 seconds of rate-limit backoff per stage/session. All
external attempts share the original full-run 120,000 ceiling, two workers
and two-second pacing. Repeated 5xx failures do not open a permanent circuit.
True credential errors, 403s, invalid requests, global budget exhaustion and
ambiguous local relay failures stop further collection; they are not retried
as semantic corrections. Ordinary successful responses are unchanged.

WAIT controls never reach the Judge parser. Exhausted waits raise a control
signal outside normal semantic retry handling, leaving the pair pending with
its request, successful earlier stage receipts and transport events saved.
Shared cancellation wakes other waiting workers; requests arriving after
cancellation stay pending instead of creating ENGINEERING_FAILURE results.
Already in-flight successful answers may still be saved, never discarded to
select another answer. A local relay timeout has an unknown outcome and stops
pending rather than immediately issuing an overlapping duplicate request.

`prepare --root <original-run>` freezes the new sources, existing artifacts,
current transport prefix, remaining pair IDs and P19310/P19312 recovery scope.
`recover-case --root <run> --target <new-dir> --review-id P19310|P19312
--env-file <existing-env>` runs an isolated, fully audited recovery with its
own 32-external-attempt ceiling; it never replaces the old failure.
`launch --root <run> --env-file <existing-env>` uses nohup to collect only
unfinished pairs, then runs the original full offline audit and a separate
transport completion report. No assistant monitor or notification is installed.

Judge prompts, critics, model, key, generation, semantic retry rules, reference
labels and release gates are unchanged. Five already recorded engineering
failures remain in the original full-run report; separately recovered cases
must be reported as recoveries, not silently backfilled into a clean benchmark.
