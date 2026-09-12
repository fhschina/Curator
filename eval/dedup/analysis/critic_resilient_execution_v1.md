# Recover the incomplete ordering trial without changing its semantics

The first ordering trial had 74 actual upstream calls: 73 HTTP 200 and one HTTP 500.
The old transport did not consider HTTP 500 retryable and opened a global circuit.
The remaining 280 local HTTP 400 receipts never reached NVIDIA. All 64 affected or
reference-mismatched case IDs are reviewed; no unseen judgments are inferred.
One actually returned candidate proof selected A001 for B's own-side loss field and
was correctly rejected. This schema/semantic failure is not a retryable transport error.
The original incomplete run remains untouched and is not used for semantic ranking.

New immutable transport-v2 adds only HTTP 500 to bounded retries already used for
429/502/503/504. Keep two attempts, backoff, pacing, two concurrent requests, total
upstream budget and hard deadline. HTTP 400/401 still open the circuit immediately;
repeated server failures still stop after the same bounded budget. Local integration
tests cover 500-success, 500-exhaustion, authentication, order preservation and counts.

Fresh root, same 354 frozen logical requests and all 96 original rows, both repeats,
all three arms. Both input artifacts must be byte-identical to the failed run.
No cache/output reuse, semantic prompt edit, field rename, dynamic schema constraints,
label correction or schema-answer retries are included. Outcome/schema failures remain
scored as missing outputs, and all errors require review before the next experiment.
Actual generation order and semantic accuracy remain separate. No full-development
or release promotion from a partial or selected run. Deadline remains 12:00:52 UTC,
with admission stopped 210 seconds earlier.
