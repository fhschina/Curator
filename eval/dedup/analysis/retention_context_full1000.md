# Frozen exp3 full-development diagnosis

The user authorizes one fresh run of the original 1,000 pairs with the already
frozen v0.6.2.33-exp3 main, positive-direction critic, scoped subject and fixed
verifier, compared to the saved v0.6.2.12 and v0.6.2.33-exp1 final predictions.
This is diagnostic admission after the successful reviewed pilot, not release
approval. Keep known taxonomy/witness weaknesses and reference disputes visible;
do not repair them or change references during the run.

Use identical frozen original payloads, label CSV, weights, and exactly five
previously approved comparison exclusions. All 1,000 cases execute and enter
engineering accounting, including excluded ones; 995 enter semantic comparison.
No new exclusions, gold changes, synthetic cases, cherry-picking, population
intersections or merging old and new stage responses. Reports retain missing
outputs as failures; unresolved positives remain recall misses. Legacy taxonomy
is not scored as new semantic ground truth, including translated MINOR labels.

964 main requests are expected; 12 exact input cases are deterministic and 24
incomplete-evidence cases remain unresolved (13 truncated, 11 capped). Preflight
every complete main request with the unchanged tokenizer and context reserve.
A too-large request is recorded and remains an engineering failure, not silently
truncated or removed. Runtime stage checks remain unchanged. Do not expand the
evidence-window policy as part of this experiment.

Same Qwen3.8-27B NVIDIA endpoint/model, temperature 0, top_p 1, max_tokens 4096,
thinking disabled, portable schema transport, serial tokenizer initialization,
two workers, two-second admission interval, two HTTP attempts maximum, bounded
circuit breaker. Maximum 4,000 logical stage calls / 8,000 external attempts.
No semantic/schema repair retries. Every stage uses its own fresh valid upstream
result, and every request and raw response is persisted. One candidate repeat
only; no new online baseline calls. If the circuit opens, unavailable results
remain failures and no bypassing or resetting the circuit to hide the incident.

Before any model calls, reproduce both historical scores exactly with the existing
scorer. After completion, use that same scorer for weighted/unweighted P/R/F1,
primary agreement, errors and stage deltas. Produce all-case and corrected/regressed
ledgers against each baseline, with weights and reference provenance. Report
identity_slot, meaningful_addition and translation guards, all other historical
reason cohorts, observed HIGH/MEDIUM/LOW correctness, incomplete-input abstention,
engineering failures, actual calls, HTTP retries and byte-identical-pair consistency.
Do not treat successful validation of a failure sentinel as schema completion.

Retain weighted P/R >=75%, primary agreement >=79%, over-group <=66, protection
cohort noninferiority of -3pp, 100% valid pipeline results and <=1% judge retries
as development checks. Historical-relative comparisons are descriptive: service
time variation is not controlled, and the mixed user/assistant development
reference is not independent human gold. This is not a simultaneous three-arm
trial or a strict causal estimate of the prompt change.

No automatic prompt iteration, second full run, new holdout inference, 20k,
release, default switch, MinHash guess, credential change, git commit or push.
If a future reference revision is confirmed, rescore all three immutable outputs
under that same revision, not only the new candidate.
