# .32 transport-only full-development diagnostic

This is a new, prospective execution contract, `dedup-paced-development-v1`;
the semantic version remains `dedup-judge-hs-v0.6.2.32`. The old .32 run stays
stopped under its own pure-HTTP-200 gate. Its results and cached outputs are not reused.
The user requested another evaluation on the original dataset; this protocol
authorizes development measurement, not promotion, release, holdout access or 20,000 pairs.

## Fixed inputs and execution

- Original 1,000 payloads and .8 reconciled reference/weights retain their recorded SHA-256 hashes.
  No gold labels, sampling reasons or historical predictions enter model requests.
  These references include prediction-aware AI adjudication; they are not a new human blind review.
- Same .32 V5 prompts, semantic adapter, routing, evidence validation, model,
  temperature 0, top-p 1, 4096 output tokens, native concurrency 16, two outer
  correction retries. The critic sees only this run's accepted fresh main.
- Shared transport admission spaces actual upstream starts by at least 2 seconds,
  permits at most 4 in-flight requests, and honors server Retry-After across stages.
  Retry only 429/502/503/504 or network failures: at most 4 attempts per logical request,
  exponential 10/20/40 second fallback, 60 second fallback cap, 480 second deadline.
  A larger Retry-After is never shortened. Global external-attempt budget: 5,000.
  Exhaustion/nonretryable failures open a circuit; no credential or request text in receipts.
- The new **client** context guard is 65,536 including 4096 output plus 2048 reserve.
  It is not a claim about the service's resolved context configuration. The old 32,768
  guard excludes 15 complete inputs. The longest actual .32 request must succeed against
  the endpoint before proceeding. Payloads are not shortened or retokenized into new text.
  Critic and correction inputs are checked again at actual submission.

## Frozen schedule and failure accounting

1. Eleven existing probes, freshly called. All responses must be schema/evidence valid.
   Semantic accuracy and model corrections are reported but are not promotion gates for
   this newly authorized diagnostic. Recovered HTTP 429 is not a model error.
2. The longest original development input, selected by input token count only;
   fresh main and applicable critic must bind. Its prediction is not reused below.
3. The existing 24-real/26-synthetic panel twice, same labels, order and settings.
4. One fresh pass over all original 1,000 inputs, in original order. No mid-run tuning.

Technical preflight failure, transport exhaustion or provenance/boundary failure stops
the schedule. Ordinary terminal model schema/evidence failures after bounded retries in
formal panels/full development are retained and counted, never repaired in code.
An incomplete main pair is not submitted to critic. Its terminal ID remains explicit.
Missing outputs receive **scoring-only** UNRESOLVED sentinels, never fake valid Judge
records: all 1,000 remain in the metric denominator; positive-reference failures count
as recall misses. Unsubmitted pairs after a fatal stop cannot produce a full-set score.

## Report and unchanged acceptance requirements

Report main-only and same-main-plus-critic weighted/unweighted precision, recall,
primary exact, error distributions, reason cohorts and empirical confidence-tier accuracy.
Separate owned routes, accepted strict outputs, terminal schema/evidence failures,
native/outer model corrections and all actual upstream attempts/HTTP statuses.
Saved original requests/responses are immutable and accepted results must replay exactly.

Full-development targets remain weighted precision >=75%, recall >=75%, primary exact
>=79%, over-group <=66, zero terminal errors and model-correction pair rate <=1%.
Identity, meaningful-addition and translation accuracy may not drop >3 points against .9;
historical translation MINOR is taxonomy only. Old references and scores remain intact.
No release is approved even if development targets pass: independent holdout is still required.
The deliberately selected panels and longest-input preflight are not representative estimates.
