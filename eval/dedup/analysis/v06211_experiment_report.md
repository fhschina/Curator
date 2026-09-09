# V0.6.2.11 experiment report

## Decision

V0.6.2.11 is implemented and tested, but **rejected for progression**. The policy-only ablation also fails the frozen
final gates. Neither candidate may advance to the 1,000-row development set, holdout, or 20,000-pair run. V0.6.2.9 remains
the comparison baseline, not an approved release. No defaults, reference labels or historical caches were replaced.

The protocol and thresholds were fixed before live inference in [v06211_design.md](v06211_design.md). Both live runs use
the same 127 residual IDs, visible payload membership, model, temperature, token budget and concurrency as V0.6.2.9.
The main Judge and its rubric are unchanged; the legacy record-binding rubric is retained as the first critic score.
The policy-only and final candidates have separate prompt versions, run roots and contract/cache digests.

This is an error-enriched, repeatedly examined development reference containing earlier prediction-aware AI adjudications.
Its weighted metrics are not population estimates or independent human-blind accuracy. Historical translation MINOR is
not used for primary selection; unresolved reference duplicates count as recall misses.

## Live results

| Metric | V0.6.2.9 | V0.6.2.11 policy-only | V0.6.2.11 final |
|---|---:|---:|---:|
| weighted precision | 43.70% | 36.18% | 31.37% |
| weighted recall | 77.65% | 50.37% | 42.70% |
| weighted primary exact | 76.39% | 73.76% | 71.76% |
| primary exact count | 97/127 | 94/127 | 90/127 |
| over-group | 20 | 17 | 18 |
| false containment | 1 | 4 | 4 |
| under-group | 4 | 10 | 12 |
| unresolved | 5 | 6 | 7 |
| protected primary tuples | 4/5 | 2/5 | 2/5 |
| true-containment primary tuples | 3/4 | 4/4 | 2/4 |
| diagnostic negatives resolved NO | 22/23 | 23/23 | 23/23 |

Both runs completed 127/127 schema-valid results, zero terminal errors and zero **pair-level outer retries**. The policy
run has 481 byte-exact evidence excerpts; the final run has 477. All outputs were validated against v3, all unresolved
results were LOW, and truncated inputs were never HIGH. Final-run exact replay reproduces every public output field.

Provider event logs contain 258 completed runtime requests plus 20 pilot requests per candidate, all HTTP 200. These are
reported separately from outer pair retries: event counts alone do not classify every framework-internal extra request
as a retry. A zero outer-retry count is not a claim of zero internal model retries/corrections.

Full numerical audits, including frozen gate outcomes, error IDs and digests:

- [Policy-only audit](v06211_policy_residual_smoke.md), [JSON](v06211_policy_residual_smoke_summary.json).
- [Final audit](v06211_residual_smoke.md), [JSON](v06211_residual_smoke_summary.json).

## What the scoped repair did establish

Before live runs, V0.6.2.10's recorded outputs were replayed through the proposed adapter without new LLM calls.
Unresolved fell 32 -> 4; weighted primary exact improved 59.73% -> 72.15%; under-group fell 9 -> 4. However, over-group
rose 18 -> 26 and false containment stayed at 13. This isolated the abstention problem but did not establish a deployable
semantic improvement. The replay is explicitly marked ineligible for promotion in [its artifact](v06211_offline_replay.json).

The live policy-only run resolved H0572 and H0723 as no/no and preserved all 23 diagnostic negatives. Its four true
containment directions were also correct. H0017 was already correct without the translation supplement, so that success
cannot be attributed to the later translation prompt.

## Fixed-output ablation: where the final candidate loses decisions

To distinguish deterministic rule effects from live model variation, the final run's SAME raw main and critic outputs
were adapted with three policies. No LLM was called and no source output/cache was overwritten.

| Adapter on the same final raw outputs | weighted precision | weighted recall | weighted primary exact | over | under |
|---|---:|---:|---:|---:|---:|
| final V0.6.2.11 | 31.37% | 42.70% | 71.76% | 18 | 12 |
| translation supplement disabled | 35.03% | 50.37% | 73.05% | 18 | 10 |
| original V0.6.2.9 arbitration, both supplements disabled | 41.34% | 84.72% | 75.90% | 23 | 3 |

False containment is four in all three replays. Thus disabling supplements does not solve atomic-record errors or pass
the final gates. These counterfactuals are diagnostics, not additional live candidates. Their output deltas, metrics,
source/adapter hashes and relevant raw reasoning are in [v06211_fixed_output_ablation.json](v06211_fixed_output_ablation.json).

Disabling only translation changes exactly three primary tuples:

- H0017: unresolved -> correct no/yes containment.
- H0653: unresolved -> correct no/yes containment.
- H0221: unresolved -> no/no.

For H0017 and H0653 the main Judge AND legacy critic already establish the right direction with bilateral citations.
The supplemental translation score also says B_ADDS, but cites only the added B span inside that specific field. The
adapter consequently rejects it as missing an A-side/bilateral citation. This is a remaining **ownership design flaw**:
an active but redundant confirmation was treated as mandatory new proof. It is not a semantic translation error by the
model, and evidence should not be silently fabricated to conceal it.

## Remaining semantic/protocol failures

1. **Consent UI is still mistaken for changed policy meaning.** H0347/H0453 are rejected after the critic infers an
   implicit-versus-explicit consent-state change from browsing wording and an Accept control. Related equivalent-cookie
   cases H0089, H0214, H0226, H0604 and H0792 also become misses. Against the frozen reference, these are over-conservative
   semantic decisions, not merely missing citations. Disabling policy removes seven duplicate misses but introduces five
   additional over-groups on the same raw outputs: blanket removal of the policy gate is not justified.
2. **The non-main axis remains overloaded for chrome corroboration.** Both live critics correctly describe H0748's teaser
   as benign, yet mark the policy subtype NOT_APPLICABLE because it is not a policy change. The adapter requires a focused
   equivalent-wrapper claim to flatten the main's false containment, so the error remains. NOT_APPLICABLE correctly stays
   inactive; it cannot serve as proof that every delta is chrome. The contract needs a clearer distinction between a
   policy assessment and a whole-delta chrome proof.
3. **Record granularity still drifts.** The final false containments are H0132, H0253, H0267 and H0278. Company/store context,
   collection descriptions and related reusable material still sometimes substitute for the bilateral identity of the
   actual added article/product/list. The old <=15 false-containment gate would obscure this; the new <=1 gate rejects it.

No prompts or adapter rules were changed after the live candidates were frozen. The above findings explain rejection;
they were not used to rewrite labels or make a post-hoc passing version under the same name.

## Next bounded change, not implemented in this version

First distinguish **decision-changing claims** from **redundant confirmations**: retain an independently validated
main/legacy direction when a supplemental field merely repeats it with incomplete citations; preserve the citation issue
for audit. A proposed direction change or material conflict still needs its own sufficient proof and fails closed without
it. Existing evidence ownership must be explicit, not silently borrowed between rationales.

Then test the policy veto separately against actual changed propositions/purposes/permissions and UI-only variants.
Do not infer a changed retained policy solely from a button or navigation label. A whole-delta chrome proof should have
an unambiguous scope independent of whether a policy exists. Finally, require record identity at the product/article/list
level where appropriate, while protecting genuine additions to the same organization profile. Any such changes need new
immutable experiments; this iteration provides no release approval.

## Verification

- `pytest -q tests/eval/dedup`: **270 passed**.
- Ruff lint/format checks and `git diff --check` passed. Full pre-commit hooks remain unavailable because pre-commit is
  not installed in the existing environment; no environment was created or synchronized.
- V0.6.2.9 and V0.6.2.10 prompt resources match their recorded hashes. Frozen label/subset hashes are unchanged.
- The task-owned Ray process was stopped after both runs and native summaries completed.
- No full development run, holdout access, label change, historical cache migration or full-population execution occurred.
  Native SUT/DeepSeek comparisons are diagnostics only. No MinHash replay was claimed without a resolved SUT contract.
