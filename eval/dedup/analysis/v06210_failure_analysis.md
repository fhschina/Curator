# V0.6.2.10 failure analysis and handoff

## Decision

Freeze V0.6.2.10 as a rejected development experiment. It may not advance to the full 1,000-row development set,
holdout, or 20,000-pair run. V0.6.2.9 remains a comparison baseline, not an approved release.
The reproducible numerical audit is [v06210_residual_smoke.md](v06210_residual_smoke.md) and its companion JSON.

Both versions were evaluated against the same unchanged reconciled development reference. Eight previously incorrect
primary tuples became correct, but 31 previously correct tuples regressed: 97/127 -> 74/127 exact. Weighted primary
exact fell 76.39% -> 59.73%; precision 43.70% -> 37.72%; recall 77.65% -> 60.48%. These are error-enriched slice
diagnostics, not population estimates or independent human-blind accuracy.

The new run completed 127/127 schema-valid pairs, with zero pair retries and zero terminal errors. All 377 emitted
evidence excerpts were checked against the original integer-offset visible payloads and matched byte-for-byte.
All unresolved results were LOW confidence; truncated inputs were never HIGH. Schema completion is not semantic coverage:
32/127 results were unresolved, versus 5/127 in V0.6.2.9.

## What was fixed, and what was not

| Case | V0.6.2.9 | V0.6.2.10 | Interpretation |
|---|---|---|---|
| H0572, cookie category/consent expansion | yes/yes | no/no | Correct policy separation. |
| H0017, additive Arabic organization description | unresolved | no/yes containment | Correct bilingual extra-service direction. |
| H0723, added cookie entry | false containment | unresolved | Inventory subtype was recognized, but missing bilateral citations prevented a resolved veto; not a completed fix. |
| H0748, trailing story teaser | false containment | no/no | Still wrong; generic teaser example did not establish the intended chrome boundary. |
| H0333, application instructions | correct containment | unresolved | Spurious translation branch and missing B-only citation. |
| H0653, translated service with extra contact instruction | correct containment | unresolved | Translation direction was correct, but the generic binding score cited only B010. |
| H0878, FAQ troubleshooting instruction | correct containment | unresolved | Correct addition in the same language was incorrectly routed through translation review. |

Only two of the five protected benign duplicates retained their exact primary tuples. True-containment direction
coverage fell from 3/4 to 1/4. The original diagnostic negative guard fell from 22/23 to 19/23: H0053, H0073, H0276,
and H0345 became unresolved, not false-positive duplicates. The guard explicitly requires resolved NO.

## Failure mechanisms

1. **Applicability and direction are conflated.** The critic returned A_ADDS/B_ADDS for same-language FAQ, cookie or
   forum-permission additions (for example H0878, H0073, H0276). It performed generic fact comparison inside the translation
   axis instead of first deciding whether this axis applies. The adapter correctly refused to treat those as confirmed
   translations, but the resulting abstentions erased otherwise correct decisions.
2. **Evidence failure has excessive scope.** Any critic-axis citation issue is checked before the valid main-negative or
   specialized policy branches. Ten main no/no decisions were turned unresolved by a citation issue in this global gate.
   H0653 had a correctly cited bilingual B_ADDS score, yet an uncited bilateral basis in the separate generic record score
   invalidated the whole result. Independent claims need independent validation, but inactive or redundant claims should
   not automatically invalidate a decisive, valid rejection.
3. **Citation syntax and applicability are underspecified.** The model sometimes wrote ranges such as A001-A009, while
   the parser recognizes individual IDs only; four chrome scores failed exhaustive coverage. Eight decisions became
   unresolved because the main called both sides NON_MAIN_ONLY while the critic used NOT_APPLICABLE instead of
   EQUIVALENT_MESSAGE_OR_WRAPPER, including harmless blog-template/payment navigation cases. This is a protocol mismatch,
   not evidence that these benign pairs acquired material differences.
4. **Record granularity still drifts to site/company identity.** H0119's store shell plus product listings, H0126's store
   policies plus a skate product, H0132's company description plus a service article, and H0959's recurring event/context
   plus legal-case updates were called atomic extensions. Their reasonings explicitly bind to shared site/company/context,
   not a bilateral identity for the unique product/article/list record. The critic returned ATOMIC_SAME_RECORD_EXTENSION
   53 times (versus six previously), and final false containments rose from one to 13. The existing <=15 smoke threshold
   technically passes this regression; it is too loose for the next iteration and must not be interpreted as improvement.

Of the 32 unresolved results, 21 came from the global citation gate, eight from non-main scope disagreement, one from
translation scope disagreement, and two from unresolved main/payload evidence. Thus neither prompt accuracy nor
abstention inflation can be inferred from schema completion alone.

## Offline diagnostic: removing the critic is not a fix

An adapter-only replay of each run's recorded main scores against its original visible payloads, without a critic or any
new LLM calls, produced the following. These are counterfactual diagnostics, not new version predictions or release evidence.

| Main-only replay | V0.6.2.9 raw main | V0.6.2.10 raw main |
|---|---:|---:|
| over-group | 40 | 39 |
| containment over-group | 18 | 23 |
| under-group | 3 | 3 |
| weighted precision | 29.53% | 29.27% |
| weighted recall | 84.72% | 84.72% |
| weighted primary exact | 62.02% | 62.72% |

The old critic supplied important discrimination. Simply disabling the new citation checks or critic would recover
recall while exposing many false positives. Because both prompts and arbitration changed together in V0.6.2.10,
the live experiment cannot separately attribute the effect to a particular prompt edit or to model variation.

## Next experiment, not implemented into this frozen version

- Retain the V0.6.2.9 record-binding formulation as the control. Test narrow applicability/validation changes in separate
  immutable candidates before changing all semantic axes together.
- Gate translation direction on actual cross-language alignment of the same retained record. Same-language additions
  and localized chrome should follow ordinary record/delta rules.
- Validate evidence for the active decision branch. Preserve a valid main hard conflict or material-negative decision
  when an unrelated critic axis fails; keep unsupported positive upgrades fail-closed.
- Use an explicit machine-readable citation list, or validated range expansion over known supplied IDs. Never infer
  missing IDs or accept invented ranges. Test missing-own-side evidence separately from harmless citation syntax.
- Require the shared subject to identify the unique product/article/job/list record, not merely its site or company.
  Keep a distinct counterexample for real same-organization service additions so the H0017 fix is not lost.
- Freeze tighter non-regression limits for false containment and primary exact before the next run. Do not alter this
  candidate's already evaluated gates, label reference, or cache to make it pass.

## Verification and provenance

`pytest -q tests/eval/dedup`: **237 passed**. Ruff lint and formatting checks passed on all twelve changed Python/test
files; `git diff --check` passed. Full pre-commit hooks could not run because pre-commit is absent from the existing
environments; no environment was created or synchronized. The task-owned Ray process was stopped after completion.

The V0.6.2.9 runner and all four referenced historical prompts match their frozen manifest hashes. The candidate has a
different contract digest and independent run/cache. Label and subset hashes still match the design's frozen values.
The run-root native DeepSeek agreement and SUT outcome reports are diagnostics only and were not used for selection.
No MinHash replay was claimed without a resolved SUT contract. No holdout was read and no labels were edited.
