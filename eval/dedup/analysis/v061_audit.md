# LLM Judge v0.6.1 audit

This audit freezes the evidence used to design v0.6.2. It separates human calibration from proxy challenge
diagnostics and does not treat agreement with another model as ground truth.

## Run reliability

The v0.6.1 run produced 20,000 valid results from 20,000 requested pairs, with no terminal errors. Fifty-one pairs
required at least one retry (0.255%), and one valid result remained semantically unresolved. The run used prompt
`dedup-judge-hs-minhash-v0.6.1`, output contract `dedup-judge-output-v2`, and contract digest
`9a5205bdd0e6593c1924ee490d69065718f6faadac17c2f5a7f3eaf2dd5ddbec`.

## Human blind calibration set

The calibration set contains 1,000 manually adjudicated pairs. Weighted estimates use each row's frozen
`stratum_population_n / stratum_sample_n` inverse-probability weight; unweighted estimates describe only the reviewed
sample. `Primary exact` means all three primary decisions match: duplicate group and both replacement directions.

| Version | Estimator | Primary exact | Duplicate precision | Duplicate recall | Duplicate F1 | Over-group | Under-group |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.6 | Unweighted | 73.60% | 85.57% | 53.09% | 65.52% | 29 | 153 |
| v0.6 | Weighted | 73.96% | 81.50% | 44.83% | 57.84% | — | — |
| v0.6.1 | Unweighted | 77.10% | 70.59% | 81.23% | 75.54% | 110 | 61 |
| v0.6.1 | Weighted | 77.48% | 65.71% | 76.66% | 70.76% | — | — |

v0.6.1 recovered much of v0.6's duplicate recall, but it crossed the desired operating point by admitting too many
false duplicate groups. Of its 110 over-group decisions, 102 were labeled `CONTAINMENT`; only eight were
`NEAR_SURFACE`.

## Error matrix

The following cohorts use the human error taxonomy. `Primary correct` again requires the complete three-field tuple;
over-group and under-group count group-decision errors within the cohort.

| Cohort | Pairs | Primary correct | Primary accuracy | Over-group | Under-group |
|---|---:|---:|---:|---:|---:|
| Human containment | 157 | 120 | 76.43% | 0 | 33 |
| Judge-predicted containment | 279 | 120 | 43.01% | 102 | 0 |
| Boilerplate-only | 91 | 23 | 25.27% | 59 | 7 |
| Chrome-only | 78 | 21 | 26.92% | 0 | 11 |
| Faithful translation | 54 | 45 | 83.33% | 0 | 6 |
| Page-role conflict | 94 | 73 | 77.66% | 21 | 0 |
| Truncated visible payload | 13 | 11 | 84.62% | 0 | 1 |
| Non-truncated visible payload | 987 | 760 | 77.00% | 110 | 60 |

The 279 predicted-containment pairs break down by human relation as follows: 123 containment, 69 unrelated, 54
near-surface, 24 related-non-duplicate, and nine version-related. Across all 229 primary-tuple errors, 159 were
predicted containment. The false-negative side is concentrated in meaningful additions (33 of 61), while the
false-positive side is concentrated in boilerplate-only (59 of 110) and page-role conflicts (21 of 110).

These results support a non-empty shared-main-content gate, a page-role/identity/state conflict veto, and an explicit
rule that chrome-only differences remain bidirectionally replaceable. They do not support globally suppressing
containment, because 120 of 157 true containment pairs already have the complete direction tuple correct.

## Proxy challenge set

The 14,336-pair conflict challenge contains 13,453 `CALIBRATED_CODEX_PROXY` labels (93.84%) and 883
`PRIOR_BLIND_REVIEW` labels (6.16%). It is useful for discovering failure shapes and stress-testing prompt changes, but
its aggregate score is not a human-calibrated release metric and must not decide the winning version.

## Confidence and MinHash findings

The v0.6.1 numeric confidence and evidence-quality axes were effectively saturated and therefore did not provide useful
calibration. v0.6.2 replaces the number with `HIGH/MEDIUM/LOW` and reports observed human accuracy separately for each
tier.

The v0.6.1 Judge also predicted a conventional MinHash action without possessing a resolved SUT MinHash contract. The
current SUT artifacts do not expose the exact normalization, shingling, hash family, band/row, and seed configuration.
Historical v0.6.1 results therefore remain semantic evidence only. v0.6.2 MinHash diagnostics are unavailable until the
real contract and SUT signatures are supplied; evaluation-retriever parameters must not substitute for them.

## v0.6.2 release baseline

The v0.6.1 weighted human-calibration values above are the baseline for development gates. Faithful translation is
normalized to semantic material difference `NONE` for v0.6.2 selection; historical human `MINOR` values remain intact
as legacy taxonomy and are not used to choose the release.
