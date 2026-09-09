# dedup-judge-hs-v0.6.2.11 residual audit

Residual gates: **FAIL**. A failed residual gate forbids full-development/holdout progression; passing is not release approval.

Error-enriched development reference, including prior prediction-aware AI adjudications; not an independent human holdout or population estimate.

| Metric | Baseline | Candidate |
|---|---:|---:|
| unweighted duplicate_precision | 45.95% | 33.33% |
| unweighted duplicate_recall | 80.95% | 42.86% |
| unweighted primary_decision_exact | 76.38% | 70.87% |
| unweighted taxonomy_exact | 51.97% | 50.39% |
| weighted duplicate_precision | 43.70% | 31.37% |
| weighted duplicate_recall | 77.65% | 42.70% |
| weighted primary_decision_exact | 76.39% | 71.76% |
| weighted taxonomy_exact | 53.36% | 52.31% |
| over_group | 20 | 18 |
| containment_over_group | 1 | 4 |
| under_group | 4 | 12 |

UNRESOLVED reference duplicates count as recall misses. Historical taxonomy is diagnostic only;
legacy translation MINOR is not a primary-selection criterion. Weighting does not remove residual-selection bias.

## Frozen gates

| Gate | Candidate |
|---|---|
| over_group | pass |
| containment_over_group | FAIL |
| under_group | FAIL |
| protected_duplicates | FAIL |
| protected_primary_tuples | FAIL |
| containment_direction_exact | FAIL |
| diagnostic_negatives | pass |
| schema_completion | pass |
| terminal_errors | pass |
| retry_rate | pass |
| unresolved | FAIL |
| weighted_duplicate_precision_nonregression | FAIL |
| weighted_duplicate_recall_nonregression | FAIL |
| weighted_primary_decision_exact_nonregression | FAIL |

## Confidence calibration (candidate)

| Tier | Rows | Primary exact | Weighted primary exact |
|---|---:|---:|---:|
| LOW | 7 | 14.29% | 23.07% |
| MEDIUM | 120 | 74.17% | 74.20% |

These are empirical accuracies on this slice, not calibrated probabilities.

## Residual errors

- over_group: H0056, H0132, H0186, H0189, H0253, H0267, H0278, H0384, H0668, H0702, H0713, H0733, H0797, H0825, H0856, H0907, H0917, H0928
- under_group: H0017, H0089, H0214, H0226, H0347, H0361, H0453, H0488, H0604, H0653, H0792, H0811
- direction_only: H0394, H0741, H0748
- unresolved: H0017, H0221, H0239, H0612, H0653, H0688, H0751
- missing_results: none
- diagnostic negative violations: none

## Operations and provenance

- baseline: 127/127 valid, 0 terminal errors, 0 retried pairs; `/raid/hfang/ihb/runs/v0.6.2.9-residual-smoke`.
- candidate: 127/127 valid, 0 terminal errors, 0 retried pairs; `/raid/hfang/ihb/runs/v0.6.2.11-residual-smoke`.

The companion JSON records per-review primary tuples, cohort/relation matrices, boundary outcomes, all gates
and source/result/evaluator digests. No holdout or full-population results are claimed.
