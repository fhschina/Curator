# dedup-judge-hs-v0.6.2.10 residual audit

Residual gates: **FAIL**. A failed residual gate forbids full-development/holdout progression; passing is not release approval.

Error-enriched development reference, including prior prediction-aware AI adjudications; not an independent human holdout or population estimate.

| Metric | Baseline | Candidate |
|---|---:|---:|
| unweighted duplicate_precision | 45.95% | 40.00% |
| unweighted duplicate_recall | 80.95% | 57.14% |
| unweighted primary_decision_exact | 76.38% | 58.27% |
| unweighted taxonomy_exact | 51.97% | 44.88% |
| weighted duplicate_precision | 43.70% | 37.72% |
| weighted duplicate_recall | 77.65% | 60.48% |
| weighted primary_decision_exact | 76.39% | 59.73% |
| weighted taxonomy_exact | 53.36% | 46.34% |
| over_group | 20 | 18 |
| containment_over_group | 1 | 13 |
| under_group | 4 | 9 |

UNRESOLVED reference duplicates count as recall misses. Historical taxonomy is diagnostic only;
legacy translation MINOR is not a primary-selection criterion. Weighting does not remove residual-selection bias.

## Frozen gates

| Gate | Candidate |
|---|---|
| over_group | pass |
| containment_over_group | pass |
| under_group | FAIL |
| protected_duplicates | FAIL |
| protected_primary_tuples | FAIL |
| containment_direction_exact | FAIL |
| diagnostic_negatives | FAIL |
| schema_completion | pass |
| terminal_errors | pass |
| retry_rate | pass |

## Confidence calibration (candidate)

| Tier | Rows | Primary exact | Weighted primary exact |
|---|---:|---:|---:|
| LOW | 32 | 3.12% | 4.83% |
| MEDIUM | 95 | 76.84% | 75.97% |

These are empirical accuracies on this slice, not calibrated probabilities.

## Residual errors

- over_group: H0072, H0119, H0126, H0132, H0140, H0186, H0253, H0256, H0260, H0267, H0278, H0558, H0634, H0713, H0797, H0917, H0959, H0964
- under_group: H0038, H0333, H0361, H0488, H0653, H0679, H0748, H0811, H0878
- direction_only: H0394
- unresolved: H0038, H0053, H0073, H0189, H0195, H0221, H0239, H0276, H0333, H0345, H0384, H0406, H0480, H0488, H0599, H0653, H0668, H0679, H0702, H0723, H0733, H0751, H0770, H0828, H0856, H0867, H0878, H0882, H0907, H0928, H0954, H0979
- missing_results: none
- diagnostic negative violations: H0053, H0073, H0276, H0345

## Operations and provenance

- baseline: 127/127 valid, 0 terminal errors, 0 retried pairs; `/raid/hfang/ihb/runs/v0.6.2.9-residual-smoke`.
- candidate: 127/127 valid, 0 terminal errors, 0 retried pairs; `/raid/hfang/ihb/runs/v0.6.2.10-residual-smoke`.

The companion JSON records per-review primary tuples, cohort/relation matrices, boundary outcomes, all gates
and source/result/evaluator digests. No holdout or full-population results are claimed.
