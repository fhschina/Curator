# Dedup Evaluation Results

These results were generated from saved artifacts only; new model calls: **0**.
The validated run is bound to Judge contract
`bdd2cf795479a7a6a9d225ff51c7f29f4cf8c1089bab30c75a65546814a530b6`.

## Executive summary

The Judge completed **20,000/20,000 valid outputs** and replayed all **24,105**
saved model calls identically offline. On the shared development reference it
reaches weighted precision **81.82%**, recall **80.04%**, and
primary-decision agreement **88.88%**.

The Judge is the recommended automated development reference for this
evaluation. It is not independent human ground truth, and independent holdout
validation remains pending.

## Development-reference results

The reference contains 1,000 pairs. Six fixed comparison-only exclusions leave
**994 scored pairs**. This is development data, not an independent human gold
set.

| Scoring | Judge | Precision | Recall | Primary-decision agreement |
|---|---|---:|---:|---:|
| Weighted | Version 0 | 60.40% | 40.24% | 80.49% |
| Weighted | Version 0.5 | 33.40% | 86.75% | 52.64% |
| Weighted | Current | 81.82% | 80.04% | 88.88% |
| Unweighted | Version 0 | 55.97% | 53.54% | 74.45% |
| Unweighted | Version 0.5 | 34.93% | 74.80% | 48.49% |
| Unweighted | Current | 84.80% | 83.46% | 88.23% |

| Weighted delta | vs Version 0 | vs Version 0.5 |
|---|---:|---:|
| Precision | +21.42 pp | +48.42 pp |
| Recall | +39.80 pp | -6.71 pp |
| Primary-decision agreement | +8.39 pp | +36.24 pp |

### Development error accounting

| Category | Pairs |
|---|---:|
| CORRECT | 877 |
| OVER_GROUP | 38 |
| UNDER_GROUP_RESOLVED | 36 |
| DIRECTION_ONLY | 13 |
| UNRESOLVED | 30 |

## Full 20K engineering and decision accounting

| Measure | Result |
|---|---:|
| Valid outputs | 20,000 / 20,000 |
| Semantic unresolved | 513 |
| Main-retry pairs | 644 |
| Saved calls replayed | 24,105 |
| External attempts | 24,118 |
| Anchor-ID repairs | 7 |
| Repetition repairs | 3 |
| Coverage-evidence repairs | 0 |

The 513 unresolved decisions are valid fail-closed semantic outcomes, not
engineering failures. All 20,000 output records are schema-valid.

The Hub and local-Qwen backends both passed the frozen 24-pair smoke panel,
including the main, coverage, subject, and verifier paths.

## SUT diagnostics under each Judge

These values diagnose the unchanged fuzzy-dedup system under different Judge
configurations; they do not measure Judge accuracy. Track 5a and Track 5b have
different sampling frames and must not be pooled.

| Metric | Version 0 | Version 0.5 | Current |
|---|---:|---:|---:|
| 5a Judge-rated safe removals / resolved | 59.39% | 71.76% | 58.85% |
| 5a Judge-rated wrong removals / resolved | 40.61% | 28.24% | 41.15% |
| 5b Judge-rated duplicates / resolved candidates | 2.29% | 8.89% | 5.41% |

For the current Judge, Track 5a has 5,796 safe and 4,052 wrong removals across
9,848 resolved pairs. Track 5b has 521 positives across 9,639 resolved selected
candidates. Track 5b yield is not corpus recall.

## Interpretation

- The complete run validates execution completeness, schema compliance, and
  offline replayability.
- The development reference supports comparisons among the listed Judge
  configurations, but it does not become independent ground truth.
- The independent holdout remains a separate quality gate.
- Semantic `UNRESOLVED` is an intentional fail-closed answer, not a transport
  or parsing failure.
- Track 5a measures the safety of sampled removal decisions.
- Track 5b measures positive yield within a selected candidate pool, not
  corpus-wide recall.

## Methodological limitations

- The LLM Judge is the automated reference for these metrics; it is not human
  ground truth.
- Track 5a contains sampled SUT removals, so removal precision is identifiable;
  recall, specificity, and a complete SUT confusion matrix are not.
- Track 5b has zero inclusion probability for unseen pairs, so its positive
  yield cannot be interpreted as corpus recall.
- The partial judged constraint graph cannot support corpus-level cluster
  precision, recall, or F1.
- Track 5a and Track 5b are never pooled into a single confusion matrix.
- The disagreement-enriched QA packet is diagnostic and must not be pooled with
  the blind QA packet for unweighted accuracy estimates.

## Reproduce or inspect

See the [evaluation README](README.md) for installation, Hub and local backend
commands, smoke-gate execution, status monitoring, and offline audit.

The internal dashboards provide pair-level inspection:

- [Pair Explorer](http://umb-b200-218.cl1u1.colossus.nvidia.com:18750/dedup-dashboard/)
- [Track 5a removal review queue](http://umb-b200-218.cl1u1.colossus.nvidia.com:18750/dedup-dashboard/pair_explorer_v07.html?track=5a)
- [Track 5b cross-group review queue](http://umb-b200-218.cl1u1.colossus.nvidia.com:18750/dedup-dashboard/pair_explorer_v07.html?track=5b)

These links require access to the NVIDIA internal network.

To verify a completed run without making model calls:

```bash
python -m eval.dedup audit --root /path/to/completed-run
```

Pair-level documents, prompts, responses, and evidence are intentionally not
embedded in this report. Generate or serve the Pair Explorer from an authorized
run root when pair-level inspection is required.
