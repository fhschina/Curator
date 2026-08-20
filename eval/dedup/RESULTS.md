# NeMo Curator Fuzzy Dedup Evaluation — Latest Results

This page is the repository-visible snapshot of the latest formal V0 evaluation. It summarizes the automated results without
publishing run-local paths, document excerpts, or the generated artifact inventory. The immutable report and its supporting
artifacts remain in the evaluation run bundle.

## Reference run

| Field | Value |
| --- | --- |
| Evaluation run | `dedup-full-20260813T220949Z-d4c37bb483` |
| Report artifact | `final_report.automated_v8.md` |
| Report version | `dedup-automated-report-v3` |
| Generated | 2026-08-20 18:43 UTC |
| Source report SHA-256 | `70373c948b66d8f9a32be47602b55dc6e9308114549466991926a911ba119416` |

## Executive summary

The evaluation benchmarked a frozen fuzzy-deduplication system along two separate tracks:

- **Track 5a — removal safety:** whether the selected keeper can safely replace a document marked for removal.
- **Track 5b — cross-group discovery:** whether selected candidates from different predicted groups are likely duplicates.

The automated Judge returned schema-valid results for **19,994 of 20,000 pairs (99.97%)**.

| Headline metric | Result | Scope |
| --- | --- | --- |
| Removal precision | **59.39%** (5,935 / 9,993) | Uniform sample of actual removal decisions |
| Wrong-removal rate | **40.61%** (4,058 / 9,993) | Uniform sample of actual removal decisions |
| Cross-group positive yield | **2.29%** (229 / 9,997) | Selected cross-group candidate pool |

The removal-precision estimate has a Wilson 95% confidence interval of **58.43%–60.35%**. The automated results therefore
indicate substantial over-removal in the sampled removal frame.

## Where removal quality drops

The largest degradation appears when paired documents differ substantially in length. Long documents, cross-host pairs, and
large predicted groups also show lower removal precision.

| Slice | Removal precision | Safe / resolved |
| --- | --- | --- |
| Token-length ratio 0.5–0.8 | **8.15%** | 65 / 798 |
| Long documents | **42.38%** | 359 / 847 |
| Different hostnames | **46.89%** | 1,901 / 4,054 |
| Predicted group size 21+ | **51.93%** | 2,406 / 4,633 |

These slices are the clearest starting points for additional guardrails and focused calibration of the fuzzy-deduplication
system.

## Cross-group discovery

The selected cross-group pool contained **229 Judge-positive pairs**. Under the frozen per-anchor source quotas, candidates
surfaced by both retrieval channels had the highest yield, while semantic-only retrieval contributed the most positives.

| Selected retrieval source | Positive yield | Positives / resolved |
| --- | --- | --- |
| Lexical and semantic overlap | **9.45%** | 75 / 794 |
| Semantic only | **2.64%** | 130 / 4,924 |
| Lexical only | **0.56%** | 24 / 4,279 |

These source counts are shaped by the frozen selection quotas. They support investigating retrieval-budget allocation, but they
are not an unqualified comparison of global retriever quality.

## Human QA

Human QA is an independent calibration layer and is not included in the automated metrics above. The current review packet
contains:

- a **200-pair blind sample** for the primary human–Judge agreement estimate; and
- a separate **200-pair diagnostic set** concentrated on difficult disagreements for debugging and regression analysis.

The blind and diagnostic sets must remain separate and must not be combined into one headline metric.

## How to inspect the results

- Use the [Pair Explorer](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/) to inspect automated wrong
  removals, discovered cross-group positives, Judge evidence, provenance availability, and group context.
- Use the [Human QA Dashboard](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/human-qa/) to review the blind
  and diagnostic samples. These dashboards require access to the NVIDIA internal network.
- See the [evaluation README](README.md) for the methodology, runtime profiles, evaluation contract, commands, and artifact map.

## Interpretation boundaries

- The automated Judge is a reference, not human ground truth.
- Track 5a estimates removal precision on its sampled removal-decision frame; it does not identify removal recall, specificity,
  or a full confusion matrix.
- Track 5b reports positive yield in the selected candidate pool, not corpus recall.
- Track 5a and Track 5b use different sampling frames and must not be pooled.
- The partial judged graph does not support corpus-level cluster precision, recall, or F1.
