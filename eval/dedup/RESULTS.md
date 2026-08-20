# NeMo Curator Dedup Evaluation V0

**FORMAL V0 — AUTOMATED JUDGE RESULTS**

Report version: **dedup-automated-report-v4**

## 1. Executive Summary

The automated pipeline completed 20,000 pair evaluations with a schema-valid completion rate of
**99.97%** (19,994/20,000). The sampled removal
frame produced **59.39% removal precision** with a Wilson 95% confidence interval
of **58.43%-60.35%**. The complementary wrong-removal rate was
**40.61%**.

The selected Step 5b cross-group candidate pool produced **2.29% positive yield**
(229/9,997 resolved candidates). This is candidate-pool yield, not
corpus recall.

| Headline metric | Value | Numerator / denominator | Authorized scope |
| --- | --- | --- | --- |
| Judge completion | 99.97% | 19,994 / 20,000 | All selected pairs |
| Removal precision | 59.39% | 5,935 / 9,993 | Uniform Step 5a removal frame |
| Wrong-removal rate | 40.61% | 4,058 / 9,993 | Uniform Step 5a removal frame |
| Candidate-pool positive yield | 2.29% | 229 / 9,997 | Selected Step 5b candidate pool |
| Terminal Judge errors | 6 | 6 / 20,000 | Operational accounting |
| Unresolved valid results | 4 | 4 / 19,994 | Operational accounting |

Human QA is an independent double-check and does not block, replace, calibrate, or rewrite these automated metrics.
The exports include a Judge-independent blind packet of 200 pairs and a separate
disagreement-enriched diagnostic packet of 200 pairs. Both expose
only the Judge-visible document payload to reviewers. The diagnostic packet excludes pairs already present in the
blind packet and must not be used to estimate overall prevalence or pooled with the blind packet for unweighted
accuracy metrics.

The [interactive Pair Explorer](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/) contains a filterable review queue of wrong removals, discovered
cross-group positives, unresolved/errors, and report control examples. It uses Judge-visible excerpts and evidence;
its labels remain automated Judge results rather than human ground truth.

## 2. Evaluation Scope and Authorized Claims

- Evaluation run: **dedup-full-20260813T220949Z-d4c37bb483**
- Dataset: **CC-MAIN-2025-26-dense-10m-v0**, 10,008,061 documents.
- SUT run: **d8a4d79265ef88d11ebd**
- Judge: **nvidia/deepseek-ai/deepseek-v4-pro**, prompt **dedup-judge-v0**
- Exact deduplication was an upstream precondition and was not rerun.
- Step 5a supports claims about the sampled removal-decision frame.
- Step 5b supports claims about the selected retriever candidate pool only.
- Prohibited claims: corpus-wide recall, complete cluster precision/recall/F1, pooled track 5a and track 5b confusion matrix.

## 3. Pipeline Accounting

<pre>
10,008,061 corpus documents
├── 7,986,841 singleton documents
└── 2,021,220 grouped documents
    ├── 352,601 logical group keepers
    └── 1,668,619 removals

1,668,619 SUT removal decisions
└── 10,000 uniformly sampled Step 5a keeper-to-removed pairs

1,000 Step 4 anchors (Step 5b only)
├── 27,463 relaxed-lexical anchor-candidate records
├── 50,000 semantic anchor-candidate records
└── 73,363 cross-channel union records
    └── 10,000 selected Step 5b cross-group pairs

20,000 Judge pairs
├── 19,994 schema-valid
│   ├── 19,990 resolved
│   └── 4 unresolved
└── 6 terminal errors
</pre>

Detailed stage-level timing, execution accounting, and Judge operational diagnostics are retained in the full report
appendices.

## 4. Step 5a — Removal Decision Quality

This section reports Track 5a only: the safety of actual SUT keeper-to-removed decisions sampled from the removal
frame. Removal precision is safe resolved removals divided by all resolved sampled removals. The selection was
uniform from a frame of 1,668,619 removals with inclusion probability
0.00599298.

| SUT decision | Judge safe | Judge wrong | Unresolved | Error | Total |
| --- | --- | --- | --- | --- | --- |
| REMOVE | 5935 | 4058 | 4 | 3 | 10000 |

There is no SUT-negative sampling frame, so this table is an outcome matrix rather than a full confusion matrix.
Removal recall, specificity, and F1 are not identifiable in V0.

### By predicted group size

| Slice | Selected | Valid | Resolved | Safe | Wrong | Precision | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| size_2 | 1085 | 1085 | 1084 | 709 | 375 | 65.41% | 62.52%-68.18% |
| size_21_plus | 4635 | 4634 | 4633 | 2406 | 2227 | 51.93% | 50.49%-53.37% |
| size_3_5 | 1801 | 1800 | 1799 | 1170 | 629 | 65.04% | 62.80%-67.21% |
| size_6_20 | 2479 | 2478 | 2477 | 1650 | 827 | 66.61% | 64.73%-68.44% |

### By document length

| Slice | Selected | Valid | Resolved | Safe | Wrong | Precision | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | 849 | 849 | 847 | 359 | 488 | 42.38% | 39.10%-45.74% |
| medium | 2136 | 2135 | 2135 | 992 | 1143 | 46.46% | 44.36%-48.58% |
| short | 7015 | 7013 | 7011 | 4584 | 2427 | 65.38% | 64.26%-66.49% |

### By token length ratio

| Slice | Selected | Valid | Resolved | Safe | Wrong | Precision | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-0.25 | 4 | 4 | 4 | 1 | 3 | 25.00% | 4.56%-69.94% |
| 0.25-0.5 | 143 | 143 | 143 | 36 | 107 | 25.17% | 18.77%-32.87% |
| 0.5-0.8 | 798 | 798 | 798 | 65 | 733 | 8.15% | 6.44%-10.25% |
| 0.8-1.0 | 9055 | 9052 | 9048 | 5833 | 3215 | 64.47% | 63.48%-65.45% |

### By hostname relationship

| Slice | Selected | Valid | Resolved | Safe | Wrong | Precision | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | 4056 | 4055 | 4054 | 1901 | 2153 | 46.89% | 45.36%-48.43% |
| True | 5944 | 5942 | 5939 | 4034 | 1905 | 67.92% | 66.73%-69.10% |

### By judged relation type

| Slice | Selected | Valid | Resolved | Safe | Wrong | Precision | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CANONICAL_EXACT | 13 | 13 | 13 | 13 | 0 | 100.00% | 77.19%-100.00% |
| CONTAINMENT | 284 | 284 | 284 | 125 | 159 | 44.01% | 38.36%-49.83% |
| EXACT | 4917 | 4917 | 4917 | 4917 | 0 | 100.00% | 99.92%-100.00% |
| NEAR_SURFACE | 921 | 921 | 921 | 827 | 94 | 89.79% | 87.67%-91.59% |
| None | 3 | 0 | 0 | 0 | 0 | N/A | N/A |
| RELATED_NON_DUPLICATE | 1892 | 1892 | 1892 | 0 | 1892 | 0.00% | 0.00%-0.20% |
| UNRELATED | 1750 | 1750 | 1750 | 0 | 1750 | 0.00% | 0.00%-0.22% |
| UNRESOLVED | 4 | 4 | 0 | 0 | 0 | N/A | N/A |
| VERSION_RELATED | 216 | 216 | 216 | 53 | 163 | 24.54% | 19.28%-30.69% |

### By material difference

| Slice | Selected | Valid | Resolved | Safe | Wrong | Precision | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAJOR | 3972 | 3972 | 3972 | 11 | 3961 | 0.28% | 0.15%-0.50% |
| MINOR | 1098 | 1098 | 1098 | 1002 | 96 | 91.26% | 89.44%-92.79% |
| NONE | 4923 | 4923 | 4923 | 4922 | 1 | 99.98% | 99.89%-100.00% |
| None | 3 | 0 | 0 | 0 | 0 | N/A | N/A |
| UNRESOLVED | 4 | 4 | 0 | 0 | 0 | N/A | N/A |

### By fuzzy scope

| Slice | Selected | Valid | Resolved | Safe | Wrong | Precision | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BORDERLINE | 144 | 144 | 144 | 1 | 143 | 0.69% | 0.12%-3.83% |
| IN_SCOPE | 7509 | 7509 | 7509 | 5934 | 1575 | 79.03% | 78.09%-79.93% |
| None | 3 | 0 | 0 | 0 | 0 | N/A | N/A |
| OUT_OF_SCOPE | 2340 | 2340 | 2340 | 0 | 2340 | 0.00% | 0.00%-0.16% |
| UNRESOLVED | 4 | 4 | 0 | 0 | 0 | N/A | N/A |

### Inspect removal decisions

Use the [Track 5a removal review queue](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/?track=5a) to inspect wrong removals and filter by document
length, token-length ratio, hostname relationship, group size, relation type, material difference, or reason code.

## 5. Step 5b — Cross-group Retrieval Analysis

This section reports Track 5b only: missed-duplicate discovery within the selected anchor-based cross-group candidate
pool. It does not estimate corpus recall.

### Candidate generation and selection

Track 5b starts from the Step 4 anchors and uses two parallel retrieval channels: a more-permissive
lexical MinHash/LSH configuration and semantic embedding top-k retrieval. The lexical channel was tuned by candidate
volume rather than by changing one scalar similarity threshold: the pilot selected 7 bands x
1 row per band, producing a median of
31 cross-group candidates per pilot anchor against the frozen target
range of 20-50. The upstream SUT resolved configuration was unavailable, so this report does not claim a numeric SUT-to-evaluation threshold change such as 0.8 to 0.7.

| Step 5b stage | Count | Interpretation |
| --- | --- | --- |
| Step 4 anchors | 1000 | Queries used by Track 5b only |
| Relaxed lexical candidates | 27463 | Cross-group MinHash/LSH results, ranked by exact lexical features |
| Semantic candidates | 50000 | Cross-group embedding neighbors with top-k=50 |
| Cross-channel union | 73363 | Anchor-candidate records after merging lexical and semantic membership |
| Selected unique pairs | 10000 | Per-anchor source quotas, deterministic refill, and global pair deduplication |
| Resolved Judge results | 9997 | Denominator for candidate-pool positive yield |

The selected source counts below are shaped by the frozen per-anchor quotas (up to four lexical-only, four
semantic-only, and two both-channel candidates, followed by deterministic refill). They describe the selected
candidate pool; they are not natural corpus prevalence, channel recall, or an unqualified head-to-head retriever
comparison.

### Positive yield by selected retrieval source

The pool contained 10,000 selected candidates, of which 9,997 were
resolved. Positive yield is Judge duplicate-YES divided by resolved selected candidates.

| Slice | Selected | Valid | Resolved | Duplicate YES | Duplicate NO | Yield | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| both_or_overlap | 794 | 794 | 794 | 75 | 719 | 9.45% | 7.60%-11.68% |
| lexical_only | 4279 | 4279 | 4279 | 24 | 4255 | 0.56% | 0.38%-0.83% |
| semantic_only | 4927 | 4924 | 4924 | 130 | 4794 | 2.64% | 2.23%-3.13% |

### By document length

| Slice | Selected | Valid | Resolved | Duplicate YES | Duplicate NO | Yield | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | 1401 | 1400 | 1400 | 14 | 1386 | 1.00% | 0.60%-1.67% |
| medium | 3659 | 3659 | 3659 | 72 | 3587 | 1.97% | 1.57%-2.47% |
| short | 4940 | 4938 | 4938 | 143 | 4795 | 2.90% | 2.46%-3.40% |

### By token length ratio

| Slice | Selected | Valid | Resolved | Duplicate YES | Duplicate NO | Yield | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-0.25 | 1873 | 1873 | 1873 | 6 | 1867 | 0.32% | 0.15%-0.70% |
| 0.25-0.5 | 2540 | 2540 | 2540 | 19 | 2521 | 0.75% | 0.48%-1.17% |
| 0.5-0.8 | 3286 | 3283 | 3283 | 65 | 3218 | 1.98% | 1.56%-2.52% |
| 0.8-1.0 | 2301 | 2301 | 2301 | 139 | 2162 | 6.04% | 5.14%-7.09% |

### By hostname relationship

| Slice | Selected | Valid | Resolved | Duplicate YES | Duplicate NO | Yield | Wilson 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | 8619 | 8616 | 8616 | 179 | 8437 | 2.08% | 1.80%-2.40% |
| True | 1381 | 1381 | 1381 | 50 | 1331 | 3.62% | 2.76%-4.74% |

### Inspect cross-group candidates

Use the [Track 5b cross-group review queue](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/?track=5b) to inspect discovered positives and compare
retrieval sources, Judge evidence, document context, and available SUT provenance.

## 6. Methodological Limitations

- The LLM Judge is the automated reference for these metrics; it is not human ground truth.
- Step 5a contains only sampled SUT removals, so removal precision is identifiable but recall, specificity, and a full SUT confusion matrix are not.
- Step 5b is a selected cross-group candidate pool with zero inclusion probability for unseen pairs. Its positive yield is not corpus recall.
- The partial judged constraint graph is not complete ground truth and cannot support corpus-level cluster precision, recall, or F1.
- Track 5a and Track 5b have different sampling frames and are never pooled into one confusion matrix.
- The disagreement-enriched human QA diagnostic packet is not a prevalence sample and is never pooled with the blind QA packet.
- Upstream provenance is conditionally reproducible because resolved config and several retrieval attestations were not delivered.


## 7. How to Inspect the Results

- Use the [Pair Explorer](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/) to inspect automated wrong removals, discovered cross-group positives,
  Judge evidence, provenance availability, and group context.
- Use the [Human QA Dashboard](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/human-qa/) to review the blind and diagnostic samples. These dashboards
  require access to the NVIDIA internal network.
- See the [evaluation README](https://github.com/fhschina/Curator/blob/dedup-eval/eval/dedup/README.md) for the methodology, runtime profiles, evaluation contract,
  commands, and artifact map.

The report body intentionally excludes document excerpts and pair identifiers. Use the Pair Explorer for pair-level
review.
