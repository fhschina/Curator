# Retention-v4 completed paired review

Run: `/raid/hfang/ihb/runs/critic-five-hour-20260911T070052Z/retention-v4-paired96`.
All 354 responses are valid HTTP 200, no retries. All 96 original material rows
remain in every cell. No main calls, reference change, holdout or release.

| Arm | Repeat | Weighted precision | Recall | Primary exact |
| --- | --- | --- | --- | --- |
| Fresh original .12 critic | 1 | 78.91% | 47.43% | 72.93% |
| Fresh original .12 critic | 2 | 74.21% | 43.99% | 70.59% |
| Immediate strict v3 | 1 | 64.81% | 85.66% | 72.60% |
| Immediate strict v3 | 2 | 60.99% | 85.66% | 69.40% |
| Strict v4 candidate | 1 | 61.97% | 85.66% | 72.28% |
| Strict v4 candidate | 2 | 60.82% | 85.66% | 71.27% |

These selected-material scores do not estimate full 1,000-case performance.
Both repeats fix all three targets, H0232/H0312 bare labels and H0907 failed object.
However H0606 regresses twice: the entity inside the actual liability exemption
is called a header. Immediate v3 correctly handles it twice. V4 is not expanded.
All other nineteen-guard members remain correct. H0741 reasoning improves but its
public direction cannot improve within the fixed-main, veto-only scope.

H0760 remains the sole candidate repeat disagreement and an unresolved policy question.
H0720 is a newly observed reference dispute: complete named blog description plus
a recipe, with only search/recommended headings unique to the smaller side. The old
page-role negative is retained, not automatically corrected or forced by new rules.
All cross-arm mismatches are covered in assessment-bound `review_complete.json`.
The cause-weight artifact in `retention-v4-paired96-causes` retains all disputed weights.

Next same-material trial replaces materiality order with complete-sentence binding
before label dismissal. No new output fields, schema change or source presentation change.
Full original development expansion remains conditional on reviewed target/guard outcomes.
