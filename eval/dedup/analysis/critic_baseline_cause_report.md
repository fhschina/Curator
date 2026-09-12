# Original .12 baseline: current-policy cause accounting

This is prediction-aware AI cause attribution, not independent gold or a new score.
All 1000 original rows and weights remain; .12 final still scores weighted precision
67.9854%, recall 75.0669%, primary agreement 87.7812%. Its 129 mismatches comprise
57 over-group, 38 resolved under-group, 29 unresolved and five direction-only cases.

Current-policy review separates 12 clearer group false positives (weight116.361)
from 45 reference/policy-disputed or pending group false positives (weight430.323).
The latter carry 78.72% of FP weight. A clear omission in a replacement direction
does not establish that the group must be negative; several one-way policy-content
extensions were previously scored as unequivocal group errors under old rules.

At fixed true-positive weight, reaching precision75 requires removing FP weight
159.709. Removing only the 12 currently clear group FPs would yield precision72.9569.
This conditional calculation is not a repaired-model score. Recovering true positives
also changes the denominator, so precision improvement cannot be reduced to tightening.

Contributions to the total losses (100-P and 100-R), in percentage points:

| Reviewed cause | Precision loss | Recall loss |
| --- | ---: | ---: |
| Clear semantic / approved-policy error | 6.8143 | 13.9588 |
| Reference dispute or pending interpretation | 25.2003 | 9.7368 |
| Engineering | 0 | 1.1081 |
| Input limits | 0 | 0.1293 |

These are not additive repair gains or a decomposition of only the gap below75.
Unresolved and FN weights overlap; they must not be added twice. No disputed example
is removed from scoring. Cases and source-review links are preserved in
`/raid/hfang/ihb/runs/critic-five-hour-20260911T070052Z/baseline-current-policy-causes-v1.json`.
