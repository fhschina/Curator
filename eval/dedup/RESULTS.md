# Current validated result

The selected Judge v0.7 release was validated on the frozen 20,000-pair bundle:

| Check | Result |
|---|---:|
| Schema-valid pairs | 20,000 / 20,000 |
| Saved calls replayed identically | 24,105 |
| Precision | 81.82% |
| Recall | 80.04% |
| Primary agreement | 88.88% |

The original contract digest begins with `bdd2cf` and remains an accepted legacy
audit identity. Dedup Eval v0.7.1 records its own tool version and source digest
while retaining `judge_contract_version: v0.7` and the original release lineage.

These are the only published current results in the main tree. Detailed ledgers,
raw responses, dashboards, and superseded experiment reports belong in the
external archive.
