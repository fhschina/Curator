# Final overnight retention delivery

**Final metric target not reached; acceptance did not pass. Delivery was late.**

The delivered nine prompt/config files exactly restore `candidate-06-languagefirst1`.
Original personal runs: 205/240 directions and 96/120 pairs; 206/240 and 97/120;
201/240 and 95/120. The confirmed stage is 612/720 (85%) and 288/360 (80%).
The 90%/85% target is short by 36 directions and 18 pairs.

Compared with the same-condition fixed baseline, this adds 93 directions and
46 pairs; compared with the starting version, 19 directions and 7 pairs.
All three 100-input reference views meet their fixed baseline. Original36,
extra6 and real consensus6 pass; latest18, controlled26 and full-input technical
validity fail. No candidate is eligible or accepted.

`report_en.html` contains the readable English final report.
`report_tables.json` contains its English tables. The original Chinese report,
bilingual tables and historical translation verification are preserved in the
local archive identified by `../../english-publication.json`. `strict/` contains the formal checked
comparison; `native/` preserves the original comparison. Both have complete
compressed JSON/HTML, CSV, per-version run/summary JSON and provenance hashes.
`../run-evidence/` contains the per-layer row audits, including failures.
`../frozen-candidates/` and `../candidate-diffs/` preserve all 22 frozen versions.
Incomplete and failed candidates are retained, never replaced by resampling.

C10 completed all original runs with 612/720 directions and 287/360 pairs:
it missed stage confirmation by one pair. C11 and C13 pass their first fixed
hard screen but retain supplemental format failures and lack full confirmation.
`final-selection.json` explains the frozen validity-first selection.

Actual checks in this round include 74 tutorial CPU tests, 25 native-stage CPU
tests, 32 workflow CPU tests and 12 shared-audit CPU tests. These are separate
sets, totaling 143; repeated executions do not increase the count. The selected
frozen version passed two template/window checks and four registered single-GPU
tests with no skips. The final packaging reuses those exact-hash results and
starts no new inference after the absolute deadline. Hook and hash receipts are
recorded separately.

The owned service stopped at 2026-10-28 08:44:47 PDT, before its 08:45 cutoff.
Allocation was 12.685 GPU hours including idle and startup. The fixed delivery
deadline was 09:00 PDT; post-deadline recovery began at 10:39 PDT. Delivery is
late, and the observed execution gap has no verified cause. No deadline was
extended. See `delivery-timing.json` and `ALL_RESOURCES_RELEASED.json`.

Raw corpora and complete HTTP exchanges remain outside Git in the original
session. Its external `DELIVERY_RECEIPT.json` links the final commit, remote
verification, content-addressed evidence archive and restore validation. The
archive includes final committed source, frozen source materials and historical
D074 failures alongside later replies. D121 remains pending and unscored.

For future reproduction, restore the bundle and follow [REPRODUCE.md](../REPRODUCE.md)
using the existing pinned environment/model/image and an explicitly new deadline.
The restore and prepare checks make no model calls and do not establish a full
inference replay. These are exposed development-reference results, not
independent human accuracy or a measured corpus deletion rate.
