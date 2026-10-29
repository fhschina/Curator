# Eight-hour continuation toward 90% / 85%

This is a new, independently timed continuation from commit
`8831f46fb233b86df6a056394217b8eb3e2c0667`, whose delivered policy is
candidate-06-languagefirst1. Its previous-round stage was confirmed at
612/720 directions and 288/360 pairs. That historical result is preserved;
no previous inference output is counted as a new-round result.

The fixed eight-hour window is 2026-10-28 11:11:18–19:11:18 PDT
(18:11:18 UTC through 2026-10-29 02:11:18 UTC). Semantic search ends
at 17:41:18 PDT. The owned single-GPU service stops by 18:56:18 PDT.
Recovery never resets the clock.

Read `INSTRUCTIONS.txt` for the complete authorized task and frozen criteria.
The target is one frozen candidate's original three complete runs totaling
at least 648/720 directions and 306/360 pairs. Full acceptance separately
requires all hard regressions, native technical validity, and every fixed
100-input nonregression view. No per-run target gate, best-of sampling,
voting, changed denominator or failed-run replacement is allowed.

`frozen-start85/` preserves the exact eight Jinja files and one YAML file of
the evaluated starting version. The new baseline and starting comparator each
have 15 fresh original slots declared before inference. The fixed historical
baseline retains its original policy with the shared native format adapter.
The previous39714b47 comparator remains historical and is not relabeled as
new-round evidence. Per-layer raw failures and incomplete rows remain visible.

Only the existing environment, pinned Qwen/model/image profile, original
Data Designer chain and one B200 are used. New actual CPU checks report
74 passed and five GPU tests deselected; the exact selected starting profile also
passed the registered single-GPU group with four actual tests. Supplemental
original repetitions completed, but their policy gate failed. Fresh-round
starting results are 608/720 directions and 287/360 pairs; neither target nor
full acceptance passed. Complete descendants scored 602/720 and 285/360, and
599/720 and 282/360. The retained best policy and metric-best policy are both
the starting profile. Historical 612/720 and 288/360 remain separate.

Raw inputs and HTTP bodies remain outside Git in
`/raid/hfang/retention_policy_eval/v1/experiments/round-024-resume-8h-20261028T181118Z`.
Old v1/v2/v3 evidence, source workspaces and reports remain unchanged.

`replay_frozen_v4.py` verifies the native source and frozen inputs, then prepares
or executes the original three repeats for the fixed baseline, this round's
starting version and the selected candidate. When start and best are identical,
their shared frozen version is evaluated once and the two roles remain explicit.
Use the existing Python environment and supply a new explicit future deadline,
checkout, endpoint and caller-owned container. `--prepare-only` makes no model
calls; it is preparation evidence, never a claim of completed replay. The fixed
strict audit sources are loaded byte-for-byte from the source session included
in the external evidence bundle, or an explicitly supplied `--audit-tools` path.

Final selection follows the frozen validity-first ranking. Supplemental controls have
separate fixed expectations and denominators; critic corrections do not excuse
main errors. All failed candidates and the unsuccessful coverage response-order
experiment remain in the frozen archives and per-layer row evidence.

For existing-output verification after restoring the external evidence bundle,
use `replay_restored_evidence.py --session RESTORED_SESSION --checkout RESTORED_SESSION/delivery-source/Curator --best-root candidate-00-start85 --output offline-replay`.
Run it with the existing pinned Python environment. The restored session supplies
the frozen audit sources. This entry checks source hashes, redirects recorded
absolute checksum paths to restored files, and blocks original-session/source
checkout reads and network connections in the native child processes. It also
recomputes the selected supplemental groups when their recorded summary exists.
It makes no new model call and preserves failed outcomes. The post-source-commit
`archive-and-restore-receipt.json` records the actual final TAR extraction,
independent source restoration and row-for-row replay outcome.

Final existing-evidence delivery is indexed in `delivery/final-selection.json` and the paired reports. The external session `DELIVERY_RECEIPT.json` records the actual commit, remote verification and cleanup. No unrecorded check is treated as passed.
