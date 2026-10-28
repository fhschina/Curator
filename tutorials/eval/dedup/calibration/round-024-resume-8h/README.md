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
74 passed and five GPU tests deselected. Registered single-GPU checks and
new score confirmations remain incomplete until their actual receipts exist.

Raw inputs and HTTP bodies remain outside Git in
`/raid/hfang/retention_policy_eval/v1/experiments/round-024-resume-8h-20261028T181118Z`.
Old v1/v2/v3 evidence, source workspaces and reports remain unchanged.
