# v0.6.2.33-exp3 clean contextual-conflict pilot

This immutable successor to exp2 includes its separately audited proof1 evidence
validation and serial-tokenizer startup repair. The new semantic change separates
unique additions from contextual incompatibility of shared assertions. Rename
loss fields to addition fields and conflict to context_conflict in the model
contract, system prompts, pair instructions and YAML rubric together. Shared
assertions may conflict even with NONE additions and no unique span on one side.
Conflict proof must still contain exact bilateral source evidence and at least
one selected unique modifier/value; severity remains MAJOR. Explanation text is
never parsed into a replacement decision. No raw legacy output is reinterpreted
as the new model contract. The public output schema remains V4.

Freeze the identical 48 cases from exp2: 36 original diagnostics and 12 synthetic
regressions. References, seven title-case expectations, twelve synthetic primary
expectations and the five comparison-only exclusions are unchanged. Synthetics
are prompt-informed, assistant-authored development checks, not independent gold.
No cases are removed to make this pilot pass. H0130/H0344 remain reference-pending.

Run baseline and candidate twice, clean: every active main and routed dependent
stage receives its own fresh response. Preserve all old directories and receipts.
Baseline main request bodies/digests are copied unchanged; its critic, subject and
fixed verifier remain unchanged. Candidate main/critic use the new addition/context
contract; candidate subject scope and verifier stay unchanged. Use the same Qwen
27B endpoint/model, temperature 0, top_p 1, max_tokens 4096, disabled thinking and
portable structured output. Preserve schema property order by the same frozen-main
and dynamic-dependent request construction. Do not add chain-of-thought routing.

Use the same complete-span-only presentation. Both arms keep 13 truncated and
11 capped packets explicitly unresolved without inference; one exact synthetic
bypasses deterministically. There are 23 active cases per arm/repeat, hence 92
fresh main calls if startup and context checks succeed. Serial tokenizer warmup
must finish before worker creation in a cold run process. Same two workers,
two-second paced admission, two HTTP attempts maximum, bounded circuit breaker,
maximum 768 logical calls / 1536 external attempts. No semantic-repair retries.
All 192 case/arm/repeat results stay in engineering accounting.

Predeclared gates are unchanged: candidate must get all 7 title cases and all
12 synthetic primary decisions right in both repeats, with zero engineering
failures, no repeat-primary changes and no normalized-primary disagreement among
the observed exact-input groups. Report auxiliary taxonomy errors separately.
Review every other disagreement or observed residual before expansion. A gate
pass is not an automatic 1,000-pair admission. A failure must remain in the report;
do not silently revise prompts and replace the failed responses in this root.

This is a focused package test, not a causal estimate separating the proof/startup
fixes from the semantic change (proof1 already has a zero-call offline audit).
It provides no population precision/recall estimate or independent validation,
does not relax the 75% requirements, and never launches 20k or a release.
