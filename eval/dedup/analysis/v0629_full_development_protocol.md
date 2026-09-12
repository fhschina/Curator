# V0.6.2.9 full-development diagnostic protocol

## Scope and authorization

The user explicitly requested a fresh V0.6.2.9 run on all 1,000 development pairs after checkpoint `cb66d670`.
This authorizes a diagnostic exception to the earlier residual-only progression restriction, not promotion, a release,
holdout access, or a 20,000-pair run. The prior residual failure remains recorded and is not retroactively passed.

## Frozen inputs and execution

- Reference: `v0628_policy_reconciled_labels_1000.csv`, SHA-256
  `6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096`.
- Diagnostic partition: the 127 IDs in `v0622_reconciled_residual_candidates.csv`, SHA-256
  `ab1ca4efcd6618abc702521486fd2b201631be18bb1b80ff712693a983dfadea`, and the remaining 873 development IDs.
- Policy: immutable `dedup-judge-hs-v0.6.2.9`, using its V0.6.2.6 main Judge, V0.6.2.8 record critic, and asymmetric
  arbitration. Prompt/resource hashes must match the saved V0.6.2.9 residual manifest.
- Execution: the existing Qwen inference-hub model, temperature 0, top-p 1, maximum output 4,096 tokens, visible budget
  20,000 tokens, concurrency 64, two 500-pair blocks, and the existing bounded retry policy.
- New root: `/raid/hfang/ihb/runs/v0.6.2.9-full-development-diagnostic`. No old Judge cache or predictions seed this run.
- The current checkpoint contains additional inactive version implementations, so the new implementation/contract digest
  is recorded independently. Verify the V0.6.2.9 resource hashes and replay its saved 127 raw outputs through the current
  V0.6.2.9 adapter before live inference; require identical published semantic outputs.

## Analysis fixed before results

Report weighted and unweighted precision, recall, primary-direction exactness, over-group, false containment, under-group,
unresolved, schema completion, terminal errors, sample-level retries, and empirical confidence-tier accuracy. Count unresolved
reference duplicates as recall misses. Report relation taxonomy separately; historical faithful-translation MINOR labels
do not determine version selection.

Partition primary disagreements into disjoint over-group, resolved under-group, unresolved, and wrong-direction categories.
Cross-tabulate errors by reference reason, predicted relation, span/critic branch, original 127 versus remaining 873, and
available track/language/length/truncation metadata. Inspect high-impact cases only after these aggregate tables are fixed.
Existing V0.6.1 and V0.6.2.2 outputs may be rescored on the same current labels for historical context, but changed payload
representations and labels preclude claiming a controlled equal-payload experiment or an independent human estimate.

No labels or prompts change in response to this run. The reconciled reference contains earlier prediction-aware AI reviews;
the 873-pair complement is less directly residual-tuned, not an untouched holdout. Any recommended next version is a proposal
only. This diagnostic generates no release approval regardless of the aggregate scores.
