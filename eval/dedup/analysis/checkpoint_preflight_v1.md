# .33-exp1 preflight and isolated repair protocol

Status: engineering/policy diagnostic, not a release or an online-run admission.
The existing `.33-exp1` checkpoint, historical references, five-case comparison
mask, prompts, schemas, runtime defaults and caches remain immutable.

## Audit first

Verify source and artifact hashes, replay all 1,000 main-adapter outputs and all
1,000 final outputs, and validate the full V3 results and exact evidence offsets.
Schema completion includes unresolved results; report semantic resolution
separately. The five comparison-only exclusions stay in every engineering check.

Attribute all unresolved cases by the *visible evidence actually rendered*, not
by the availability of text elsewhere on disk. Separate truncated input, capped
span inventory, invalid citations and auxiliary-ledger uncertainty. Raw direction
scores are diagnostic evidence, not validated fallback decisions. Report weighted
error mass, without presenting an oracle correction as achieved accuracy.

Scan the full population for byte-identical document pairs, also allowing A/B
reversal. Normalize directions when comparing stages. Do not normalize whitespace
or use approximate strings for this audit. Record actual payload and rendered
initial pair-prompt digests; do not infer identical transport/retry histories or
assert a stochastic root cause from repeated text alone.

## Repair A: policy/contract alignment, proposed .33-exp2

This is a proposed experiment name, not a registered prompt or finished candidate.
Before new inference, create an isolated versioned implementation. Do not mutate
V3 to make historical artifacts appear to have used the new policy.

The semantic delta is the user-confirmed complete-retention rule: nonempty proper
subsets allow only the complete side to replace the subset, including policy-only,
independent additional articles and short title/navigation additions. Incompatible
context, negated quotations, changed identities within tokens, mutual losses and
incomplete evidence are not established containment. No unconditional substring or
length override is allowed. Complete identity and faithful translation remain
bidirectional; formatting alone is not newly retained content.

Separate replacement direction from semantic difference severity. A truthful
non-main/minor addition must not be coerced to a major main-content addition.
Design the new output contract and compatibility reader together, including the
mapping of containment, equivalent text, genuine state conflict and two-sided
uncovered content. Validate bilateral exact evidence and direction consistency.
Update system, pair, YAML descriptions, adapter and critic output projection
together. Retain the narrowed critic's role: positive-direction loss checks and
verified concrete subject conflicts, not automatic vetoes for content profiles.

Use the same saved raw outputs for an offline adapter-only comparison where the
new required evidence is available. Missing new-contract fields must remain
explicitly unsupported; do not invent semantic evidence or combine partial
replays into a complete candidate score. A separate main-prompt change needs
fresh responses. Keep input-length/packet repairs out of this first contrast.

Freeze a version-selection panel *before* new responses: include the diagnostic
targets and source-reviewed protections for existing true containment, distinct
SKU/subject/controller, substantive policy changes, both-side losses, translation,
negated quotations, suffix identity changes, whitespace and empty/truncated input.
The audit's mechanism panel alone is not a representative sample or a complete
protection panel. Label provenance and any disputed inherited references remain
visible to analysts, never to the model; no score-driven relabeling.

## Repair B: evidence availability, a separate contrast

For the span-capped cases, first measure the actual rendered input/token budget.
Consider bounded full-context or coverage-complete evidence routing with exact
locators. Do not merely raise every limit or treat a capped packet as COMPLETE.
For truncated inputs, preserve uncertainty until complete retention can actually
be supported. For invalid citations, use bounded, request-bound repair feedback
and distinguish engineering repair from a semantic rejudgment. Auxiliary-ledger
uncertainty must be reported accurately; do not silently translate it to no/no.

## Online expansion gates

1. Policy/contract synthetic checks pass; original formats remain readable;
   frozen history is unchanged. The diagnostic and protection panel is fixed.
2. A small fresh paired pilot uses the same model and generation settings for
   the baseline and candidate. Record every main/critic/verifier response and
   stage routing, retries, token limits and repeated/orientation consistency.
   Preserve all sampled cases, including comparison-excluded ones.
3. Only after reviewed pilot/protection success, run the complete original 1,000
   pairs with fresh main, coverage and any routed subject/verifier outputs in a
   new run root/cache. Clearly distinguish baseline `.33-exp1` from any `.33-exp2`
   behavior. A fixed-upstream critic run is not end-to-end reproduction.
4. Retain the user's development gates: weighted precision and recall >=75%,
   primary agreement >=79%, 100% schema completion, judge retry rate <=1%, and
   the previously agreed over-group/protection requirements. Report unresolved
   counts and each confidence tier separately, with uncertainty and provenance.
5. Only then prepare and freeze a genuinely unused 400-case holdout (200
   representative, 200 challenge). Check source/document overlap, not merely pair
   IDs. Preserve blind review and the earlier 50+50 double-review/arbitration
   arrangement. Assistant review is not a second independent human reviewer.
6. View the final candidate holdout once under the previously agreed thresholds.
   Failure makes it development data and requires a fresh subsequent holdout.
   Run 20k only after the gate passes. Semantic evaluation, proxy challenge, SUT
   outcome and optional MinHash replay remain separate. Missing actual resolved
   SUT configuration still means UNAVAILABLE_MISSING_CONTRACT.

No API credential change, online expense, reference revision, default-version
switch, release, git commit or push is performed by the preflight command.
