# NeMo Curator Fuzzy Dedup Evaluation

## Introduction

This package implements a reproducible, ten-stage evaluation of a completed fuzzy-deduplication system over the frozen
CC-MAIN-2025-26 10M handoff. It evaluates existing SUT outputs; it does not rerun exact deduplication or physically remove rows.

It answers four practical questions:

- Were keeper-to-removed decisions supported by the pair evidence?
- How often do sampled cross-group candidates appear to be missed duplicates?
- Which languages, group sizes, and relation types account for failures?
- Where do the SUT, fuzzy-dedup outcome, LLM Judge, and human reviewers disagree?

The automated benchmark uses a schema-constrained LLM Judge as its reference, not as human ground truth. Human QA is an
independent calibration layer: the random blind sample supports the primary agreement estimate, while the disagreement-focused
diagnostic set is a separate challenge set for regression analysis and debugging. Do not combine them into one headline metric.

The design source is the [V0.3 operational proposal](docs/proposals/v0.3/NeMo_Curator_Dedup_Evaluation_V0_3_Operational_Proposal.docx).
Earlier revisions are in the [proposal archive](docs/proposals/archive/). The latest technical presentation is
[NeMo Curator Dedup Evaluation — Current Status](docs/presentations/NeMo_Curator_Dedup_Evaluation_Current_Status.pptx).

## Current results and dashboards

### .12 critic scope v1 — completed local pilot, do not expand

The [critic-scope report](analysis/critic_scope_v1_report.md) keeps saved .12 main outputs fixed and tests a
veto-only critic on 19 frozen cases, two arms and two repeats. All 64 fresh requests return HTTP 200, but the
candidate validates only 16/32 called outputs: each repeat has three contract-name errors and five directional-proof
failures. It also misses a real forum-identity conflict. Under the unchanged partial policy draft, candidate primary
agreement is 9/19 in each repeat versus 13/19 and 14/19 for the fresh .12 critic controls. These are selected-panel,
reference-sensitive results, not full-development performance. Exact-text bypass gains are reported separately.
The report preserves the earlier stopped execution and flags H0093 for policy-boundary review without changing labels.
Use `eval.dedup.analysis.critic_scope_execution` for the failure-durable runner; the original experiment runner's
failure-recorder defect is retained only as frozen history. No release or larger online run is authorized by this pilot.

### .9 / .12 / .32 offline rebenchmark — partial policy-v2 reference sensitivity

The [same-output comparison](analysis/reference_policy_v2_rebenchmark_v1.md) replays all six main/final views on the
identical original 1,000 inputs. Historical references and outputs remain unchanged. A separate, explicitly partial
AI development reference applies five primary-decision revisions under the approved policy-only rule; H0998 remains
an unchanged positive control. Final weighted precision/recall under that draft are **73.86%/57.10% (.9)**,
**67.99%/75.07% (.12)**, and **44.70%/80.80% (.32)**. These are label-sensitivity results, not new model gains or
independent gold. The text-only all-population screen leaves 184 additive candidates without completed individual
review, and two exact-document reference conflicts remain unresolved. Recommend .12 as the next experimental
control and isolate critic intervention; no new online calls, Judge version, release, or 20,000-pair run.

### Policy-only clarification — unchanged complete policy plus product is one-way replaceable

The user-approved [policy v2](analysis/composite_containment_policy_v2.md) permits a page with the complete same policy
plus substantive product content to replace the policy-only page. The [application inventory](analysis/policy_only_application_v1.md)
supports five previously disputed cases and the existing H0998 positive control as AI development suggestions, not new
human gold. Four short credit/warranty notices remain pending; H0352's mandatory negative expectation is suspended.
The old 121-case review and 48/64-case freezes remain intact, with a separate policy overlay covering all 64 IDs.
No new model calls, reference edits, scores or runtime changes; new-policy requests and independent annotations are not yet frozen.

### .12 → .32 full 121-case review — evidence-bound triage and revised panel

The [complete review](analysis/v06212_v06232_review121_report.md) classifies all 121 incremental over-groups:
79 clear model errors, 40 policy/reference disputes and 2 pending cases. These are prediction-aware AI development
judgments, not independent human gold. All 20 lost historical repairs have a separate inventory; 5 supported negative
conclusions enter development protection, while 15 disputed/pending conclusions are not forced onto a new model.
The original 48-case freeze remains unchanged. A [64-case panel revision](analysis/independent_x_probe_v2.json) retains
every old member and flag, adds missing mechanisms and all 5 supported repairs, and uses full anonymous inputs.
Independent X annotations and executable preflight are still pending; no new model calls, reference changes or scores.

### .12 → .32 incremental audit — offline ledger and independent-X probe freeze

The [incremental audit](analysis/v06212_v06232_incremental_audit.md) strictly replays the saved four component views
on all original 1,000 inputs: 47 final decisions improve, 152 regress, 77 remain wrong, and 724 remain reference-correct.
Of 121 newly over-grouped pairs, 101 were already correct in the old main and 20 lose an old postprocessing repair.
A [48-pair independent-X protocol](analysis/independent_x_probe_v1.json) freezes real inputs, protection candidates,
anonymous requests and an empty annotation template. X labels still need independent review; X absence does not imply
no/no for equivalent non-main pages. No new model calls, label changes, Judge version or release are included.

### V0.6.2.32 paced full development — transport recovered, semantic gates fail

The [full audit](analysis/v06232_paced_audit.md) evaluates the original 1,000 pairs under a new
[transport-only execution protocol](analysis/v06232_paced_protocol.md), with unchanged `.32` prompts and references.
Final weighted precision/recall/primary exact are **42.90% / 80.16% / 77.63%**; 156 over-groups include
134 false containments. The critic fixes only two over-groups and one direction. Strict completion is 997/1,000,
and 42/964 called pairs require model corrections. All 1,491 upstream requests, including preflight/local panels,
return HTTP 200; there are no transport retries. See the [machine-readable assessment](analysis/v06232_paced_assessment.json).
The broad semantic-content gate and weak critic correction remain the bottlenecks. This is a completed diagnostic,
not a release or a 75% gate pass; historical failures, labels and caches remain intact.

### V0.6.2.32 directional binding — all semantic probes pass; HTTP 429 gate stops expansion

The [audit](analysis/v06232_directional_audit.md) reports 11/11 correct, fully valid main and final preflight decisions,
with zero model corrections. Both closed-list directions now bind evidence correctly, including the previous terminal
mirror. Critic requests encountered six HTTP 429s before succeeding; the frozen single-HTTP-200 gate therefore stopped
both formal 50-pair repeats. This is a transport-gate failure, not a remaining evidence error or a full-development pass.
The [assessment](analysis/v06232_directional_assessment.json) accounts for all 23 external HTTP requests (17 successful),
while [996 CPU and four native tests](analysis/v06232_boundary.json) and the
[offline feedback probe](analysis/v06232_feedback_probe.json) verify unchanged V5 acceptance and precise retry feedback.
See the [design](analysis/v06232_design.md). References and prior freezes remain unchanged; this is not a release.

### V0.6.2.31 contextual witnesses — targeted fixes pass, mirrored preflight still fails

The [audit](analysis/v06231_context_audit.md) records 10/11 valid first-pass main outputs, all ten matching the original
primary labels. The original forward closed-list case and core-translation-plus-addition case now pass; the mirrored
list still cites a wrong-side counterpart and two shared root witnesses across three attempts. All 13 external HTTP
requests succeeded; strict JSON shape passed, but evidence completion and the correction gate failed. No critic or
formal repeat was submitted. See the [assessment](analysis/v06231_context_assessment.json),
[design](analysis/v06231_design.md), [972 CPU plus four native tests](analysis/v06231_boundary.json), and
[offline adapter probe](analysis/v06231_adapter_probe.json). V5 permits narrowly grounded shared-context contradictions;
public v3, substantive independent-Y containment, references and historical runs remain preserved. This is not a release
or a 75% accuracy-gate pass.

### V0.6.2.30 format-only follow-up — preflight passed, first formal main stopped

The [audit](analysis/v06230_format_audit.md) reports 8/8 correct first-pass preflight decisions for main and final,
with zero corrections. First-round main then produced 49/50 valid outputs; a closed-list conflict was rejected by
the own-unique-only loss-witness constraint. A separate core-translation-plus-addition case required retries to stop
claiming complete-pair translation. No formal critic or second repeat was submitted, so no semantic ranking is claimed.
All 70 external requests remain accounted for (66 HTTP 200, four 429), including one successful response without saved
assistant-trace attribution. See the [assessment](analysis/v06230_format_assessment.json),
[frozen design](analysis/v06230_design.md) and [boundary tests](analysis/v06230_boundary.json).
Only encoding instructions changed from `.29`; references, semantic policy, strict schema and historical runs remain unchanged.

### V0.6.2.29 composite containment — credentials recovered, online preflight stopped

The [credential recovery and preflight audit](analysis/v06229_credential_recovery.md) confirms that the existing
repository `.env` key works: all 12 external requests returned HTTP 200. The main preflight produced 7/8 finally
valid outputs; two pairs needed retries and one repeatedly emitted cross-branch fields. The runner stopped before
critic and both formal repeats. This is a schema-completion failure, not a credential failure or a semantic gate pass.

The earlier [offline implementation checkpoint](analysis/v06229_implementation_report.md) and
[frozen design](analysis/v06229_design.md) implement substantive X + independent Y containment with a fresh V4 main
and a same-policy critic. The candidate preserves the nonempty anchor and actual conflict gates, exports public v3,
does not reuse historical main outputs or caches, and leaves reference labels unchanged. It is an opt-in experiment,
not a release.
Verification passed 900 CPU tests and four native local-HTTP integration cases; these are not model accuracy results.
The frozen schedule was eight preflight cases, then two repeats of 24 real development plus 26 synthetic cases scored
separately; the latter were not submitted after preflight failed. No 75% gate pass or semantic improvement is claimed.
Historical freezes remain unchanged.

### Composite-containment policy approved — historical runs unchanged

The [approved policy](analysis/composite_containment_policy_v1.md) permits substantive X + independent Y to
replace X in one direction, without requiring Y to extend the same record. Nonempty substantive overlap,
full directional retention and actual conflict protections remain required. The
[19-case development review and paired synthetic fixtures](analysis/composite_containment_review_v1.json)
keep policy approval separate from per-case gold adjudication. Old references, scores, prompts, adapters and
caches were unchanged at that policy-only checkpoint; implementation followed in the `.29` candidate above.
The policy review itself made no new model calls.

### Offline bottleneck audit — prompt iteration paused for policy arbitration

The [full-1,000 audit](analysis/bottleneck_audit_v1.md) separates 124 final `.12` primary disagreements into
34 development-reviewed semantic errors, 56 policy/reference disputes, four engineering issues, 23 input limitations,
and seven pending reviews. Disputes are not approved label corrections. It strictly replays the saved main/critic
outputs, reports weighted FP/FN contributions, and keeps `.28` fixed-main local results separate.
See the [arbitration priorities](analysis/bottleneck_arbitration_v1.md), [case-level review notes](analysis/bottleneck_reviews_v1.json),
and [22-case mixed diagnostic panel](analysis/bottleneck_panel_v1.json). This work makes no model calls or reference changes;
the panel is prediction-aware development work, not independent human calibration or a replacement release gate.

### V0.6.2.27 same-contract semantic blocks — complete paired schedule, no promotion

The [audit](analysis/v06227_policy_audit.md) reports 258/258 valid outputs in all four cells. The expanded
policy block improves paired weighted primary by 1.71/0.94 percentage points, but candidate weighted
precision remains 66.70%/64.99% and recall 69.73%/68.20%. Negative/benign protections, repeat stability,
and the 1% correction gate fail. These are fixed-main local diagnostics, not full-development results.
See the [assessment](analysis/v06227_policy_assessment.json), [attribution](analysis/v06227_response_attribution.json),
[protocol](analysis/v06227_design.md), [boundary tests](analysis/v06227_boundary.json), and
[next sequencing design](analysis/v06228_design.md). All 645 external HTTP requests remain accounted for;
the audit separately discloses one dropped native row with two successful requests lacking saved assistant traces.

### V0.6.2.26 typed coverage — candidate preflight passed, flat control stopped the experiment

The [audit](analysis/v06226_typed_audit.md) records typed coverage 8/8 first-pass valid with zero corrections,
but flat control 7/8, then 8/8 after one outer retry (17 HTTP 200 requests total). Both arms had to pass,
so no formal cells or semantic ranking were submitted. Typed sides remove redundant inactive fields;
strict original-response, transport, evidence and semantic protection checks remain separate.
See the [protocol](analysis/v06226_run_protocol.md), [assessment](analysis/v06226_typed_assessment.json),
[boundary tests](analysis/v06226_boundary.json), and [next semantic-block comparison](analysis/v06227_design.md).
The next design uses the same typed contract on both arms; it does not reinterpret the failed .25/.26 schedules.

### V0.6.2.25 proposition-first prompt — preflight stopped before formal scoring

The [audit](analysis/v06225_proposition_audit.md) records control 8/8 first-pass valid and candidate 7/8,
then 8/8 after one outer retry (17 HTTP 200 requests total). A harmless-only mode with nonempty counterpart
references violated the unchanged V2 contract, so the zero-correction preflight gate stopped formal submission.
No semantic ranking is published. Both arms share the .24 scope postprocessor; this control is the .23 selection
prompt, not the old .14 critic. See the [protocol](analysis/v06225_design.md),
[assessment](analysis/v06225_proposition_assessment.json), [boundary tests](analysis/v06225_boundary.json), and
[next contract design](analysis/v06226_design.md). Original results and thresholds remain unchanged.

### V0.6.2.24 offline scope probe — routing gains, semantic protections still fail

The [offline audit](analysis/v06224_scope_audit.md) isolates main/critic content-taxonomy disagreement using
the accepted .23 responses, with no new model calls or edits to either response. Unresolved outputs fall
from 20 to 3 per repeat and weighted primary rises about 2.4 percentage points, but one identity false positive
is exposed and recall remains 68.00%/61.89%. Containment, translation, benign and retry protections still fail.
See the [predeclared protocol](analysis/v06224_design.md), [replay](analysis/selection_scope_replay.py), and
[all-pair assessment](analysis/v06224_scope_assessment.json). This is a code counterfactual, not an online .24
score or a production policy. Labels, weights, historical results and all original gates remain unchanged.

### V0.6.2.23 span selection — complete paired schedule, no promotion

The [online audit](analysis/v06223_selection_audit.md) records 258/258 valid outputs in all four formal cells
and 147/147 final strict candidate contracts per repeat. Selecting original spans removes quote-regeneration
failures without relaxing evidence alignment. Weighted precision is 74.68%/76.28%, recall 65.37%/59.11%,
and primary 74.45%/73.60%; both repeats fail semantic protections and primary stability. Native/outer
correction rates and separate HTTP 429/500/503 failures remain reported, not erased by final completion.
See the [assessment](analysis/v06223_selection_assessment.json), [protocol](analysis/v06223_run_protocol.md),
and [full-pipeline boundary tests](analysis/v06223_boundary.json). These are fixed-main local diagnostics,
not the complete 1,000-pair development evaluation or independent holdout results.

### V0.6.2.22 routed coverage — preflight passed, first formal round stopped

The [audit](analysis/v06222_coverage_audit.md) records a clean 8/8 preflight on both arms, followed by
258/258 control outputs and 256/258 coverage outputs. Coverage requested 147 pairs and retained 111 deterministic
outputs without fabricated critic proofs; its actual called-pair retry rate was 9/147 (6.12%). Two terminal
errors stopped the second repeat. Control also exceeded the 1% correction gate and encountered separately recorded
HTTP 429s. No semantic ranking or promotion is claimed from this incomplete schedule. See the
[assessment](analysis/v06222_coverage_assessment.json), [protocol](analysis/v06222_run_protocol.md), and
[boundary tests](analysis/v06222_boundary.json). The [.23 design](analysis/v06223_design.md) proposes selecting
original spans instead of regenerating quotes, while keeping the substantive-anchor gate and all original targets.
The audit also discloses the frozen prompt's incorrect side-field count (13 stated; the actual schema has 12).

### V0.6.2.21 coverage preflight — transport fixed, retry gate not passed

The [online preflight audit](analysis/v06221_coverage_audit.md) records 8/8 final valid pairs on both arms and
18 external HTTP requests. Coverage needed one outer retry and one native correction on the same pair, so the
frozen zero-retry preflight gate stopped formal submission. Actual native prompts and transport checks passed;
this is not evidence of semantic improvement. See the [assessment](analysis/v06221_coverage_assessment.json),
[run protocol](analysis/v06221_run_protocol.md), and [runtime boundary tests](analysis/v06221_boundary.json).
An [offline ownership-routing audit](analysis/v06221_coverage_routing_audit.json) preserves all 258 pair IDs and
identifies 147 requiring coverage, with 111 deterministic branches unchanged by valid coverage probes.
At the time of that audit the routing was offline only; the subsequent .22 experiment above uses fresh frozen
requests and separate called-pair retry accounting. Original labels and gates remain unchanged.

### V0.6.2.20 structured coverage preflight — stopped before formal evaluation

The [preflight audit](analysis/v06220_coverage_audit.md) records control 8/8 valid and coverage 0/8 accepted:
the new payload equality check rejected Arrow's null padding and numeric representation changes. All 32 external
HTTP requests are retained; no formal 258-pair cell or full-development run was submitted. A separate strict
[transport binding](judging/payload_transport.py) and stable retry-feedback encoding now pass two complete Ray/HTTP
boundary cases. [Offline verification](analysis/v06220_transport_verification.json) still finds eight response-contract
failures among 24 saved attempts, so neither the replay nor transport fix establishes semantic improvement.
See the unchanged [contract](analysis/v06220_design.md), [frozen protocol](analysis/v06220_run_protocol.md),
[original failed assessment](analysis/v06220_coverage_assessment.json), and [.21 follow-up scope](analysis/v06221_design.md).
Historical contracts, labels, weights and run results remain unchanged; no release is approved.

### V0.6.2.19 non-main consistency counterfactual — not advanced

The [offline audit](analysis/v06219_consistency_audit.md) replays all 3,612 historical critic responses from .16–.18
without new model calls. A narrowly supported equivalence/extension disagreement occurs in 107 responses across 11
pair IDs. Turning those decisions into abstentions raises precision but lowers primary accuracy in all 14 cells;
no primary error becomes correct. This is not a production policy or release. See the
[protocol](analysis/v06219_design.md), [replay tool](analysis/nonmain_consistency.py), and
[all-cell assessment summary](analysis/v06219_consistency_assessment.json). Original labels, weights, and frozen runs
remain unchanged; 3,612 repeated responses are not 3,612 independent examples.

### V0.6.2.18 local-context diagnostic — not advanced

The [paired-repeat audit](analysis/v06218_context_audit.md) preserves full span quotations and adds only short original-line
context. Both repeats underperform their control; over-group counts rise from 23/23 to 25/25. The predeclared 136 unchanged
requests separate response variance from actual prompt changes. All 1,032 critic-only judgments are valid, but this is not
a full online candidate or release. See the [protocol](analysis/v06218_design.md), [assessment](analysis/v06218_context_assessment.json),
[response attribution](analysis/v06218_response_attribution.json), [main ownership audit](analysis/v06218_main_ownership_audit.json),
and [pending reference-policy checks](analysis/v06218_reference_pressure.md). The ongoing improvement goal remains unmet;
original labels, weights, production policies and historical results are unchanged.

### V0.6.2.17 fixed-main presentation diagnostic — not advanced

The [paired-repeat audit](analysis/v06217_presentation_audit.md) tests complete original-order text plus a positional span
index against the unchanged .14 critic, with main responses and v8 arbitration frozen. It improves the two consent-condition
targets, but the short story pointer remains misclassified and over-group counts increase from 24/24 to 29/31 across repeats.
Both repeats fail negative protections; the 1,032 critic-only judgments do not authorize a full-development run or release.
See the [protocol](analysis/v06217_design.md), [assessment](analysis/v06217_presentation_assessment.json),
[raw-response attribution and repeated inputs](analysis/v06217_response_attribution.json), and
[pending reference review](analysis/v06217_reference_review.md). Original labels, weights, and historical results remain unchanged.

### V0.6.2.16 fixed-main critic diagnostics — neither patch advanced

The [paired-repeat audit](analysis/v06216_critic_audit.md) isolates two minimal .14 critic patches on the same 258 pairs,
with .14 main responses and v8 arbitration frozen. Both consent-only and pointer-only patches underperform the unchanged
control in both repeats and fail negative protections. The 1,548 critic-only judgments are diagnostic, not a complete
online Judge candidate; no full-development run or production policy was registered.
The audit also finds 230 distinct visible inputs among 258 pairs and one identical-input reference conflict (H0177/H0634).
Original labels and weights remain unchanged. See the [protocol](analysis/v06216_design.md),
[assessment](analysis/v06216_critic_assessment.json), [response attribution](analysis/v06216_response_attribution.json),
and [input redundancy audit](analysis/v06216_input_redundancy.json).

### V0.6.2.15 critic experiment — rejected

The [258-pair audit](analysis/v06215_local_audit.md) restores all five benign guards but regresses weighted primary
agreement from 73.10% (.14) to 69.05%, fails translation and negative protections, and does not proceed to full development.
The main prompts were unchanged, yet 39/258 repeated main decisions differed. Fixed-main counterfactuals still show the new
critic regressing; those mixed-response diagnostics are not an online candidate. See the [43-case AI review](analysis/v06215_critic_review.md),
[frozen protocol](analysis/v06215_design.md), and [component attribution](analysis/v06215_component_attribution.json).
No labels, historical resources, caches, or release approvals were changed.

### V0.6.2.14 local development — not promoted

The [258-pair local audit](analysis/v06214_local_audit.md) reports weighted primary agreement improving from
66.92% (.13) to 73.10% (.14), with all four containment guards restored, but weighted precision falling from
70.23% to 65.85%. Two frozen continuation guards still fail (H0480 and H0347), so no full 1,000-pair .14 run
was submitted. This is an error-enriched development subset, not independent human calibration or release evidence.
See the [frozen protocol](analysis/v06214_design.md), [code-only replay](analysis/v06214_arbitration_replay.json),
[local assessment](analysis/v06214_local_assessment.json), and [same-response critic attribution](analysis/v06214_component_attribution.json).
Historical prompts, results, reference labels, and caches remain unchanged.

### Experimental checkpoint through V0.6.2.11

V0.6.2.9 remains the comparison baseline for the 127-pair residual experiment, **not an approved release**.
V0.6.2.10 and both V0.6.2.11 candidates failed their frozen gates; preserving them here does not promote them.
See the [V0.6.2.9 audit](analysis/v0629_residual_smoke.md),
[V0.6.2.10 failure analysis](analysis/v06210_failure_analysis.md), and
[V0.6.2.11 experiment report](analysis/v06211_experiment_report.md).
This checkpoint does not change release approvals or authorize full-development, holdout, or 20,000-pair runs.

Large historical review CSV/HTML snapshots are stored with Git LFS; run `git lfs pull` after cloning to retrieve them.
Prompt versions, audit summaries, and diagnostic labels are preserved together. Runtime caches, credentials, model
outputs, and run roots outside this repository are not part of the Git checkpoint. Prediction-aware diagnostic labels
must not be treated as an independent human holdout.

Checkpoint verification: all 270 dedup tests passed. The repository's pre-commit hooks were run using an existing cached
installation. Large-file, case-conflict, YAML, private-key, trailing-whitespace, and Ruff lint checks passed; EOF and Ruff
format checks reported existing formatting differences. Their automatic edits were reverted to preserve the frozen source,
prompt, and artifact bytes. This checkpoint therefore does not claim a clean formatting gate or release readiness.

### Existing shared dashboards

These stable links require access to the NVIDIA internal network:

| Result | Link | Contents |
|---|---|---|
| Latest automated result | [Pair Explorer](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/) | Result-linked removal errors, cross-group positives, Judge decisions, provenance, and group context. |
| Latest Human QA set | [Human QA Dashboard](http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/human-qa/) | The 200-pair blind sample and 200-pair diagnostic set, review progress, and CSV export. |

The shared Human QA URL always serves the dashboard from the latest run with a completed Step 8. Every run keeps its own
immutable HTML file, so automatic updates do not overwrite historical results. The current reference run is
`dedup-full-20260813T220949Z-d4c37bb483`.

## Runtime profiles

- `smoke`: all 10,008,061 handoff documents, 20 anchors, 50 removal decisions, up to 50 cross-group pairs, and at most 100 judge requests. Its report is explicitly non-V0.
- `full`: 1,000 anchors, 10,000 removal decisions, up to 10,000 cross-group pairs, a 200-pair blind Human QA sample, and a separate 200-pair diagnostic set.

The default smoke and full configs now use Sarah's MinHash/surface-overlap NDD Judge contract,
`dedup-judge-sarah-minhash-v1`, with the local `Qwen/Qwen3.8-27B` model on one B200. Both Sarah-backed profiles are
explicitly non-formal-V0.

The original NVIDIA API setup is unchanged in
`v0_config.legacy_nvidia.example.json` and `v0_config.legacy_nvidia.full.json`. The legacy full profile freezes
`nvidia/deepseek-ai/deepseek-v4-pro` and remains the formal V0 configuration.

The first V0 implementation deliberately uses the proposal's allowed path of skipping the optional minimum-diff challenge slice. The full 10,000 Step 5a budget is a uniform sample of actual keeper-to-removed decisions.

## Evaluation contract

The relaxed lexical grid is frozen as `(5,1), (6,1), (7,1), (8,1)` after real-corpus calibration. The pilot still applies the proposal's hard rule: select a configuration with median cross-group candidates in `[20,50]` closest to 35, or fail. The per-anchor safety limit is 250,000 and every trial is written to `retrieval_config.json`.

All backends are validated locally. The legacy JSON-mode path retains its evidence realignment policy. The Sarah path runs one
Ray/Dynamo/vLLM/Qwen service across the initial batch and at most two failed-subset retries. Missing rows, duplicate rows,
malformed rubrics, invalid enums, and cross-field inconsistencies are retried with safe structured validation feedback only.
Every pair ends with either a schema-valid result or an explicit terminal error; valid records are fsynced individually to the
existing Judge cache so an interrupted run submits only pending pairs on resume.

The tracked Sarah YAML disables Data Designer's batch-level early shutdown, sets deterministic conversation restarts to zero,
and allows two parser-correction turns per row. This prevents a handful of malformed fenced-JSON replies from abandoning the
rest of a blind batch; at temperature zero, correction feedback is useful while a fresh restart would repeat the same reply.
The correction trace and NDD reasoning remain runtime-only and are never written to formal dedup artifacts.

The Sarah YAML, Jinja prompts, and compatibility shim are content-hashed into the Judge contract, run manifest, and cache key.
Changing any of them creates a different execution contract.

### Judge contract versions

- `dedup-judge-hs-v0.6.2.5` with `dedup-judge-output-v3` is the rejected boundary-delta experiment. It separates verified same-record
  extensions from record/role/state changes, material non-main-message changes, two-sided changes, and genuinely harmless
  universal UI/repetition. It uses the normal bilateral exact-evidence contract without V0.6.2.4's redundant field-specific
  identity quotes.
- `dedup-judge-hs-v0.6.2.4` with `dedup-judge-output-v3` is the rejected verified-identity and surface-delta experiment. It adds
  field-specific, byte-aligned identity evidence; distinguishes document-wide overlap from copied local passages and generic
  templates; and separates benign labels/repetition from added propositions. The adapter applies these gates before the
  frozen V0.6.2.3 ledger resolution while preserving V0/V1/V2/V3 read compatibility.
- `dedup-judge-hs-v0.6.2.2` with `dedup-judge-output-v3` is the semantic-ledger development candidate. It asks for
  content profiles, shared-content basis, hard conflict, and decisive-difference location, then derives the published
  replacement decisions in the adapter's fixed gate order. These ledger values remain diagnostic reason codes rather than
  new output-contract fields. Its frozen 1,000-pair development run passed precision, primary-exact, over-group, schema,
  retry, identity, and translation gates, but failed recall and meaningful-addition non-regression; it must not advance to
  holdout.
- `dedup-judge-hs-v0.6.2.1` with `dedup-judge-output-v3` is the second semantic-only release candidate. It preserves
  the immutable v0.6.2 prompt while tightening the identity/state/page-role veto for boilerplate-dominated and
  template-sibling pages. It also requires a non-empty page-specific anchor for near-surface and containment decisions.
- `dedup-judge-hs-v0.6.2` with `dedup-judge-output-v3` is the first semantic-only release candidate. It keeps the two
  replacement directions, derived duplicate-group decision, relation/material taxonomy, primary difference, overlap
  source, risk factor, ordinal confidence tier, derived reason codes, and exact two-sided evidence. It removes LLM-predicted
  MinHash behavior, `evidence_quality`, and numeric confidence. Containment requires a shared substantive anchor, exactly one
  safe direction, a major difference, and a meaningful main-content addition. Identity, state, list-membership, and page-role
  conflicts are always no/no; faithful full translations remain yes/yes with no semantic material difference.
- `dedup-judge-hs-minhash-v0.6.1` with `dedup-judge-output-v2` removes the overloaded `fuzzy_scope` field. The Judge
  independently predicts `expected_minhash_action` (`GROUP`, `KEEP_SEPARATE`, or `UNCERTAIN`); the adapter then derives
  surface-evidence sufficiency and the expected true-positive, true-negative, false-positive-risk, false-negative-risk, or
  ambiguous outcome. Semantic duplicate/replacement decisions are the primary score axis, relation/material labels are a
  descriptive axis, and MinHash diagnostics never determine the version winner. Non-exact resolved decisions must retain
  aligned quotes from both documents, while overlap source, primary material difference, and primary risk remain available
  in result artifacts and the Pair Explorer.
- `dedup-judge-sarah-minhash-v1` uses Sarah's NDD surface-overlap policy and `judge-visible-payload-v2`, but adapts the
  output to the existing `dedup-judge-output-v0` artifact schema. NDD enums are uppercased, boolean reason rubrics become the
  existing flat reason-code array, confidence uses a discrete rubric, and `evidence=[]`. NDD reasoning is not stored; only a
  canonical response SHA-256 is retained.
- `dedup-judge-v0` with `dedup-judge-output-v0` preserves the original prompt, flat reason-code array, and
  `judge-visible-payload-v1` neutral document metadata.
- `dedup-judge-v1` with `dedup-judge-output-v1` uses the polished multilingual policy, structured V2 reasons, stronger
  cross-field consistency validation, and `judge-visible-payload-v2`. The v2 payload exposes only cleaned text plus explicit
  long-document truncation windows; it excludes URL, hostname, timestamps, language tags, character/token counts, and all SUT
  provenance.

Select v1 by changing both version fields together:

```json
{"prompt_version":"dedup-judge-v1","schema_version":"dedup-judge-output-v1"}
```

Rejudge the frozen comparison population with V0.6.1 through the versioned runner:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v061 --run-id v0.6.1
```

Use the printed run root with the `run` and `summarize` subcommands. The summary compares the primary decision tuple and
descriptive taxonomy separately; legacy `fuzzy_scope` agreement is intentionally omitted across incompatible contracts.
V0.6.1 asks the Judge for the two replacement directions and deterministically derives `same_duplicate_group`, avoiding a
third independently generated answer that could contradict containment. Exact quoted evidence is retained only when it
aligns byte-for-byte with the visible text, including standard single-, double-, or curly-quoted spans. If a resolved side
lacks an aligned model quote, the adapter records the repair and supplies an exact visible excerpt from that side; it never
constructs evidence for empty visible text.

The frozen v0.6.1 human-calibration and proxy-challenge findings are in
[`analysis/v061_audit.md`](analysis/v061_audit.md). The v0.6.2 development comparison uses three immutable policies with the
same model, decoding settings, and selected payload IDs:

- `hs-v062-dev-baseline`: v0.6.1 semantic logic transplanted to output v3;
- `hs-v062-dev-gate`: adds the shared-main-content and non-empty-containment gate;
- `hs-v062`: adds the first complete boundary examples;
- `hs-v0621`: preserves v0.6.2 and adds stricter template-slot/page-role examples after v0.6.2 missed its development
  precision, over-group, meaningful-addition, primary-exact, and retry gates.
- `hs-v0622`: adds an explicit content-profile/shared-basis/conflict/difference ledger. The adapter resolves replacement
  directions from this fixed gate order while keeping the published `dedup-judge-output-v3` field contract unchanged.
- `hs-v0623`: adds reviewed record-alignment, non-main-message, and translation gates. The adapter permits substantive
  equivalence/containment only for a confirmed same record and keeps materially different non-main messages separate.
- `hs-v0624`: adds verifiable record-identity evidence, overlap scope, and surface-delta gates. It is evaluated against the
  cumulative reconciled development labels only after the V0.6.2.3 diagnostic follow-up and three-row review revision.
- `hs-v0625`: adds the decisive boundary-delta class and restores verified one-sided containment while separating universal
  UI/repetition from record/role/state and material non-main-message changes.
- `hs-v0626`: replaces free-form evidence with a deterministic shared/A-only/B-only span packet. The model cites stable span
  IDs and the adapter derives byte-aligned bilateral evidence and fails closed on incomplete or invalid citations.
- `hs-v0627`: keeps the span Judge and adds an independent record-binding critic. Containment survives only when the critic
  confirms that the unique content predicates the same atomic record rather than an attached record or reusable block.
- `hs-v0628`: calibrates the independent critic on FAQ, field-order, page-role, policy, product/store, and collection/list
  boundaries and allows it to veto substantive positive decisions.
- `hs-v0629`: makes critic arbitration asymmetric: the span ledger owns non-main equivalence and additive translation,
  while explicit substantive attachment, conflict, and policy-change verdicts retain veto authority.
- `hs-v06210`: separates non-main policy subtypes, bilateral translation direction, and atomic-record review into three
  independently cited critic axes. Only exhaustively cited chrome may flatten a main extension. See the frozen
  [design and residual gates](analysis/v06210_design.md); no old judge cache is reused.
- `hs-v06211-policy`: keeps the V0.6.2.9 main Judge and legacy record-binding rubric, then adds a scoped non-main policy
  supplement. This is a development ablation, not a release candidate.
- `hs-v06211`: adds an applicability-gated translation supplement to that ablation. Citation failures affect the active
  decision branch; valid negative decisions survive unrelated issues. See [the frozen protocol](analysis/v06211_design.md).
- `hs-v06212-route`: retains the exact .9 prompts and isolates the deterministic complete non-main translation fix.
- `hs-v06212`: synchronizes main/system/pair rubrics, distinguishes equivalence from extension in the critic, and requires
  a separately cited retained identity/service/role/permission/state conflict to override non-main equivalence. See the
  [implementation design](analysis/v06212_design.md) and [22-case development review](analysis/v06212_extension_review.md).
  This candidate has not been evaluated online or approved for release.

Prepare each development run with the same 1,000-pair label CSV:

```bash
python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v062-dev-baseline --pair-ids eval/dedup/analysis/hs_blind_adjudication_1000.csv
python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v062-dev-gate --pair-ids eval/dedup/analysis/hs_blind_adjudication_1000.csv
python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v062 --pair-ids eval/dedup/analysis/hs_blind_adjudication_1000.csv
python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v0621 --pair-ids eval/dedup/analysis/hs_blind_adjudication_1000.csv
python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v0622 --pair-ids eval/dedup/analysis/hs_blind_adjudication_1000.csv
python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v0623 --pair-ids eval/dedup/analysis/v0622_policy_reconciled_labels_1000.csv
```

After running those roots, `analysis.judge_calibration` verifies equal payload/model/temperature contracts, reports weighted
and unweighted human metrics plus true accuracy by confidence tier, and applies the frozen development gates:

```bash
python -m eval.dedup.analysis.judge_calibration \
  --labels eval/dedup/analysis/hs_blind_adjudication_1000.csv \
  --baseline-run-root <baseline-run-root> \
  --candidate gate=<gate-run-root> --candidate final=<final-run-root> \
  --output <development-comparison.json>
```

The frozen v0.6.2.2 result and gate decisions are recorded in
[`analysis/v0622_development_audit.md`](analysis/v0622_development_audit.md) and
[`analysis/v0622_development_comparison.json`](analysis/v0622_development_comparison.json). Generate its disagreement
packet before changing the prompt or any development labels:

```bash
python -m eval.dedup.analysis.policy_review \
  --labels eval/dedup/analysis/hs_blind_adjudication_1000.csv \
  --candidate-run-root /raid/hfang/ihb/runs/v0.6.2.2-dev-final \
  --previous-run-root /raid/hfang/ihb/runs/v0.6.2.1-dev-final \
  --output-csv eval/dedup/analysis/v0622_policy_review_candidates.csv \
  --summary eval/dedup/analysis/v0622_policy_review_summary.json
```

The packet is a review queue, not an automatic relabeling instruction. In particular, old containment labels that rely on a
substantive-page/non-main-only pairing must be adjudicated against the new non-empty-containment policy before another prompt
is tuned to the same development set.

The requested 85-row policy reconciliation, its preserved-label overlay, and the resulting V0.6.2.3 design are recorded in
[`analysis/v0622_policy_reconciliation.md`](analysis/v0622_policy_reconciliation.md) and
[`analysis/v0623_design.md`](analysis/v0623_design.md). Use the reconciled labels only for subsequent V0.6.2.x development;
retain the original-label comparison as historical agreement with the old taxonomy. Before a full 1,000-pair V0.6.2.3 run,
run its immutable contract against `analysis/v0622_reconciled_residual_candidates.csv` as the targeted residual smoke set.

That smoke is now complete and rejected; its measured gates and diagnosis are in
[`analysis/v0623_residual_smoke.md`](analysis/v0623_residual_smoke.md). Do not mutate or promote V0.6.2.3, and do not run it
on the holdout. Any follow-up prompt must use a new immutable version after the newly exposed translation and containment-label
conflicts are adjudicated.

Those 20 exposed cases and three additional review revisions are now adjudicated in separate diagnostic ledgers. The
cumulative label baseline and the immutable V0.6.2.4 contract, deterministic gates, and staged admission thresholds are
documented in [`analysis/v0624_design.md`](analysis/v0624_design.md). Its completed residual-smoke decision is documented in
[`analysis/v0624_residual_smoke.md`](analysis/v0624_residual_smoke.md): it failed schema, retry, recall, protected-duplicate,
and diagnostic-negative gates, so no full-development or holdout run is permitted merely because its prompt is registered.

The immutable V0.6.2.5 contract and its frozen residual-smoke gates are documented in
[`analysis/v0625_design.md`](analysis/v0625_design.md). Its completed gate result and architecture-level diagnosis are in
[`analysis/v0625_residual_smoke.md`](analysis/v0625_residual_smoke.md). It failed five semantic gates, so registration does
not authorize development or holdout evaluation.

V0.6.2.6's deterministic span architecture and V0.6.2.7's independent record-binding critic are documented in
[`analysis/v0626_residual_smoke.md`](analysis/v0626_residual_smoke.md),
[`analysis/v0627_design.md`](analysis/v0627_design.md), and
[`analysis/v0627_residual_smoke.md`](analysis/v0627_residual_smoke.md). V0.6.2.7 sharply reduced false containment but lost
one protected benign duplicate and one exact containment direction, so neither version may advance to holdout or a full
development run.

The calibrated V0.6.2.8 experiment is documented in
[`analysis/v0628_design.md`](analysis/v0628_design.md) and
[`analysis/v0628_residual_smoke.md`](analysis/v0628_residual_smoke.md). It reduced false containment to one but over-applied
its non-main veto, failed recall/protected-duplicate/containment-direction gates, and must also remain development-only.

V0.6.2.9's asymmetric arbitration is documented in
[`analysis/v0629_design.md`](analysis/v0629_design.md) and
[`analysis/v0629_residual_smoke.md`](analysis/v0629_residual_smoke.md). It restored recall, all protected benign duplicates,
and three of four true-containment directions, but reopened one diagnostic non-main negative and missed one additive
translation. It is frozen and may not advance to the full development set or holdout.

A subsequent explicit user request authorized a one-off full 1,000-pair **diagnostic exception**, not promotion.
The [frozen diagnostic protocol](analysis/v0629_full_development_protocol.md),
[full-development audit](analysis/v0629_full_development_audit.md),
[metrics and fixed-output arbitration analysis](analysis/v0629_full_development_summary.json), and
[154-row disagreement queue](analysis/v0629_full_development_errors.csv) preserve that experiment separately.
Weighted precision/recall are 73.86%/59.01%; 91 of 95 reference duplicate misses occur outside the old 127-pair residual.
The earlier rejection remains in effect. No labels, frozen Judge resources, release approval, or holdout were changed.

The [V0.6.2.12 isolated translation replay](analysis/v06212_translation_route_replay.json) uses all 1,000 saved .9 outputs,
not new model calls: weighted precision/recall become 74.92%/62.37%, with 23 recovered reference positives and no new
over-group. All 38 critic-repaired negative guards survive. The revised .12 critic prompt is **not** evaluated by this
replay; its required retained-conflict field cannot be invented from historical outputs. For future full-development
diagnostics, pass `--guard-replay eval/dedup/analysis/v06212_translation_route_replay.json` to enforce that protection suite.

The subsequently authorized [V0.6.2.12 online development audit](analysis/v06212_online_development_audit.md)
completed a separate 20-pair technical preflight and all 1,000 development pairs with 100% valid schema and no terminal
errors or outer pair retries. Weighted precision/recall are **67.99%/77.59%**, with 57 over-groups and 22 false containments;
only 20 of 38 critic-repaired negative guards survive. It fails development promotion despite its translation/recall gains.
The [frozen protocol](analysis/v06212_online_development_protocol.md),
[metrics](analysis/v06212_online_development_summary.json),
[gate comparison](analysis/v06212_online_development_comparison.json),
[saved-output attribution](analysis/v06212_online_development_attribution.json), and
[124-row disagreement queue](analysis/v06212_online_development_errors.csv) preserve the failure without rewriting labels
or frozen Judge resources. No holdout, 20,000-pair run, or release approval is authorized by this diagnostic.

V0.6.2.13's [implementation and predeclared continuation protocol](analysis/v06213_design.md) separates an
exact-only adapter control (`hs-v06213-exact`, unchanged .12 prompts) from the new specific-record-scope experiment
(`hs-v06213`). The [exact replay](analysis/v06213_exact_replay.json) fixes two previously unresolved identical pairs
without changing any other primary decision. The [258-pair local online audit](analysis/v06213_local_audit.md) is
**rejected**: false containments fall from 22 to 3, but only 1/4 true-containment directions and 2/5 benign guards survive;
six previously preserved negative guards regress. The prepared full 1,000-pair run was not submitted.
Use the [weight-audited evaluation labels](analysis/v06213_local_evaluation_labels.csv) for local scoring, not the frozen
selection CSV, which omitted the original stratum weight columns. Original reference labels and weights remain unchanged.
The [local metrics and arbitration ablation](analysis/v06213_local_assessment.json),
[cross-run counterfactuals](analysis/v06213_cross_run_attribution.json), and
[73-row disagreement queue](analysis/v06213_local_errors.csv) are development diagnostics, not promotion or holdout results.

V0.6.2.10's evidence-scoped boundary review is documented in
[`analysis/v06210_design.md`](analysis/v06210_design.md). Its [residual audit](analysis/v06210_residual_smoke.md) and
[failure analysis](analysis/v06210_failure_analysis.md) reject progression: citation/applicability abstentions and
site-as-record containment errors outweigh the two targeted fixes. Reproduce the audit without re-judging or reading holdout
data (using the existing environment with NeMo Curator installed):

```bash
python -m eval.dedup.analysis.residual_audit \
  --labels eval/dedup/analysis/v0628_policy_reconciled_labels_1000.csv \
  --subset eval/dedup/analysis/v0622_reconciled_residual_candidates.csv \
  --negative-review eval/dedup/analysis/v0623_followup_adjudications.csv \
  --negative-review eval/dedup/analysis/v0624_preflight_adjudications.csv \
  --baseline-run-root /raid/hfang/ihb/runs/v0.6.2.9-residual-smoke \
  --candidate-run-root /raid/hfang/ihb/runs/v0.6.2.10-residual-smoke \
  --output eval/dedup/analysis/v06210_residual_smoke_summary.json \
  --report eval/dedup/analysis/v06210_residual_smoke.md
```

The shared calibration evaluator counts unresolved reference duplicates as recall/containment misses. The residual audit
checks exact protected directions as well as groups, records confidence-tier accuracy, and verifies equal execution and
result digests. These reconciled, error-enriched development references are not independent human holdout labels.

For V0.6.2.11 audits, pass `--gate-profile v06211`, keep the V0.6.2.9 baseline, and select the new policy or final run root.
The profile tightens false containment to <=1 and requires weighted metric non-regression, among the other frozen gates.
Both candidates failed these gates; see [the experiment report](analysis/v06211_experiment_report.md) and
[fixed-output ablation](analysis/v06211_fixed_output_ablation.json). Neither is approved for progression.
Its [offline replay](analysis/v06211_offline_replay.json) is an adapter-only diagnostic, not a new model run or judge cache.
Reproduce that diagnostic to a new output file with:

```bash
python -m eval.dedup.analysis.replay_boundary \
  --source-run-root /raid/hfang/ihb/runs/v0.6.2.10-residual-smoke \
  --labels eval/dedup/analysis/v0628_policy_reconciled_labels_1000.csv \
  --output /path/to/new-v06211-offline-replay.json
```

Only a candidate that passes every development gate may build the new 400-pair holdout with
`analysis.select_v062_holdout`. The private manifest retains split and review-load
assignments; the reviewer packet contains only opaque QA IDs and blind visible payloads. Exactly 50 representative and 50
difficult pairs receive double-independent review, with third-reviewer adjudication declared for disagreements. The final
candidate is evaluated on this holdout once; a failed holdout becomes development data and a new holdout must be sampled.

```bash
python -m eval.dedup.analysis.select_v062_holdout \
  --comparisons /raid/hfang/ihb/runs/v0.6.1/data/pair_comparisons.parquet \
  --payloads /raid/hfang/ihb/runs/v0.6.1/data/judge_payloads.jsonl \
  --exclude-csv eval/dedup/analysis/hs_blind_adjudication_1000.csv \
  --exclude-csv eval/dedup/analysis/v05_v06_v061_adjudicated_benchmark_14336.csv \
  --private-manifest <holdout-private.jsonl> --review-packet <holdout-review.jsonl> \
  --review-dashboard <holdout-review.html> --selection-summary <holdout-selection.json>
```

Run only the final prompt on the frozen private holdout IDs, consolidate the independent reviews and third-reviewer
adjudications into one 400-row labels CSV, then consume the holdout once:

```bash
python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v0623 --pair-ids <holdout-private.jsonl> --run-id v0.6.2.3-holdout
python -m eval.dedup.analysis.holdout_evaluation \
  --labels <adjudicated-holdout-labels.csv> --private-manifest <holdout-private.jsonl> \
  --baseline-results /raid/hfang/ihb/runs/v0.6.1/data/judge_results.jsonl \
  --candidate-run-root /raid/hfang/ihb/runs/v0.6.2.3-holdout \
  --report <holdout-report.json> --approval <v0.6.2.3-release-approval.json>
```

The approval file is created only if every frozen holdout gate passes. A full release prepare without that matching Judge
contract approval is rejected:

```bash
python -m eval.dedup.rejudge_comparison prepare \
  --judge-policy hs-v0623 --release-approval <v0.6.2.3-release-approval.json> --run-id v0.6.2.3
```

MinHash is an optional second-stage diagnostic in `analysis.minhash_diagnostics`. It accepts only a resolved SUT contract
whose provenance is `sut_resolved_config` and real SUT-produced signatures, then deterministically replays band collisions.
Missing normalization, shingling, hash, band/row, or seed fields returns `UNAVAILABLE_MISSING_CONTRACT`; evaluation-retriever
settings are explicitly rejected as a substitute.

The blind Judge records observable dedup risk factors, not hidden SUT error direction. Reporting may combine those factors with
the SUT result later to derive overmerge or undermatch categories without leaking the SUT decision into the Judge prompt.

## How to run the evaluation

Run every command below from the repository root.

### 1. Prepare the environment

You need Linux with CUDA 12, access to the frozen 10M handoff, the local model at
`/raid/hfang/hf_cache/Qwen3.8-27B`, and sufficient `/raid` space. The project requires uv `>=0.12.0`; the validated
installation used uv 0.12.3 and Python 3.11.

Keep the Sarah-capable environment separate from the existing dedup environment:

```bash
export UV_PROJECT_ENVIRONMENT=/raid/hfang/llm_judge_env_pr2324_latest
export UV_CACHE_DIR=/raid/hfang/dedup_eval_cache/uv
export HF_HOME=/raid/hfang/dedup_eval_cache/huggingface
export PATH=/raid/hfang/llm_judge_tools/bin:$PATH
/raid/hfang/dedup_eval_tools/uv sync --locked --python 3.11 \
  --extra deduplication_cuda12 --extra text_cuda12 --extra sdg_cuda12
```

`text_cuda12` supplies the text/runner stack, `sdg_cuda12` supplies Data Designer and Dynamo, and
`deduplication_cuda12` supplies the RAPIDS retrieval path. The lock fixes `data-designer==0.9.1` and applies
`pyarrow>=19,<24` because RAPIDS/cuDF does not accept Data Designer's upstream `pyarrow>=24` requirement; the validated
environment resolves PyArrow 23.0.1. Do not replace the locked install with an unconstrained `uv pip install`. Because uv
environments normally omit pip but Ray 2.57's uv actor bootstrap needs it, the local Judge runner seeds pip from Python's
bundled `ensurepip` wheel before Ray starts when necessary.

The local YAML expects `etcd` and `nats-server` on `PATH`, writes Ray/uv/checkpoint state under
`/raid/hfang/dedup_eval_cache`, allows 1800 seconds for runtime setup, and explicitly loads
`local_ndd/cutlass_compat/sitecustomize.py` for the verified CUTLASS compatibility aliases. Gemma is not started by the
default dedup backend. Keep `ray_temp_dir` short because Ray's dashboard Unix socket has a 107-byte path limit after its
session suffix is added; `preflight` and `LocalJudgeRuntime` reject paths that exceed the conservative budget.

### 2. Configure data and credentials

Review the absolute handoff, output, cache, model, Ray, and checkpoint paths before running:

- Sarah/Qwen default smoke: `eval/dedup/resources/v0_config.example.json`
- Sarah/Qwen default full: `eval/dedup/resources/v0_config.full.json`
- Legacy NVIDIA smoke: `eval/dedup/resources/v0_config.legacy_nvidia.example.json`
- Legacy NVIDIA formal V0 full: `eval/dedup/resources/v0_config.legacy_nvidia.full.json`

Sarah's local backend needs no API key. Only the legacy NVIDIA configs read `NVIDIA_API_KEY`; place it in a repository-root
`.env` or export it before invoking the CLI:

```dotenv
NVIDIA_API_KEY=replace_with_your_nvidia_api_key
```


The CLI does not override an exported value. `.env` is Git-ignored, and credentials are never written to run artifacts.

### 3. Run preflight, then the evaluation

Preflight validates the handoff, storage, tokenizer, GPU retrieval prerequisites, and Judge contract without creating a run.

Smoke:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.example.json \
  --profile smoke

/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.example.json \
  --profile smoke
```

Full:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.full.json \
  --profile full

/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.full.json \
  --profile full
```

Legacy formal V0:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup preflight \
  --config eval/dedup/resources/v0_config.legacy_nvidia.full.json \
  --profile full

/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup run \
  --config eval/dedup/resources/v0_config.legacy_nvidia.full.json \
  --profile full
```

The `STRUCTURED_OUTPUT_FALLBACK_REQUIRED` preflight error and `json_object_plus_local_schema` fallback apply only to the
legacy NVIDIA API backend.

Each `run` command prints the immutable `run_root` used by all later commands.

### 4. Resume, inspect, and validate

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup resume --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup status --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run

/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup validate --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run
```

Run creation freezes the `eval/dedup` Python, YAML, and Jinja source digest. `run` and `resume` stop with
`RESUME_SOURCE_MISMATCH` if the implementation changes. Resume an older run only with the source revision that created it;
start a new immutable run after changing Judge code or resources.

### 5. Read or regenerate the automated report

Step 10 publishes the automated report without waiting for Human QA. For an older immutable run whose Step 9 artifacts are complete, render a versioned derived report without altering its stage markers:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup report --run-root <v0_run>
```

By default, derived reports are exported to
`/home/nfs/hfang/dedup_eval/dedup_eval_runs/<evaluation_run_id>/v0_run/reports/`.
Large run artifacts remain under the frozen run root on `/raid`. Use `--output-root`
to select a different run-scoped report export root.

Derived renders default to the `automated_v9` output label and also write a matching self-contained
`pair_explorer[.<output-label>].html` plus `report_generation_manifest[.<output-label>].json`. The report retains the detailed
headline, accounting, and `By ...` slice tables, but routes pair-level examples, document excerpts, and identifiers to the Pair
Explorer instead of embedding them in the report body. Stage-level operations, reproducibility data, machine paths, and the
artifact inventory remain in the appendices.

After a derived report is written, the renderer copies everything before the first appendix verbatim to
`eval/dedup/RESULTS.md`. This keeps the repository-visible results synchronized without recomputing or separately summarizing
the metrics. The Pair Explorer separates the observed SUT grouping/action, the Judge group and directional-replacement verdicts,
and the derived evaluation outcome. It also includes endpoint group/action context, Judge evidence-repair coverage, 5b
evaluation-retrieval scores, bounded deterministic group-member summaries, and explicit SUT-provenance availability. Human
review annotations are stored in browser `localStorage` and can be imported or exported as CSV/JSON; they do not modify frozen
run artifacts.

### 6. Complete and import Human QA

Step 8 writes the self-contained `reports/human_qa_dashboard.html`. Its selector switches between the blind sample and
diagnostic set while keeping their progress and CSV exports separate. The reviewer-blind UI omits Judge decisions, payload
hashes, SUT outcomes, and sampling strata. The labels CSV stays compact and directly importable by `qa-import`; the packet JSON
is the self-contained sharing artifact, with each review next to its two Judge-visible documents. The Sarah default and Judge
v1 packets contain only cleaned text and explicit long-document windows. Legacy Judge v0 packets retain their neutral metadata
contract.

When the blind Human QA budget exhausts the candidate population (as it can in the smoke profile), the independent diagnostic
set is empty by construction. Step 8 still writes its empty packet and labels artifacts and records zero counts; the dashboard
then exposes only the non-empty blind sample, and the remaining stages continue normally.

Only three decisions are required:

- `same_duplicate_group`
- `a_can_replace_b`
- `b_can_replace_a`

Relation type, material difference, fuzzy scope, reason codes, and notes are optional. For CSV compatibility, optional Human QA
reason labels retain the legacy flat taxonomy and are independent of the automated Judge's versioned reason structure. Reviews
live in browser `localStorage`, so export the CSV before changing browsers or clearing site data.

For a full run, export the completed **blind-sample** CSV and import it. `qa-import` validates IDs, fields, enums, and reason codes:

```bash
/raid/hfang/llm_judge_env_pr2324_latest/bin/python -m eval.dedup qa-import \
  --run-root /raid/hfang/dedup_eval_runs/<evaluation_run_id>/v0_run \
  --labels /path/to/completed_blind_labels.csv
```

The import writes `reports/human_qa_report.md` and `reports/human_qa_metrics.json` without changing the automated report.
Keep the diagnostic export as a separate challenge/regression result; do not merge it into the blind-sample headline metric.

`dashboard_server.py` serves the latest Human QA dashboard through the stable URL above. It switches only after Step 8 has a
complete marker and never copies, overwrites, or deletes a run-scoped dashboard. Historical run files remain immutable.

## Key outputs

All canonical artifacts live below the immutable `run_root`:

| Stage | Artifact | Purpose |
|---|---|---|
| 6 | `data/judge_results.jsonl` | Schema-valid automated Judge decisions. |
| 6 | `data/human_qa_packet.jsonl`, `data/human_qa_labels.csv` | Frozen blind Human QA packet and template. |
| 8 | `data/pair_comparisons.parquet` | Pair-level SUT-versus-Judge comparisons. |
| 8 | `data/human_qa_diagnostic_packet.jsonl`, `data/human_qa_diagnostic_labels.csv` | Disagreement challenge set. |
| 8 | `reports/human_qa_dashboard.html` | Self-contained Human QA reviewer UI. |
| 9 | `reports/metrics.json`, `reports/metrics_by_slice.csv` | Canonical metrics and slice results. |
| 9 | `reports/pipeline_accounting.csv` | Stage and population accounting. |
| 10 | `reports/final_report.md` | Automated evaluation report. |
| 10 | `reports/pair_explorer.html` | Self-contained result explorer. |

Formal V0 reporting requires at least 99% schema-valid Judge completion. Step 10 remains independent of Human QA completion.

## Module and artifact map

- `handoff/`: validates and wraps Steps 1–2 artifacts.
- `pair_construction/`: writes document outcomes, anchors, the two pair tracks, canonical queue, and provenance.
- `judging/`: constructs blind payloads, handles long documents, calls a provider, validates the schema, retries, and resumes.
- `analysis/`: creates the partial graph, pair comparison, and frame-valid metrics.
- `run.py`: stage transactions and resume validation; domain logic remains in the packages above.
- `report.py`: deterministic fact reporting and example selection, bounded DeepSeek recommendations, report publication,
  and the independent blind human-QA exchange.
- `dashboard.py`: Pair Explorer data joins, bounded group context, static HTML rendering, and local human-review tooling.
- `human_qa_dashboard.py`: reviewer-blind QA packet rendering, browser-local progress, and contract-compatible CSV export.
- `dashboard_server.py`: stable internal URLs for Pair Explorer and the latest completed, immutable Human QA dashboard.

Large generated data remains beneath the configured external run root. Derived automated and Human-QA reports are exported beneath `/home/nfs/hfang/dedup_eval/dedup_eval_runs/<evaluation_run_id>/v0_run/reports/` by default; `eval/dedup/docs/` is reserved for design documents, proposals, and templates rather than run-specific results. Imports are side-effect free.
