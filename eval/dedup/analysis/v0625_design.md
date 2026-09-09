# V0.6.2.5 boundary-delta design

> Final status: rejected on the 127-row residual smoke. The run completed 127/127 with one retry, but containment
> over-group was 25, under-group was 21, only 2/5 protected duplicates survived, only 18/23 diagnostic negatives remained
> separate, and only 6/30 true containments had the correct direction tuple. See `v0625_residual_smoke.md` and
> `v0625_residual_smoke_summary.json`. No full-development or holdout run is authorized.

## Decision context

V0.6.2.4 is preserved as a rejected immutable experiment. On the cumulative-label 127-row residual smoke it reduced
over-group from 46 to 25 and containment over-group from 22 to zero, but produced no containment prediction, increased
under-group from five to 21, retained only two of five protected benign duplicates, preserved only 21 of 23 newly reviewed
negative labels, completed only 126/127 rows, and retried seven pairs. The full diagnosis is in
`v0624_residual_smoke.md`.

The V0.6.2.4 fields were useful diagnostically, but its deterministic resolution made two opposite mistakes:

- `MATERIAL_PROPOSITION_OR_RECORD` was an unconditional no/no veto, so a legitimate one-sided addition to a verified same
  record could never become containment;
- `SAME_NON_MAIN_MESSAGE` plus `COMPLETE_NON_MAIN_MESSAGE` remained permissive when the model mislabeled a page-specific
  title, category/product slot, profile, image/page target, date, count, page role, or changed legal/cookie proposition as
  harmless navigation or formatting.

The independent evidence mini-contract inside `record_identity_support` also created six recovered retries and one terminal
failure even though the normal bilateral quote-evidence contract was available.

## Immutable candidate

V0.6.2.5 uses prompt version `dedup-judge-hs-v0.6.2.5` with the unchanged published `dedup-judge-output-v3` schema. It keeps
the eleven V0.6.2.4 ledger fields and adds `boundary_delta_class`:

- `NONE_OR_FORMATTING`;
- `UNIVERSAL_UI_OR_REDUNDANT_REPETITION`;
- `SAME_RECORD_CONTENT_EXTENSION`;
- `RECORD_IDENTITY_ROLE_OR_STATE_CHANGE`;
- `MATERIAL_NON_MAIN_MESSAGE_CHANGE`;
- `TWO_SIDED_CONTENT_CHANGE`;
- `UNRESOLVED`.

The new field creates a distinct V4 ledger shape and a distinct adapter branch, so V0.6.2.4 results remain reproducible.
V0.6.2.5 removes only the redundant field-specific identity-quote validation; every resolved non-exact result still needs
the published general A/B byte-aligned exact evidence.

The adapter applies this resolution order:

1. unreadable or unresolved gates produce the fully unresolved/low result;
2. hard conflict, different identity, record/role/state change, material non-main-message change, and two-sided content
   change produce no/no and major;
3. a same-record content extension produces containment only when both profiles are substantive, coverage is exactly
   one-sided, the basis is substantive, alignment and scope are same-record/document-wide, and identity support is positive;
4. none/formatting and universal UI/repetition produce yes/yes only with equivalent coverage and verified same-record or
   complete-message scope;
5. every incompatible or insufficient combination is reconciled to no/no rather than triggering a retry.

The prompt clarifies that a specific policy, notice, form, restriction, or message is substantive when it is the retained
record itself. It also distinguishes genuinely universal controls from record-bearing short labels: a product/category,
profile, image/file/page target, count, date, status, list, or page-function value is material regardless of length.

Boundary examples cover the observed families without including review IDs: a recurring-purchase clause plus a lone product
name; cookie text plus a page/service heading; different edit targets and image/profile names; changed counts/dates/states;
added analytics/advertising/GDPR/category propositions; forum descriptions and auth targets; harmless wishlist/accept/
learn-more/payment controls; harmless stories/byline headers; placeholder repetition; verified one-sided record extension;
and faithful versus changed translation.

## Frozen residual-smoke admission

Run the immutable candidate on exactly the same 127 residual IDs against `v0624_policy_reconciled_labels_1000.csv`. Advance
only if every check passes:

- total over-group is at most 32;
- containment over-group is at most 15;
- under-group is at most five;
- all five protected benign duplicates `H0038`, `H0347`, `H0453`, `H0679`, and `H0748` remain grouped;
- all 23 labels changed in the diagnostic follow-up and preflight revision remain `NO`;
- at least eight of the 30 true containment rows have the complete human direction tuple exactly correct;
- schema completion is 100%, terminal errors are zero, and pair-level retry rate is at most 2%.

The containment-direction floor is new. It prevents a candidate from passing the false-containment gate merely by eliminating
containment as a relation. The floor is intentionally modest for this adversarial slice but materially exceeds the one exact
containment direction produced by V0.6.2.3 and the zero produced by V0.6.2.4.

Failure freezes V0.6.2.5 as rejected and prevents a full-development or holdout run.

## Full development and holdout boundary

If the smoke passes, run all 1,000 cumulative development labels with identical model, decoding, and payload settings. Reuse
the V0.6.2.4 full-development thresholds: weighted precision and recall at least 75%, weighted primary exact at least 87%,
over-group at most 50, containment over-group at most 18, no more than a three-point regression in identity,
meaningful-addition, or translation cohorts, 100% completion, zero terminal errors, and retry rate at most 1%.

Only an all-gates development pass may expose the frozen 400-pair holdout. The holdout remains unseen until then. The
V0.6.2.5 smoke uses a fresh run root, contract digest, and cache; no V0.6.2.4 cache entry is migrated or reused.
