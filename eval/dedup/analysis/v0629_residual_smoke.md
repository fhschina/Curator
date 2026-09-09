# V0.6.2.9 residual-smoke result

## Outcome

V0.6.2.9 is rejected and must not advance to the full development set or holdout. The independent run completed with
127/127 valid results, zero pair-level retries, and zero terminal errors. Asymmetric arbitration recovered recall from
38.10% to 80.95%, reduced under-group from 13 to four, restored all five protected benign duplicates, and recovered three
of the four reconciled true-containment directions.

It nevertheless failed two frozen semantic gates. `H0572` reopened one of the 23 diagnostic negatives because the adapter
unconditionally deferred a `NON_MAIN_POLICY_OR_STATE_CHANGE` critic verdict to an overly permissive main non-main ledger.
`H0017` became complete translation in the stochastic main output and was then made unresolved by the critic, so only 3/4
true-containment direction tuples were exact.

## Reconciled result

| Check | V0.6.2.8 | V0.6.2.9 | Residual requirement | Result |
|---|---:|---:|---:|---|
| total over-group | 19 | 20 | at most 32 | pass |
| containment over-group | 1 | 1 | at most 15 | pass |
| total under-group | 13 | 4 | at most 5 | pass |
| protected benign duplicates | 3/5 | 5/5 | 5/5 | pass |
| true containment primary tuple | 0/4 | 3/4 | 4/4 | fail |
| original diagnostic negatives preserved | 23/23 | 22/23 | 23/23 | fail |
| schema completion | 100% | 100% | 100% | pass |
| pair-level retries | 0 | 0 | at most 2% | pass |
| terminal errors | 0 | 0 | 0 | pass |

On this error-enriched slice, unweighted precision is 45.95%, recall 80.95%, primary-tuple exact is 76.38%, and taxonomy
exact is 51.97%. Weighted precision is 43.70%, recall 77.65%, primary-tuple exact is 76.39%, and taxonomy exact is 53.36%.
These slice values are development diagnostics, not corpus-wide quality estimates.

## Failure analysis

For `H0572`, both main profiles were `NON_MAIN_ONLY`. The main ledger called a 703-character cookie category and consent
expansion universal UI, while the critic correctly identified a policy/state change. The new unconditional ownership rule
discarded that warning and produced yes/yes. Non-main arbitration therefore needs a separate structured distinction between
wrapper/chrome and a cited category, purpose, permission, controller, or consent consequence; neither current verdict alone
is reliable enough.

For `H0017`, the main Judge changed from the prior run's additive translation to `COMPLETE_FAITHFUL`, despite Arabic B adding
publication, social-account, and service information. The critic noticed the lexical binding weakness but returned
`UNRESOLVED`, so the additive-translation ownership branch never ran. This is model variance at the translation boundary,
not an adapter-direction bug.

`H0748` remained grouped, satisfying the protected gate, but the main ledger's false content-extension label was no longer
flattened by the benign critic. `H0723` remains the sole false containment: an added cookie-inventory entry is still treated
as an atomic substantive-record extension instead of a non-main policy/list difference.

## Next boundary

V0.6.2.9 should remain frozen. A future candidate should add two narrow internal diagnostics rather than another broad
ownership override:

1. a structured non-main delta subtype separating wrapper/chrome from policy proposition, state, and page-context changes;
2. a bilateral translation delta direction (`EQUIVALENT`, `A_ADDS`, `B_ADDS`, `CONFLICTING`, or `UNRESOLVED`) that does not
   depend on literal shared-span identity.

It should also explicitly route cookie inventories and purpose lists through the non-main/list policy before containment.
No full-development run, label change, or holdout access is justified from this candidate.
