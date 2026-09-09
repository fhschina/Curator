# V0.6.2.9 design: asymmetric critic arbitration

V0.6.2.9 reuses the immutable V0.6.2.6 span Judge and V0.6.2.8 critic prompts. It changes only the deterministic
arbitration policy, so the residual experiment isolates ownership between the two existing semantic ledgers.

The main span ledger owns:

- equivalence when both content profiles are `NON_MAIN_ONLY`; a critic trained for record binding cannot reopen that
  decision, while a main-ledger material non-main delta still yields no/no before arbitration;
- a cited `SAME_RECORD_CONTENT_EXTENSION` when the critic calls the unique text benign;
- a cited partial/additive translation extension. Critic disagreement cannot reverse its direction and instead caps
  confidence at `LOW`.

The independent critic retains veto authority for substantive `SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT`,
`TWO_SIDED_OR_CONFLICTING`, and `NON_MAIN_POLICY_OR_STATE_CHANGE` verdicts. Invalid or unresolved critic evidence still
fails closed outside the separately owned non-main and additive-translation branches.

The public output remains `dedup-judge-output-v3`. The candidate has a new prompt version, runner resource, contract
digest, run root, and cache; no V0.6.2.8 artifact is migrated or reused.
