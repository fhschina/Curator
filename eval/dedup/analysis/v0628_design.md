# V0.6.2.8 design: calibrated record-binding critic

V0.6.2.8 keeps the V0.6.2.6 span-grounded main Judge and the independent V0.6.2.7 critic architecture. It changes only the
critic rubric and adapter policy:

- add explicit boundaries for FAQ instructions, reordered same-record fields, trailing story teasers, forum search pages,
  product attributes attached to checkout/store policy, cookie-policy state, and collection-list membership;
- allow the critic to veto positive substantive-main decisions, not only containment;
- add a `NON_MAIN_POLICY_OR_STATE_CHANGE` verdict while otherwise leaving non-main equivalence to the main span ledger;
- retain fail-closed citation validation and the unchanged `dedup-judge-output-v3` public contract.

The version has a new runner resource, prompt version, contract digest, run root, and cache. V0.6.2.7 remains immutable.
