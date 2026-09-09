# V0.6.2.7 design: independent record-binding critic

V0.6.2.7 keeps the immutable V0.6.2.6 main Judge and `judge-visible-payload-v3`, then adds a second independent model call
over the same blind deterministic span packet. The critic has one narrow task: distinguish a true one-sided extension of an
atomic record from a different article, product, profile, event, list, page role, or content record attached to reusable
context.

The critic cites only stable S###/A###/B### IDs. The adapter validates the IDs and bilateral basis. A main-Judge containment
survives only when the critic returns `ATOMIC_SAME_RECORD_EXTENSION`; a benign delta becomes bidirectional near-surface;
separate-record/template attachment or two-sided/conflicting content becomes no/no; missing, invalid, or unresolved critic
evidence fails closed to unresolved. Non-containment main decisions are unchanged in this first critic version.

The output remains `dedup-judge-output-v3`. V0.6.2.7 has a new prompt version, runner resource, contract digest, run root, and
cache. It does not alter or reuse V0.6.2.6 results.
