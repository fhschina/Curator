# Exp5: shared-anchor numbering tolerance

Opt-in entry point: `exp5_anchor_id_fix.execute_case`, with the same arguments as
the frozen Exp5 runtime. The distinct runtime identifier is
`v0.6.2.33-exp5+anchor-id-fix1`; it is not a renamed Exp5 result or a release.
The existing `exp5_full20k` entry point deliberately remains frozen.

Only the initial coverage critic's `shared_anchor_ids` gain deterministic
zero-padding: e.g. `S41` → `S041`, `S21` → `S021`, `S1` → `S001`.
The target must already exist as a unique, fully aligned shared span in the
original payload. The existing inventory and proof validators remain authoritative.
Unknown IDs, duplicate IDs (including duplicates created by normalization),
wrong prefixes/kinds, ambiguous inventories, whitespace, non-ASCII digits and
over-padded IDs are not accepted. Corrections are all-or-nothing and preserve
list length/order; IDs are never guessed, dropped, or deduplicated.

Loss/context IDs, evidence text, semantic fields, main and critic prompts,
generation settings, routing and retry limits are unchanged. The previous
conflict-evidence repair remains limited to its original trigger and editable
fields. The generation/output-limit failure is outside this patch.

Each correction records the original review, normalized review and index-specific
mapping in `coverage_anchor_id_repair`; the original response receipt is retained
unchanged. Normalization makes zero extra model calls. Downstream stages must
still run when required; fixing an ID alone is not proof of semantic correctness
or a substitute for a missing downstream response.

Use a new run root and contract/source digest for any future patched collection.
Do not overwrite or relabel the completed Exp5 20K run, reuse its cache for a
fresh benchmark, or count an offline replay as a fresh model evaluation.
