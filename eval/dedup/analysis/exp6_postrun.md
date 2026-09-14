# Exp6 post-run checkpoint and local coverage-format recovery

The completed `v0.6.2.33-exp6+main-repetition-fix2` 20K run is retained as an
experimental checkpoint, not a release. Old sources, results, failures,
reference labels and exclusions remain immutable. `checkpoint` records the
user's acceptance and binds the completed report, manifests and transport log.

The opt-in candidate `v0.6.2.33-exp6+coverage-format-fix1` includes that exact
runtime plus two failure-only changes. Within `shared_anchor_ids`, a bare
three-ASCII-digit reference may gain the S prefix only if the original
bilaterally aligned shared inventory contains that exact canonical reference.
Correction is all-or-nothing, preserves order and rejects collisions. Loss and
context references are unchanged; the old zero-padding behavior is preserved.

Only an initial coverage response ending with `finish_reason=length`, invalid
JSON and a 4,096-character suffix at least 98% literal tab characters receives
one format-only recovery. The complete original request and schema are kept,
with static feedback appended. The failed text is not fed back. Other errors,
valid JSON, ordinary truncation and other stages do not gain this retry. The
recovered result must pass all original evidence and semantic validators.
Existing downstream reviews and conflict-evidence handling are unchanged.

`prepare` freezes new source/test hashes in a separate target directory.
`offline` replays every old receipt, compares all unaffected complete results
except the version field, and records any missing new stage without network
access. The original two failed pairs are the only authorized differences.
`recover` requires that gate, retains exact main/coverage receipts, and collects
only the missing recovery or mandatory downstream review, never another main
answer. Each isolated case has an eight-external-attempt ceiling; completed
or failed recovery directories cannot be overwritten or retried until success.
The resulting pipeline must replay identically offline. This is a local
engineering recovery, not a fresh benchmark and not semantic accuracy proof.

The content-addition review is a separate assistant review of saved source
texts and outputs, not independent blind gold. Observed direction errors,
policy/reference disputes, and evidence/routing causes are kept distinct.
No review suggestion silently updates the reference or scores.
