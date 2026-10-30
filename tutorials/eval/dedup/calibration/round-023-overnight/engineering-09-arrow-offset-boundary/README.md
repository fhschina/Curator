# Arrow/Pandas source-offset compatibility

The first exact-CORE index passed 998 direct-record renders but failed native
execution: Arrow/Pandas converts nullable integer span offsets to floating
values. Python string slices require integer offsets. Native early shutdown
preserved null outputs and missing inference receipts; the integrity boundary
stopped the run. Its original slot and all actual calls remain intact.

A real Arrow/Pandas conversion reproduces 18 old-template render failures on
the 44-row preflight panel. Explicit integer casts at string slice boundaries
allow all 44 to render. Two independent conditional branches preserve the native
AST depth limit of ten; failed depth-eleven drafts remain external. The revised
index passes 998 native render projections, including 30 with an index, and
proves every indexed nonwhitespace character was already visible in the
opposite original packet. Removing only the index restores each original
rendered prompt byte-for-byte. These are CPU checks with zero model calls.

The actual native xref1 evaluation failure is not repaired retrospectively.
Xref2 has a distinct frozen prompt hash and a separately predeclared group.
No native Python, model, sampling, visible window, parser or validator changes.
The live GPU outcome is tracked separately from the passing CPU proof.
