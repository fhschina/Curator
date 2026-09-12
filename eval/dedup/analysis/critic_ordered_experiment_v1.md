# Explanation-property-first experiment on unchanged v4 semantics

Offline inspection found that the frozen request writer canonicalizes JSON objects:
all v3/v4 and observed v5 candidate schemas put a_context_span_id, a_loss_span_id,
b_context_span_id, b_loss_span_id and conflict before explanation. Every inspected
strict output followed that property order. Prompt instructions alone did not move
the justification before evidence selection. V5 still explains a bare label is not
a fact while selecting it as material loss. This is a testable generation-order
hypothesis, not proof of why the model made the error.

After all v5 errors are reviewed, keep v4's entire prompt, rubric, input and schema
semantics unchanged. Compare the exact v4 control with v4 whose schema property
explanation alone is moved to the first position. All field names, required lists,
types, enums, instructions, validators and examples remain identical. The candidate
is not v5 plus another rule. Original .12 remains a third fresh control. Same 96
cases, all weights, two repeats, 354 planned calls. No semantic retries or cache reuse.

Canonical semantic hashes cannot distinguish object ordering. Therefore additionally
freeze an order-preserving JSON string and its byte hash, verify both at loading, and
send the restored ordered object through the existing pacing relay. A real loopback
integration test must verify the nested order survives both client and relay. Empirically
check actual NVIDIA response field order separately from semantic accuracy. If the
endpoint ignores this order, report the intervention as unapplied, not ineffective.

Preserve the same three targets and nineteen guards. Pending H0720/H0760 and all old
labels remain in scores. No declaration of full-development or release success from
selected material. Every completed error requires review before the next iteration.
Main .12 and hard window 07:00:52–12:00:52 UTC remain fixed; stop admission 210 s early.
