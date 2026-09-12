# Exp1: conflict-evidence-only patch

The opt-in entry point is `exp1_conflict_evidence_fix.execute_case`, using the
same arguments as the immutable `exp1_reproduction_runtime.execute_case`.
The old runtime and all frozen resources remain unchanged for historical replay.
The checkpoint suffix `+conflict-evidence-fix1` distinguishes repaired outputs;
it is not a release or a replacement of historical results.

Only the coverage critic's `CRITIC_SCOPE_CONFLICT` failure gains one additional
repair opportunity. The same critic prompt, model, generation parameters and
eight-field response schema are retained. Failure-only feedback asks for valid
bilateral context witnesses. Only the two context-span fields and explanation
may change; conflict, loss IDs, overlap basis and shared anchors are frozen.
The original validator still decides whether the proof is admissible. An invalid
repair remains failed; there is no second repair and no guessed evidence.

No sample ID triggers behavior. No main prompt/parser/output-format change,
normal-path critic change, semantic-policy change, transport change, label change,
new exclusion, or Exp4 component is included. Main-format retries are intentionally
unchanged. The H0074/H0075 transport failures are outside this patch.

Validation replays the saved Exp1 requests and responses, preserving every
unaffected result. Only H0537's evidence-repair request may be sent online, once
as a logical call, with the existing model and key. Its saved main answer and
initial critic answer are reused explicitly for this regression check. This is
not a fresh full1000 evaluation, and no historical failure is overwritten.
