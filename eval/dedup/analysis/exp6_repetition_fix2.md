# Exp6: citation-preserving repetition recovery candidate

Opt-in entry point: `exp6_repetition_fix2.execute_case`, runtime identifier
`v0.6.2.33-exp6+main-repetition-fix2`. This is not a change to frozen Exp6.

The detector, one-call recovery budget, original parser/adapter/evidence checks,
normal-path behavior, model, generation settings and downstream stages are
identical to `exp6_repetition_fix`. Only the static failure feedback is clarified:
avoid repetitive source-name lists, but retain actual concrete span citations
in every reasoning field where the original rubric requires them.

The first candidate repaired P01329's output loop, but its wording discouraged
identifier enumeration so broadly that P14187 omitted required A/B citations.
Its raw no/no judgment became UNRESOLVED at evidence validation. P06727 remained
unresolved but also omitted citations. Both first-candidate results are retained;
this clarification does not patch their answers, relax validators, infer missing
IDs, or change semantic labels. All three trigger cases must be compared again
under the same frozen failure inputs, one recovery call per case.

Historical records for the first diagnostic and its stopped, zero-call execution
harness are preserved. A successful repair is an engineering result, not proof
of calibrated semantic accuracy or a fresh 20K evaluation. The upstream token
limit/report discrepancy remains unverified and outside this patch.
