# Exp6: failure-only main repetition recovery candidate

Opt-in entry point: `exp6_repetition_fix.execute_case`, runtime identifier
`v0.6.2.33-exp6+main-repetition-fix1`. Frozen Exp6 and all historical artifacts
remain unchanged. This candidate is not a renamed Exp6 release or a fresh benchmark.

The original main request, native parsing/format retries, prompts, model,
generation settings and downstream critics remain unchanged unless all these
conditions hold for a main response:

- Exactly one response choice terminated with `finish_reason=length`.
- The visible answer is at least 4,096 characters and is not a complete JSON
  value, allowing the normal Markdown JSON fence.
- Its last 4,096 characters contain alphanumeric content and have a period of
  1–128 characters with at least 98% agreement (at least 32 cycles in the window).

Detection is independent of pair IDs, reference labels and specific repeated
strings. It runs before the native parser sees the failed response. A dedicated
control signal bypasses SDK correction so the large repetitive answer is never
fed back as an assistant message.

Exactly one logical recovery call is allowed per affected pair. It reuses the
entire frozen initial request and adds only a static, content-free format note:
produce all required fields, keep explanations brief, do not repeat identifiers,
and continue applying the unchanged semantic rubric and evidence requirements.
The failed text and intermediate correction conversations are omitted. Model,
temperature and output-token limits do not change. The original parser, adapter,
schema and evidence validation must all accept the recovered answer.

Failure of that recovery remains an engineering failure: no second recovery,
no subsequent native main retry, no guessed JSON completion, and no invented
semantic decision. Successful recovery continues through the unchanged Exp6
coverage, subject and verifier routes. Receipts and the trigger/attempt metadata
are retained separately from the frozen historical run.

Ordinary non-repetitive failures and complete JSON outputs do not activate this
path. In particular, the patch does not fix or claim to diagnose the upstream
4,096-requested versus 32,768-reported token discrepancy.

Validation must distinguish protection cases that never trigger from historical
cases that already recovered through native retries. The latter need paired
checks before promoting this candidate; a repaired terminal failure alone is
not evidence of net semantic improvement. Online diagnosis must use new run
roots and explicitly record any saved-response replay, never overwrite old
results or count recovery-only calls as a fresh full evaluation.
