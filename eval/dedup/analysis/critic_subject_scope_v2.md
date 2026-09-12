# Subject specialist action-scope ablation

After reviewing every error in subject v1, keep exactly the same model outputs,
input payloads, schema, prompt and saved v4 coverage results. The only change is
which specialist findings may veto: LIABILITY_PARTY, POLICY_SERVICE, FAILED_OBJECT.
ACCESS_TARGET and RECORD_SUBJECT remain diagnostics, not authorized actions. This
does not remove v4's own identity/access/role protections. It only limits the new
specialist that mistook Czech spacing/grammar for a different access location.

All proofs are validated before action filtering. Invalid output is still an
engineering failure, not silently accepted. No case ID, reference label, language,
particular name or spelling is used by the scope rule. No fresh model calls or
reference changes occur in this ablation; both repeats and all 96 denominators
remain. The narrow deterministic scope can be verified exactly on saved answers.

Only if H0606 is corrected twice, the frozen paired targets/guards plus H0017 are
correct, no previously correct panel case is newly wrong, and no proof fails, may
a separately frozen full original 1000-case pipeline diagnostic start after the
complete cause review. All historical and partial-draft references remain separate.
This does not establish the 75% gate, release readiness, or independent holdout gain.
