# Complete directional ledger coherence audit

This revision additionally checks an explicit negative against its own covered
removed-side delta. The check requires a verified retained basis, no hard
conflict and `not_translation`; protected substantive body translations are not
inferred harmless. Positive constraints from engineering 03 remain unchanged.
The frozen requirement already forbids field contradictions. No new semantic
reference, label, model, arbiter or inference retry is introduced.

Every baseline, starting and candidate output receives the same read-only check.
Contradictions lose all row credit while keeping raw decisions and error signals.
Original denominators and scoring functions remain unchanged. The integration
receipt checks both the native projection and the prior audit, with no credit
increase, and verifies the native file hashes remain unchanged.

New per-run artifacts use `audits/engineering04-ledger-coherence/`, preserving
all original native and engineering 03 artifacts. Old snapshots remain intact;
current pointers are recomputed under the complete audit, with the supersession
record retained. New snapshots copy row artifacts into a content-addressed store.

Use the already installed environment with the checkout and
`tutorials/eval/dedup` on `PYTHONPATH`:

```bash
python compare_strict.py --session /absolute/experiment/session \
  --checkout /absolute/Curator-checkout --best-root candidate-02 \
  --output new-unique-comparison
```

The twelve actual CPU checks include the earlier six checks and six additional
negative-direction/translation-boundary checks. See the source manifest for
exact evaluated and repository-formatted hashes; formatting removes only an
unused lint suppression. Complete source text and model exchanges stay external.
