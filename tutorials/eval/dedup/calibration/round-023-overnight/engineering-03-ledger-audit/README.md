# Literal ledger coherence audit

The native generator, parser, evidence checks and raw decisions stay unchanged.
This read-only check enforces the frozen requirement that a positive replacement
cannot contradict its own uncovered-material delta, material hard conflict or
absence of a verified retained basis. It removes credit from inconsistent rows;
it never changes a decision or adds a positive judgment. The same check applies
to baseline, starting and candidate versions with the original denominators.

`compare_strict.py` first retains the native comparison in a separate directory,
then reuses the native agreement and aggregation functions for the checked view.
Raw inputs and full model exchanges remain in the external experiment directory.
`manifest.json` freezes the three source hashes. The integration receipt verifies
unchanged native file content, denominators and raw error signals and confirms
that the checked view never increases credit. It made no model calls.

Run using the already installed environment, with the checkout root and
`tutorials/eval/dedup` on `PYTHONPATH`:

```bash
python compare_strict.py --session /absolute/experiment/session \
  --checkout /absolute/Curator-checkout --best-root candidate-02 \
  --output new-unique-comparison
```

The caller must provide the unchanged frozen run roots and a new output name.
All legacy reports and attempt failures are preserved. Audit source changes
require a separate version and re-audit; they do not authorize model resampling.
