# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_cause_accounting as subject
from eval.dedup.validation import DedupEvaluationError

SOURCE = subject.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v3-material96"


def test_missing_cause_or_missing_row_never_becomes_clear_error_or_smaller_denominator():
    row = {"review_id": "x", "canonical_pair_id": "p", "draft_label": {"sample_weight": 4}}
    cell = {"pairs": [{"review_id": "x", "draft_error": "OVER_GROUP", "status": "VALID"}]}
    with pytest.raises(DedupEvaluationError):
        subject.cell_ledger([row], cell, {})
    with pytest.raises(DedupEvaluationError):
        subject.cell_ledger([row], {"pairs": []}, {})


@pytest.mark.skipif(not (SOURCE / "review_complete.json").exists(), reason="local reviewed source unavailable")
def test_real_cause_contributions_keep_pending_errors_and_reconcile_full_scores(tmp_path):
    root = tmp_path / "accounting"
    subject.build(root, SOURCE)
    subject.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    result = json.loads((root / "cause_accounting.json").read_text())
    assert result["population"] == 96
    first = result["cells"]["1/candidate"]
    assert len(first["ledger"]) == 96
    pending = first["accounting"]["by_cause"]["REFERENCE_POLICY_DISPUTE"]
    assert pending["fp_pairs"] == 17
    assert pending["fp_weight"] == pytest.approx(178.89920743639917)
    assert first["scores_unchanged"]["duplicate_precision"] == pytest.approx(0.6373572169205372)
    clear = first["accounting"]["by_cause"]["CLEAR_CRITIC_AND_LEGACY_SEMANTIC_ERROR"]
    assert clear["fp_pairs"] == 1
    assert clear["review_ids"] == ["H0907"]
