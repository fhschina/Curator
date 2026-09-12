# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_subject_context_verify as trial
from eval.dedup.validation import DedupEvaluationError

PARENT = trial.proof.previous.SESSION / "subject-context-v5-pilot32"


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed context trial absent")
def test_all_first_repeat_in_scope_proposals_are_verified(tmp_path):
    root = tmp_path / "proof"
    result = trial.prepare(root, PARENT)
    trial.proof.reference.verify_freeze(root / "manifest.json")
    assert result["population"] == 32
    assert result["logical_requests"] == 6
    rows = {r["canonical_pair_id"]: r for r in json.loads((root / "panel_private.json").read_text())}
    requests = json.loads((root / "requests_frozen.json").read_text())
    assert {rows[r["canonical_pair_id"]]["review_id"] for r in requests} == {"H0606", "H0809", "H0920"}
    assert all(r["body"]["messages"][0]["content"] == trial.proof.SYSTEM.read_text() for r in requests)
    result = trial.proof.assess(root)
    assert not result["proof_verifier_pilot_gate"]
    assert all(result["cells"][f"{r}/candidate"]["engineering_failures"] == 3 for r in (1, 2))


@pytest.mark.skipif(
    not (PARENT.parent / "subject-context-v5-proof-pilot32/review_complete.json").exists(),
    reason="reviewed composed context trial absent",
)
def test_full_preparation_rejects_oversize_context_before_any_artifact(tmp_path):
    root = tmp_path / "full"
    with pytest.raises(DedupEvaluationError, match="CONTEXT_VERIFY_LENGTH"):
        trial.prepare(root, PARENT.parent / "subject-context-v5-proof-pilot32", full=True)
    assert not root.exists()
