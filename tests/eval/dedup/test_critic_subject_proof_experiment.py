# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_subject_proof_experiment as trial

PARENT = trial.previous.SESSION / "subject-kinds-v3-pilot32"


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed target-kind trial absent")
def test_only_actual_fixed_vetoes_are_called_and_original_text_is_preserved(tmp_path):
    root = tmp_path / "pilot"
    result = trial.prepare(root, PARENT)
    trial.reference.verify_freeze(root / "manifest.json")
    rows = json.loads((root / "panel_private.json").read_text())
    by = {r["canonical_pair_id"]: r for r in rows}
    requests = json.loads((root / "requests_frozen.json").read_text())
    assert result["logical_requests"] == 8
    assert result["population"] == 32
    assert {by[r["canonical_pair_id"]]["review_id"] for r in requests} == {"H0606", "H0669", "H0809", "H0820"}
    for request in requests:
        row = by[request["canonical_pair_id"]]
        text = request["body"]["messages"][1]["content"]
        assert row["payload"]["document_a"]["text"] in text
        assert row["payload"]["document_b"]["text"] in text
        assert row["payload"]["subject_proposal"]["explanation"] not in text
        assert "human_" not in str(request["body"])
        assert "review_id" not in str(request["body"])


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed target-kind trial absent")
def test_missing_verifications_are_not_counted_as_safe_bypasses(tmp_path):
    root = tmp_path / "pilot"
    trial.prepare(root, PARENT)
    (root / "responses").mkdir()
    result = trial.assess(root)
    assert not result["proof_verifier_pilot_gate"]
    assert not result["full_development_precision_recall_75"]
    for rep in (1, 2):
        cell = result["cells"][f"{rep}/candidate"]
        assert cell["scores"]["partial_draft"]["weighted"]["rows"] == 32
        assert cell["engineering_failures"] == 4
        assert cell["scores"]["partial_draft"]["missing_output_pairs"] == 4
