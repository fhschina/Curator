# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_subject_kinds_experiment as trial

PARENT = trial.previous.SESSION / "v4-subject-v2-full1000"


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed full result unavailable")
def test_real_panel_freezes_identical_bases_fresh_controls_and_named_schema(tmp_path):
    root = tmp_path / "pilot"
    result = trial.prepare(root, PARENT)
    trial.reference.verify_freeze(root / "manifest.json")
    rows = json.loads((root / "panel_private.json").read_text())
    requests = json.loads((root / "requests_frozen.json").read_text())
    assert result["population"] == 32
    assert result["logical_requests"] == 128
    assert {r["review_id"] for r in rows} == set(trial.PANEL)
    assert all(r["main_public"] == r["coverage_base"]["1"] for r in rows)
    old = {r["canonical_pair_id"]: r for r in json.loads((PARENT / "specialist/requests_frozen.json").read_text())}
    for request in requests:
        assert "human_" not in str(request["body"])
        assert "review_id" not in str(request["body"])
        if request["arm"] == "subject_control":
            assert request["request_sha256"] == old[request["canonical_pair_id"]]["request_sha256"]
        else:
            schema = request["body"]["response_format"]["json_schema"]["schema"]
            assert "a_subject_kind" in schema["required"]


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed full result unavailable")
def test_missing_responses_remain_in_denominator_and_cannot_pass_pilot(tmp_path):
    root = tmp_path / "pilot"
    trial.prepare(root, PARENT)
    (root / "responses").mkdir()
    result = trial.assess(root)
    assert not result["named_target_pilot_gate"]
    assert not result["full_development_precision_recall_75"]
    for name, cell in result["cells"].items():
        assert cell["scores"]["partial_draft"]["weighted"]["rows"] == 32
        if "saved_coverage" not in name:
            assert cell["engineering_failures"] == 32
            assert cell["scores"]["partial_draft"]["missing_output_pairs"] == 32
