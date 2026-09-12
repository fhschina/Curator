# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_subject_experiment as subject


@pytest.mark.skipif(
    not (subject.SESSION / "retention-v6-sourceorder96/review_complete.json").exists(),
    reason="reviewed source-order outputs unavailable",
)
def test_real_saved_base_trial_freezes_no_new_general_judge_calls(tmp_path):
    root = tmp_path / "subject"
    result = subject.prepare(root, subject.SESSION / "retention-v6-sourceorder96")
    subject.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    assert result["population"] == 96
    assert result["logical_requests"] == 34
    assert result["main_online_calls"] == result["coverage_online_calls"] == 0
    rows = json.loads((root / "panel_private.json").read_text())
    requests = json.loads((root / "requests_frozen.json").read_text())
    assert len({r["canonical_pair_id"] for r in requests}) == 17
    assert all(r["arm"] == "candidate" for r in requests)
    primary = subject.selected.base.previous.reference.primary
    assert all(primary(r["coverage_base"]["1"]) == primary(r["coverage_base"]["2"]) for r in rows)
    assert all("human_" not in str(r["body"]) and "review_id" not in str(r["body"]) for r in requests)
    result = subject.assess(root)
    assert all(result["cells"][f"{rep}/candidate"]["engineering_failures"] == 17 for rep in (1, 2))
    assert all(
        result["cells"][f"{rep}/candidate"]["scores"]["partial_draft"]["weighted"]["rows"] == 96 for rep in (1, 2)
    )
    assert result["full_development_75_gate"] is False
