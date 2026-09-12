# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_subject_scope_experiment as subject


@pytest.mark.skipif(
    not (subject.trial.SESSION / "subject-v1-savedbase96/review_complete.json").exists(),
    reason="reviewed specialist trial unavailable",
)
def test_real_offline_scope_replay_keeps_every_denominator_and_original_response(tmp_path):
    source = subject.trial.SESSION / "subject-v1-savedbase96"
    root = tmp_path / "scope"
    result = subject.replay(root, source)
    subject.trial.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    assert result["online_model_calls"] == 0
    assert result["population"] == 96
    assert result["candidate_repeat_disagreements"] == []
    assert result["new_errors_from_previously_correct"] == {"1": [], "2": []}
    assert result["clear_guard_failures"] == {"1": [], "2": []}
    original = json.loads((source / "assessment.json").read_text())
    for rep in (1, 2):
        before = original["cells"][f"{rep}/candidate"]["pairs"]
        after = result["cells"][f"{rep}/candidate"]["pairs"]
        changed = [a["review_id"] for a, b in zip(before, after, strict=True) if a["primary"] != b["primary"]]
        assert changed == (["H0312"] if rep == 1 else [])
        assert result["cells"][f"{rep}/candidate"]["scores"]["partial_draft"]["weighted"]["rows"] == 96
