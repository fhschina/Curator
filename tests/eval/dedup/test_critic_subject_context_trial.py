# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_subject_context_trial as trial

PARENT = trial.paired.previous.SESSION / "subject-proof-v4-fresh-upstream/verification"


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed upstream stability trial absent")
def test_paired_requests_change_only_source_presentation(tmp_path):
    root = tmp_path / "pilot"
    manifest = trial.prepare(root, PARENT)
    trial.paired.reference.verify_freeze(root / "manifest.json")
    assert manifest["logical_requests"] == 128
    assert manifest["population"] == 32
    requests = json.loads((root / "requests_frozen.json").read_text())
    by = {(r["repeat"], r["canonical_pair_id"], r["arm"]): r for r in requests}
    for (rep, pid, arm), request in by.items():
        if arm != "candidate":
            continue
        control = by[(rep, pid, "subject_control")]["body"]
        candidate = request["body"]
        assert {k: v for k, v in candidate.items() if k != "messages"} == {
            k: v for k, v in control.items() if k != "messages"
        }
        assert candidate["messages"][0] == control["messages"][0]
        assert candidate["messages"][1]["content"].endswith(control["messages"][1]["content"])
        assert request["input_tokens"] + 6144 <= 32768
    result = trial.paired.assess(root)
    for rep in (1, 2):
        for arm in ("candidate", "subject_control"):
            assert result["cells"][f"{rep}/{arm}"]["engineering_failures"] == 32
    assert not result["named_target_pilot_gate"]
