# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from eval.dedup.analysis import critic_paired_trial as subject

PREDECESSOR = subject.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v3-material96"


@pytest.mark.skipif(not (PREDECESSOR / "review_complete.json").exists(), reason="local reviewed material unavailable")
def test_frozen_real_trial_compares_immediate_control_on_all_original_material(tmp_path):
    root = tmp_path / "trial"
    result = subject.prepare(
        root,
        PREDECESSOR,
        "eval.dedup.judging.critic_retention_v4",
        subject.selected.base.previous.RESOURCES / "v06212_retention_v4.yaml",
        Path(subject.__file__).with_name("critic_retention_v4_protocol.md"),
    )
    subject.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    assert result["population"] == 96
    assert result["logical_requests"] == 354
    requests = json.loads((root / "requests_frozen.json").read_text())
    for candidate in (r for r in requests if r["arm"] == "candidate"):
        control = next(
            r
            for r in requests
            if r["arm"] == "encoding_control"
            and r["repeat"] == candidate["repeat"]
            and r["canonical_pair_id"] == candidate["canonical_pair_id"]
        )
        assert candidate["body"]["response_format"] == control["body"]["response_format"]
        assert candidate["body"]["messages"] != control["body"]["messages"]
        assert {k: v for k, v in candidate["body"].items() if k != "messages"} == {
            k: v for k, v in control["body"].items() if k != "messages"
        }
    subject.selected.assess(root)
    paired = subject.paired_assessment(root)
    assert paired["full_development_75_gate"] is False
    assert paired["release_eligible"] is False
    for cell in paired["cells"].values():
        assert cell["candidate_engineering_failures"] == 59
        assert cell["target_failures"] == list(subject.TARGETS)
