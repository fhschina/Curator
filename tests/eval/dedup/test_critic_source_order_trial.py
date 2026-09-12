# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_source_order_trial as subject


@pytest.mark.skipif(
    not (subject.SESSION / "retention-v4-order96-transport-v2/review_complete.json").exists(),
    reason="latest review unavailable",
)
def test_real_trial_freezes_latest_review_but_changes_only_presentation_against_v4(tmp_path):
    root = tmp_path / "source-order"
    result = subject.prepare(root)
    subject.paired.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    assert result["population"] == 96
    assert result["logical_requests"] == 354
    assert result["transport_contract"] == "dedup-paced-transport-v2"
    assert "candidate_schema_property_order" not in result
    assert result["latest_reviewed_run"].endswith("retention-v4-order96-transport-v2")
    assert any(p.endswith("retention-v4-order96-transport-v2/review_complete.json") for p in result["sources"])
    requests = json.loads((root / "requests_frozen.json").read_text())
    controls = {(r["repeat"], r["canonical_pair_id"]): r for r in requests if r["arm"] == "encoding_control"}
    for request in (r for r in requests if r["arm"] == "candidate"):
        control = controls[(request["repeat"], request["canonical_pair_id"])]
        assert {k: v for k, v in request["body"].items() if k != "messages"} == {
            k: v for k, v in control["body"].items() if k != "messages"
        }
        assert request["body"]["messages"][0] == control["body"]["messages"][0]
        assert request["body"]["messages"][1] != control["body"]["messages"][1]
        assert "ordered_body_json" not in request
