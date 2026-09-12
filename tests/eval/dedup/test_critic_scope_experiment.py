# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from eval.dedup.analysis import critic_scope_experiment as subject
from eval.dedup.validation import DedupEvaluationError, sha256_json


def test_duplicate_json_keys_and_extra_control_fields_are_not_pruned():
    with pytest.raises(DedupEvaluationError, match="DUPLICATE_JSON_KEY"):
        subject.strict_json('{"action":"KEEP_MAIN","action":"REJECT_BOTH"}')
    with pytest.raises(DedupEvaluationError, match="CONTROL_SCHEMA"):
        subject.parse_output("control", '{"record_binding_verdict":{},"retained_conflict":{},"extra":true}', {})


def test_existing_output_root_is_not_overwritten(tmp_path):
    with pytest.raises(DedupEvaluationError, match="ROOT_EXISTS"):
        subject.prepare(tmp_path)


@pytest.mark.skipif(
    not (subject.REFERENCE_ROOT / "summary.json").exists(), reason="local frozen development artifacts unavailable"
)
def test_real_prepare_freezes_main_requests_and_all_39_and_64_members(tmp_path):
    root = tmp_path / "scope"
    result = subject.prepare(root)
    subject.reference.verify_freeze(root / "manifest.json")
    rows = json.loads((root / "panel_private.json").read_text())
    requests = json.loads((root / "requests_frozen.json").read_text())
    assert sum(r["in_changed39"] for r in rows) == 39
    assert sum(r["in_original64"] for r in rows) == 64
    assert sum(r["in_pilot"] for r in rows) == 19
    assert result["main_online_calls"] == 0
    assert result["reference_changed"] is False
    assert result["logical_requests"] <= 76
    assert result["generation"]["temperature"] == 0
    assert result["generation"]["max_tokens"] == 4096
    by_id = {r["canonical_pair_id"]: r for r in rows}
    for request in requests:
        row = by_id[request["canonical_pair_id"]]
        assert request["request_sha256"] == sha256_json(request["body"])
        assert row["fixed_main_sha256"] == sha256_json(row["raw_main"])
        assert request["input_tokens"] + 4096 + 2048 <= 32768
        messages = json.dumps(request["body"]["messages"])
        assert row["review_id"] not in messages
        assert row["canonical_pair_id"] not in messages
        assert "human_same_duplicate_group" not in messages
    assert {r["arm"] for r in requests} == {"control", "candidate"}
    assert {r["repeat"] for r in requests} == {1, 2}
    assert all(Path(p).is_file() for p in result["sources"])
    assessment = subject.assess(root)
    assert assessment["next_step"] == "STOP_BEFORE_EXPANSION"
    assert assessment["logical_responses_saved"] == 0
    for cell in assessment["cells"].values():
        assert cell["scores"]["partial_draft"]["unweighted"]["rows"] == 19
        assert cell["engineering_failures"] > 0
    assert (
        assessment["saved_baselines"]["route_only"]["unweighted"]["primary_decision_exact"]
        > assessment["saved_baselines"]["saved_final"]["unweighted"]["primary_decision_exact"]
    )
