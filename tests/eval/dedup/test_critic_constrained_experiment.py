# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from eval.dedup.analysis import critic_constrained_experiment as subject

SESSION = subject.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z"


@pytest.mark.skipif(
    not (SESSION / "schema-projection-v1/assessment.json").exists(), reason="local compatibility unavailable"
)
def test_real_freeze_separates_schema_transport_from_semantic_messages(tmp_path):
    root = tmp_path / "constrained"
    manifest = subject.prepare(
        root,
        stage="pilot",
        parent=None,
        adapter=subject.selected.DEFAULT_ADAPTER,
        config=subject.selected.DEFAULT_CONFIG,
        protocol=Path(subject.__file__).with_name("critic_retention_v3_constrained_protocol.md"),
        control_adapter=subject.selected.DEFAULT_ADAPTER,
        control_config=subject.selected.DEFAULT_CONFIG,
        compatibility=SESSION / "schema-projection-v1",
        reviewed=SESSION / "retention-v3-pilot",
    )
    subject.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    subject.selected.base.previous.reference.verify_freeze(root / "unconstrained_freeze/manifest.json")
    requests = json.loads((root / "requests_frozen.json").read_text())
    assert len(requests) == 96
    assert manifest["population"] == 19
    for candidate in (r for r in requests if r["arm"] == "candidate"):
        control = next(
            r
            for r in requests
            if r["arm"] == "encoding_control"
            and r["repeat"] == candidate["repeat"]
            and r["canonical_pair_id"] == candidate["canonical_pair_id"]
        )
        assert {k: v for k, v in candidate["body"].items() if k != "response_format"} == {
            k: v for k, v in control["body"].items() if k != "response_format"
        }
        assert candidate["request_sha256"] != control["request_sha256"]
        assert candidate["unconstrained_request_sha256"] == control["request_sha256"]
        schema = candidate["body"]["response_format"]["json_schema"]["schema"]
        assert "uniqueItems" not in schema["properties"]["shared_anchor_ids"]
        assert schema["properties"]["a_context_span_id"]["minLength"] == 1
    result = subject.selected.assess(root)
    assert result["full_development_75_gate"] is False
    assert all(cell["engineering_failures"] == 16 for cell in result["cells"].values())
