# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_schema_confirmation as subject


@pytest.mark.skipif(
    not (
        subject.probe.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v3-pilot/manifest.json"
    ).exists(),
    reason="local frozen v3 requests unavailable",
)
def test_actual_contract_probes_change_only_response_format():
    session = subject.probe.base.PRIOR.parent / "critic-five-hour-20260911T070052Z"
    requests, rows = subject.planned_requests(session)
    old = json.loads((session / "retention-v3-pilot/requests_frozen.json").read_text())
    assert len(requests) == 6
    assert set(rows) == {"H0701", "H0893", "H0850"}
    for request in requests:
        assert request["request_sha256"] == subject.sha256_json(request["body"])
        if request["mode"] != "actual_critic_schema":
            continue
        original = next(r for r in old if r["request_sha256"] == request["original_request_sha256"])
        assert {k: v for k, v in original["body"].items() if k != "response_format"} == {
            k: v for k, v in request["body"].items() if k != "response_format"
        }
        assert request["body"]["response_format"]["json_schema"]["schema"] == subject.critic.response_schema()
