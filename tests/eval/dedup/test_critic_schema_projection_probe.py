# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from eval.dedup.analysis import critic_schema_projection_probe as subject


@pytest.mark.skipif(
    not (subject.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v3-pilot/manifest.json").exists(),
    reason="local frozen input unavailable",
)
def test_requests_change_only_server_schema():
    session = subject.base.PRIOR.parent / "critic-five-hour-20260911T070052Z"
    original, rows = subject.prior.planned_requests(session)
    projected, projected_rows = subject.planned_requests(session)
    assert projected_rows == rows
    assert len(projected) == 6
    for request in projected:
        matching = next(r for r in original if r["id"] == request["id"])
        assert {k: v for k, v in request["body"].items() if k != "response_format"} == {
            k: v for k, v in matching["body"].items() if k != "response_format"
        }
        assert request["request_sha256"] == subject.sha256_json(request["body"])
