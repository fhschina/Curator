# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from eval.dedup.analysis import critic_resilient_execution as subject

ROOT = subject.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v4-order96"


@pytest.mark.skipif(not (ROOT / "review_complete.json").exists(), reason="local reviewed aborted trial unavailable")
def test_recovery_keeps_all_frozen_requests_and_does_not_reuse_outputs(tmp_path):
    root = tmp_path / "recovery"
    result = subject.prepare(root, ROOT, Path(subject.__file__).with_name("critic_resilient_execution_v1.md"))
    subject.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    for name in ("panel_private.json", "requests_frozen.json"):
        assert (root / name).read_bytes() == (ROOT / name).read_bytes()
    assert result["transport_contract"] == "dedup-paced-transport-v2"
    assert result["old_responses_reused"] == 0
    assert not (root / "responses").exists()
    frozen = json.loads((root / "requests_frozen.json").read_text())
    assert len(frozen) == 354
    for request in frozen:
        assert subject.ordered.thaw(request)["body"] == request["body"]
