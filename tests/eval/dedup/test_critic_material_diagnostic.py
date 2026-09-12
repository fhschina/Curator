# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from eval.dedup.analysis import critic_material_diagnostic as subject

PREDECESSOR = subject.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v3-constrained-pilot"


@pytest.mark.skipif(not (PREDECESSOR / "review_complete.json").exists(), reason="local reviewed pilot unavailable")
def test_real_diagnostic_keeps_failed_gate_all_rows_and_identical_pilot_requests(tmp_path):
    root = tmp_path / "material"
    result = subject.prepare(root, PREDECESSOR, Path(subject.__file__).with_name("critic_material_diagnostic_v1.md"))
    subject.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    assert result["population"] == 96
    assert result["logical_requests"] == 236
    assert result["predecessor_gate_passed"] is False
    assert result["predecessor_repeat_disagreements"] == ["H0760"]
    rows = json.loads((root / "panel_private.json").read_text())
    assert sum(r["in_original64"] for r in rows) == 64
    assert sum(r["in_pilot"] for r in rows) == 19
    assert sum(r["in_changed39"] for r in rows) == 39
    assessment = subject.selected.assess(root)
    assert assessment["full_development_75_gate"] is False
    assert assessment["release_eligible"] is False
    for cell in assessment["cells"].values():
        assert cell["scores"]["partial_draft"]["unweighted"]["rows"] == 96
        assert cell["engineering_failures"] == 59
