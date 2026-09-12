# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import critic_subject_full_pipeline as subject
from eval.dedup.validation import write_json_atomic


@pytest.mark.skipif(
    not (subject.subject_trial.SESSION / "subject-v2-scope96/review_complete.json").exists(),
    reason="reviewed local scope trial unavailable",
)
def test_full_pipeline_prefreezes_both_coverage_arms_and_every_potential_subject_request(tmp_path):
    root = tmp_path / "full"
    result = subject.prepare(root, subject.subject_trial.SESSION / "subject-v2-scope96")
    subject.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    subject.selected.base.previous.reference.verify_freeze(root / "coverage/manifest.json")
    assert result["population"] == 1000
    assert result["coverage_logical_requests"] == 500
    assert result["main_online_calls"] == 0
    rows = json.loads((root / "panel_private.json").read_text())
    potential = json.loads((root / "potential_subject_requests.json").read_text())
    expected = {
        r["canonical_pair_id"]
        for r in rows
        if subject.subject_trial.subject.route(r["main_public"], r["payload"]) == "REVIEW_BILATERAL_SUBJECTS"
    }
    assert {r["canonical_pair_id"] for r in potential} == expected
    assert all(r["repeat"] == 1 and r["arm"] == "candidate" for r in potential)
    assert all("review_id" not in str(r["body"]) and "human_" not in str(r["body"]) for r in potential)


def test_final_metrics_propagate_upstream_engineering_failure_even_if_later_stage_bypasses(tmp_path):
    root = tmp_path / "full"
    unresolved = {
        "a_can_replace_b": "UNRESOLVED",
        "b_can_replace_a": "UNRESOLVED",
        "same_duplicate_group": "UNRESOLVED",
    }
    rows = []
    pairs = []
    for i in range(1000):
        pid = f"pair-{i}"
        label = {"canonical_pair_id": pid, **{"human_" + k: v for k, v in unresolved.items()}}
        rows.append(
            {
                "canonical_pair_id": pid,
                "review_id": str(i),
                "main_public": deepcopy(unresolved),
                "saved_final": deepcopy(unresolved),
                "historical_label": label,
                "draft_label": label,
            }
        )
        pairs.append(
            {
                "canonical_pair_id": pid,
                "review_id": str(i),
                "primary": deepcopy(unresolved),
                "draft_error": "CORRECT",
                "status": "DETERMINISTIC_BYPASS",
                "rule": "BYPASS",
                "requires_cause_review": False,
            }
        )
    cell = {"pairs": pairs, "engineering_failures": 0}
    coverage = {"cells": {"1/control": deepcopy(cell), "1/candidate": deepcopy(cell)}}
    coverage["cells"]["1/candidate"]["pairs"][0].update(status="ENGINEERING_FAILURE", requires_cause_review=True)
    limited = {"cells": {"1/candidate": deepcopy(cell)}}
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "coverage/assessment.json", coverage)
    write_json_atomic(root / "specialist/assessment.json", {})
    write_json_atomic(root / "limited_scope/assessment.json", limited)
    for path in ("coverage/complete.json", "specialist/complete.json"):
        write_json_atomic(root / path, {"external_attempts": 1, "local_rejections": 0})
    subject.freeze_manifest(root, {"sources": {}}, ("panel_private.json",))
    result = subject.final_assessment(root)
    candidate = result["cells"]["1/candidate"]
    assert candidate["engineering_failures"] == 1
    assert candidate["scores"]["partial_draft"]["missing_output_pairs"] == 1
    assert candidate["scores"]["partial_draft"]["weighted"]["primary_decision_exact"] == pytest.approx(0.999)
    assert candidate["pairs"][0]["status"] == "ENGINEERING_FAILURE"
