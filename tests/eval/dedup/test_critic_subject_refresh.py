# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from eval.dedup.analysis import critic_subject_refresh as trial
from eval.dedup.validation import DedupEvaluationError, write_json_atomic

PARENT = trial.subject.SESSION / "subject-proof-v4-full1000"


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed full conditional trial absent")
def test_refresh_preserves_all_admitted_bodies_and_original_population(tmp_path):
    root = tmp_path / "refresh"
    manifest = trial.prepare(root, PARENT)
    trial.reference.verify_freeze(root / "manifest.json")
    origin = Path(manifest["coverage_origin"])
    assert json.loads((root / "requests_frozen.json").read_text()) == json.loads(
        (origin / "specialist/requests_frozen.json").read_text()
    )
    assert json.loads((root / "panel_private.json").read_text()) == json.loads(
        (origin / "specialist/panel_private.json").read_text()
    )
    assert manifest["population"] == 1000
    assert manifest["logical_requests"] == 129
    assert manifest["main_online_calls"] == manifest["coverage_online_calls"] == 0
    assert not (root / "responses").exists()
    assert manifest["final_action_contract"] == trial.proof.candidate.CONTRACT
    with pytest.raises(DedupEvaluationError, match="SUBJECT_REFRESH_COMPLETE"):
        trial.prepare_verifier(root)


def test_refresh_rejects_unknown_source_before_creating_artifacts(tmp_path):
    root = tmp_path / "refresh"
    with pytest.raises((DedupEvaluationError, FileNotFoundError)):
        trial.prepare(root, tmp_path / "missing_parent")
    assert not root.exists()


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed full conditional trial absent")
def test_verification_uses_all_bound_proposals_and_rejects_bad_receipts(tmp_path):
    root = tmp_path / "refresh"
    manifest = trial.prepare(root, PARENT)
    origin = Path(manifest["coverage_origin"])
    for path in (origin / "specialist/responses").glob("*.json"):
        write_json_atomic(root / "responses" / path.name, json.loads(path.read_text()))
    trial.subject.assess(root)
    write_json_atomic(root / "complete.json", {"offline_fixture": True})
    result = trial.prepare_verifier(root)
    trial.reference.verify_freeze(root / "verification/manifest.json")
    rows = json.loads((root / "verification/panel_private.json").read_text())
    requests = json.loads((root / "verification/requests_frozen.json").read_text())
    expected = {
        r["canonical_pair_id"]
        for r in rows
        if trial.proof.candidate.route(r["main_public"], r["payload"]) == "VERIFY_FIXED_SUBJECT_VETO"
    }
    assert {r["canonical_pair_id"] for r in requests} == expected
    assert len(requests) == 2 * len(expected) == 8
    assert result["fresh_subject_logical_requests"] == 129
    assert len(rows) == 1000

    bad_root = tmp_path / "bad_refresh"
    trial.prepare(bad_root, PARENT)
    receipt = json.loads(next((origin / "specialist/responses").glob("*.json")).read_text())
    receipt["status"] = "INVALID"
    write_json_atomic(bad_root / "responses/bad.json", receipt)
    write_json_atomic(bad_root / "complete.json", {"offline_fixture": True})
    with pytest.raises(DedupEvaluationError, match="SUBJECT_REFRESH_VALID"):
        trial.prepare_verifier(bad_root)
    assert not (bad_root / "verification").exists()
