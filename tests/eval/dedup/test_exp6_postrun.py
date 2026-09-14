# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import exp6_postrun as subject
from eval.dedup.analysis import exp6_repetition_fix2 as baseline
from eval.dedup.validation import sha256_file, sha256_json, write_json_atomic
from tests.eval.dedup import test_exp1_reproduction_runtime as native
from tests.eval.dedup.test_coverage_format_recovery import response
from tests.eval.dedup.test_exp1_conflict_evidence_fix import conflict_fixture, run_case


@pytest.fixture(autouse=True)
def response_envelopes(monkeypatch):
    original = native.response
    monkeypatch.setattr(
        native, "response", lambda value: value if isinstance(value, dict) and "choices" in value else original(value)
    )


def saved_source(root, answers):
    payload, _, _, _ = conflict_fixture()
    result, calls = run_case(baseline, root, payload, answers)
    write_json_atomic(
        root / "main_requests/test-pair.json", {"body": calls[0], "request_sha256": sha256_json(calls[0])}
    )
    return result, {"canonical_pair_id": "test-pair", "review_id": "P16717", "payload": payload}


def test_offline_unknown_stage_does_not_call_model_or_fabricate_failure(tmp_path):
    _, main, _, _ = conflict_fixture()
    source = tmp_path / "source"
    result, row = saved_source(source, [main, response()])
    before = sha256_file(source / "results/test-pair.json")
    with pytest.raises(subject.MissingStage) as failure:
        subject.replay_case(source, tmp_path / "replay", row)
    assert failure.value.key == "test-pair-coverage-format-repair"
    assert not (tmp_path / "replay/results/test-pair.json").exists()
    assert sha256_file(source / "results/test-pair.json") == before
    assert result["status"] == "ENGINEERING_FAILURE"


def test_hybrid_keeps_original_receipts_and_replays_one_new_answer_exactly(tmp_path):
    _, main, _, proof = conflict_fixture()
    source, target = tmp_path / "source", tmp_path / "target"
    _, row = saved_source(source, [main, response()])
    before = {str(p): sha256_file(p) for p in source.rglob("*.json")}
    collector = subject.ServiceWaitCollector(target, target / "session")
    with native.server([json.dumps(proof)]) as (endpoint, calls):
        report = subject.execute_hybrid(source, target, row, endpoint, collector)
    assert len(calls) == 1
    assert report["status"] == "VALID"
    assert report["offline_replay_identical"]
    assert report["new_stages"] == ["test-pair-coverage-format-repair"]
    assert report["reused_stages"] == ["test-pair-main-01-01", "test-pair-coverage"]
    assert all(sha256_file(p) == digest for p, digest in before.items())


def test_changed_replay_request_is_rejected(tmp_path):
    _, main, _, proof = conflict_fixture()
    source = tmp_path / "source"
    saved_source(source, [main, json.dumps(proof)])
    body = subject.read(source / "requests/test-pair-main-01-01.json")["body"]
    with pytest.raises(subject.recovery.CheckpointIntegrityError):
        subject.saved_receipt(source, "test-pair-main-01-01", {**body, "temperature": 1})
