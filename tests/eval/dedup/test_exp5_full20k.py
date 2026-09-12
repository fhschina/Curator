# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import exp5_full20k as subject
from eval.dedup.validation import sha256_file, sha256_json, write_json_atomic
from tests.eval.dedup.test_exp1_conflict_evidence_fix import conflict_fixture
from tests.eval.dedup.test_exp1_reproduction_runtime import server


def test_upgrade_preserves_visible_text_and_existing_span_contract():
    payload, _, _, _ = conflict_fixture()
    expected = {**payload, "payload_schema_version": "judge-visible-payload-v3"}
    source = {k: v for k, v in expected.items() if k != "semantic_diff_evidence"}
    source["payload_schema_version"] = "judge-visible-payload-v2"
    before = deepcopy(source)
    assert subject.upgrade_payload(source) == expected
    assert source == before


def frozen_fixture(root):
    payload, main, bad, repaired = conflict_fixture()
    payload["payload_schema_version"] = "judge-visible-payload-v3"
    row = {"canonical_pair_id": "test-pair", "review_id": "T1"}
    request = subject.runtime.old.body(subject.runtime.old.main_messages(payload))
    write_json_atomic(root / "inputs/test-pair.json", {**row, "payload": payload})
    write_json_atomic(root / "main_requests/test-pair.json", {"body": request, "request_sha256": sha256_json(request)})
    write_json_atomic(root / "panel_index.json", [row])
    artifacts = {str(p): sha256_file(p) for p in root.rglob("*.json")}
    manifest = {"population": 1, "sources": {}, "artifacts": artifacts}
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return row, [main, json.dumps(bad), json.dumps(repaired)]


def test_execute_uses_exact_exp5_requests_and_audit_replays_evidence_repair(tmp_path):
    row, answers = frozen_fixture(tmp_path)
    with server(answers) as (endpoint, requests):
        result = subject.execute(tmp_path, row, endpoint, subject.runtime.old.coverage_renderer())
    assert len(requests) == 3
    assert "response_format" not in requests[0]
    assert result["version"] == subject.runtime.VERSION
    assert result["coverage_evidence_repair"]["attempts"] == 1
    before = sha256_file(tmp_path / "results/test-pair.json")
    report = subject.audit(tmp_path)
    assert report["statuses"] == {"VALID": 1}
    assert report["evidence_repair_pairs"] == 1
    assert report["saved_calls_replayed"] == 3
    assert report["release_eligible"] is False
    assert sha256_file(tmp_path / "results/test-pair.json") == before
    assert subject.audit(tmp_path) == report


def test_changed_frozen_input_is_rejected(tmp_path):
    frozen_fixture(tmp_path)
    subject.verify(tmp_path)
    (tmp_path / "inputs/test-pair.json").write_text("{}")
    with pytest.raises(subject.recovery.CheckpointIntegrityError):
        subject.verify(tmp_path)


def test_missing_results_cannot_complete(tmp_path):
    frozen_fixture(tmp_path)
    with pytest.raises(subject.recovery.CheckpointIntegrityError):
        subject.audit(tmp_path)
    assert not (tmp_path / "complete.json").exists()


def test_same_run_resume_reuses_exact_receipts_without_new_requests(tmp_path):
    row, answers = frozen_fixture(tmp_path)
    collector = subject.limits.RateLimitCollector(tmp_path, tmp_path / "session")
    with server(answers) as (endpoint, requests), subject.recovery.use_collector(collector):
        first = subject.execute(tmp_path, row, endpoint, subject.runtime.old.coverage_renderer())
        second = subject.execute(tmp_path, row, endpoint, subject.runtime.old.coverage_renderer())
    assert first == second
    assert len(requests) == 3
    assert collector.replayed == 3
