# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
import time
from contextlib import suppress
from pathlib import Path

import pytest

from eval.dedup.analysis import checkpoint_preflight as fixtures
from eval.dedup.analysis import exp6_full20k as subject
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic
from tests.eval.dedup import test_exp1_reproduction_runtime as native
from tests.eval.dedup.test_exp1_conflict_evidence_fix import conflict_fixture
from tests.eval.dedup.test_main_repetition_recovery import repeated_response


@pytest.fixture(autouse=True)
def envelopes(monkeypatch):
    original = native.response
    monkeypatch.setattr(
        native, "response", lambda value: value if isinstance(value, dict) and "choices" in value else original(value)
    )


def freeze(root, *, two_pairs=False):
    payload = fixtures.synthetic_payload("Same retained text.", "Same retained text.")
    raw = fixtures.synthetic_raw(payload, deltas=("none", "none"))
    request = subject.runtime.old.body(subject.runtime.old.main_messages(payload))
    rows = [{"canonical_pair_id": "test-pair", "review_id": "T1"}]
    if two_pairs:
        rows.append({"canonical_pair_id": "second-pair", "review_id": "T2"})
    for row in rows:
        key = row["canonical_pair_id"]
        write_json_atomic(root / "inputs" / (key + ".json"), {**row, "payload": payload})
        write_json_atomic(
            root / "main_requests" / (key + ".json"), {"body": request, "request_sha256": sha256_json(request)}
        )
    write_json_atomic(root / "panel_index.json", rows)
    write_json_atomic(root / "smoke_panel.json", rows[:1])
    manifest = {
        "version": subject.VERSION,
        "runtime_version": subject.VERSION,
        "generation": subject.runtime.old.GENERATION,
        "old_answers_reused": False,
        "population": len(rows),
        "smoke_size": 1,
        "required_smoke_stages": ["main"],
        "sources": {},
        "artifacts": {str(p): sha256_file(p) for p in root.rglob("*.json")},
        "max_external_attempts": 20,
        "workers": 1,
        "min_interval_seconds": 0.001,
        "model": subject.runtime.old.LOGICAL_MODEL,
        "endpoint": "http://unused",
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return rows, raw, manifest


def set_endpoint(root, manifest, endpoint):
    manifest["endpoint"] = endpoint
    manifest.pop("contract_digest")
    manifest["contract_digest"] = sha256_json(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_latest_runtime_repetition_repair_and_read_only_full_audit(tmp_path):
    rows, raw, _ = freeze(tmp_path)
    with native.server([repeated_response(), raw]) as (endpoint, requests):
        result = subject.execute(tmp_path, rows[0], endpoint, subject.runtime.old.coverage_renderer())
    assert len(requests) == 2
    assert requests[1] == subject.runtime.repetition.repair_request(requests[0])
    assert result["version"] == subject.VERSION
    assert result["main_repetition_repair"]["attempts"] == 1
    path = tmp_path / "results/test-pair.json"
    before = (sha256_file(path), path.stat().st_mtime_ns)
    report = subject.audit(tmp_path)
    assert report["saved_calls_replayed"] == 2
    assert report["main_retry_pairs"] == 1
    assert report["statuses"] == {"VALID": 1}
    assert subject.audit(tmp_path) == report
    assert (sha256_file(path), path.stat().st_mtime_ns) == before


def test_anchor_tolerance_is_wired_into_entry_point(tmp_path):
    rows, _, _ = freeze(tmp_path)
    payload, raw, _, proof = conflict_fixture()
    proof["shared_anchor_ids"] = ["S1"]
    request = subject.runtime.old.body(subject.runtime.old.main_messages(payload))
    (tmp_path / "inputs/test-pair.json").write_text(json.dumps({**rows[0], "payload": payload}))
    (tmp_path / "main_requests/test-pair.json").write_text(
        json.dumps({"body": request, "request_sha256": sha256_json(request)})
    )
    with native.server([raw, json.dumps(proof)]) as (endpoint, requests):
        result = subject.execute(tmp_path, rows[0], endpoint, subject.runtime.old.coverage_renderer())
    assert len(requests) == 2
    assert result["status"] == "VALID"
    assert result["coverage_anchor_id_repair"]["additional_model_calls"] == 0


@pytest.mark.parametrize("failure", ["result", "receipt", "input", "missing"])
def test_audit_rejects_tampering_without_rewriting_results(tmp_path, failure):
    rows, raw, _ = freeze(tmp_path)
    with native.server([raw]) as (endpoint, _):
        subject.execute(tmp_path, rows[0], endpoint, subject.runtime.old.coverage_renderer())
    path = tmp_path / "results/test-pair.json"
    if failure == "result":
        result = subject.read(path)
        result["public"]["same_duplicate_group"] = "NO"
        path.write_text(json.dumps(result))
    elif failure == "receipt":
        (tmp_path / "responses/test-pair-main-01-01.json").write_text("{}")
    elif failure == "input":
        (tmp_path / "inputs/test-pair.json").write_text("{}")
    else:
        path.rename(tmp_path / "missing-result.json")
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(subject.recovery.CheckpointIntegrityError):
        subject.audit(tmp_path)
    assert (path.read_bytes() if path.exists() else None) == before
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize("valid", [True, False])
def test_real_transport_smoke_gate_controls_full_continuation(tmp_path, monkeypatch, valid):
    _, raw, manifest = freeze(tmp_path, two_pairs=True)
    monkeypatch.setenv("NVIDIA_API_KEY", "local-test-credential")
    answers = [raw, raw] if valid else [repeated_response(), repeated_response()]
    with native.server(answers) as (endpoint, requests):
        set_endpoint(tmp_path, manifest, endpoint)
        session = tmp_path / "recovery/sessions/test"
        if valid:
            subject.run(tmp_path, tmp_path / "absent.env", session)
            assert (tmp_path / "results/second-pair.json").exists()
            assert subject.read(tmp_path / "complete.json")["population"] == 2
        else:
            with pytest.raises(DedupEvaluationError, match="EXP6_SMOKE_FAILED"):
                subject.run(tmp_path, tmp_path / "absent.env", session)
            assert not (tmp_path / "results/second-pair.json").exists()
            assert subject.read(session / "exit.json")["status"] == "STOPPED"
        assert len(requests) == 2
    assert subject.read(tmp_path / "smoke_complete.json")["passed"] is valid


def test_missing_critic_stage_blocks_gate(tmp_path):
    rows, raw, manifest = freeze(tmp_path)
    with native.server([raw]) as (endpoint, _):
        subject.execute(tmp_path, rows[0], endpoint, subject.runtime.old.coverage_renderer())
    manifest["required_smoke_stages"] = ["main", "subject"]
    with pytest.raises(DedupEvaluationError, match="EXP6_SMOKE_FAILED"):
        subject.check_smoke(tmp_path, manifest)
    assert subject.read(tmp_path / "smoke_complete.json")["missing_required_stages"] == ["subject"]


def test_smoke_only_then_resume_never_recalls_completed_pairs(tmp_path, monkeypatch):
    _, raw, manifest = freeze(tmp_path, two_pairs=True)
    monkeypatch.setenv("NVIDIA_API_KEY", "local-test-credential")
    with native.server([raw, raw]) as (endpoint, requests):
        set_endpoint(tmp_path, manifest, endpoint)
        subject.run(tmp_path, tmp_path / "absent.env", tmp_path / "recovery/sessions/first", smoke_only=True)
        assert len(requests) == 1
        assert not (tmp_path / "complete.json").exists()
        before = sha256_file(tmp_path / "results/test-pair.json")
        subject.run(tmp_path, tmp_path / "absent.env", tmp_path / "recovery/sessions/second")
        assert len(requests) == 2
    assert sha256_file(tmp_path / "results/test-pair.json") == before


def test_nohup_launch_really_detaches_and_logs_failure_without_exposing_key(tmp_path, monkeypatch):
    freeze(tmp_path)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    launched = subject.launch(tmp_path, tmp_path / "absent.env", smoke_only=True)
    pid = launched["pid"]
    session = Path(launched["session"])
    try:
        assert os.getsid(pid) == pid
        assert Path(f"/proc/{pid}/fd/0").resolve() == Path("/dev/null")
        deadline = time.monotonic() + 20
        while not (session / "exit.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert subject.read(session / "exit.json")["error_code"] == "EXP6_CREDENTIAL"
        assert subject.status(tmp_path)["running"] is False
        assert subject.status(tmp_path)["population"] == 1
        assert "NVIDIA_API_KEY" not in Path(launched["log"]).read_text()
    finally:
        with suppress(ChildProcessError):
            os.waitpid(pid, 0)


def test_live_launch_record_blocks_duplicate_start_before_started_file(tmp_path):
    freeze(tmp_path)
    session = tmp_path / "recovery/sessions/test"
    write_json_atomic(
        session / "launch.json", {"pid": os.getpid(), "process_start": subject.recovery.process_start(os.getpid())}
    )
    with pytest.raises(DedupEvaluationError, match="EXP6_LAUNCH"):
        subject.launch(tmp_path, tmp_path / "absent.env")


def test_prepare_copies_only_bound_inputs_and_freezes_latest_code(tmp_path, monkeypatch):
    source, checkpoint, root = (tmp_path / name for name in ("source", "checkpoint", "new"))
    rows, raw, original = freeze(source, two_pairs=True)
    original["development_payload_matches"] = 1
    original.pop("contract_digest")
    original["contract_digest"] = sha256_json(original)
    (source / "manifest.json").write_text(json.dumps(original))
    with native.server([raw, raw]) as (endpoint, _):
        for row in rows:
            subject.execute(source, row, endpoint, subject.runtime.old.coverage_renderer())
    write_json_atomic(
        source / "complete.json", {"artifacts": {str(p): sha256_file(p) for p in (source / "results").glob("*.json")}}
    )
    checkpoint_manifest = {
        "runtime_version": subject.VERSION,
        "sources": {},
        "artifacts": {},
        "offline_control_ids": [rows[1]["canonical_pair_id"]],
    }
    checkpoint_manifest["contract_digest"] = sha256_json(checkpoint_manifest)
    write_json_atomic(checkpoint / "manifest.json", checkpoint_manifest)
    write_json_atomic(checkpoint / "completion.json", {})
    for name, value in (
        ("SOURCE", source),
        ("CHECKPOINT", checkpoint),
        ("POPULATION", 2),
        ("SMOKE_SIZE", 2),
        ("KNOWN_CASES", ("T1",)),
    ):
        monkeypatch.setattr(subject, name, value)
    report = subject.prepare(root)
    assert report["runtime_version"] == subject.runtime.VERSION
    assert report["old_answers_reused"] is False
    assert not (root / "responses").exists()
    assert not (root / "results").exists()
    for folder in ("inputs", "main_requests"):
        for path in (source / folder).glob("*.json"):
            assert path.read_bytes() == (root / folder / path.name).read_bytes()
    subject.verify(root)
    with pytest.raises(DedupEvaluationError, match="EXP6_FRESH_ROOT"):
        subject.prepare(root)
