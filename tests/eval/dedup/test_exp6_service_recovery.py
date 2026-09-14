# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import time
from contextlib import suppress
from pathlib import Path

import pytest

from eval.dedup.analysis import exp6_service_recovery as subject
from eval.dedup.judging.service_wait_collector import ServiceWaitPending
from eval.dedup.validation import DedupEvaluationError, sha256_file
from tests.eval.dedup import test_exp1_reproduction_runtime as native
from tests.eval.dedup.test_exp6_auth_db_recovery import interrupted


def frozen_patch(root, httpserver, monkeypatch):
    raw, recovery_raw = interrupted(root, httpserver)
    subject.previous.prepare(root)
    monkeypatch.setattr(subject, "CASE_IDS", ("AUTH",))
    prepared = subject.prepare(root)
    assert prepared["saved"] == 2
    assert prepared["pending"] == 1
    return raw, recovery_raw


def test_real_continuation_keeps_prior_results_and_only_collects_pending(httpserver, tmp_path, monkeypatch):
    raw, _ = frozen_patch(tmp_path, httpserver, monkeypatch)
    old_classes = subject.full.limits.RateLimitRelay, subject.full.limits.RateLimitCollector
    before = {
        str(p): sha256_file(p)
        for folder in ("requests", "responses", "results")
        for p in (tmp_path / folder).glob("*.json")
    }
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    httpserver.expect_request("/v1/chat/completions").respond_with_json(native.response(raw))
    subject.run(tmp_path, tmp_path / "absent.env", tmp_path / "recovery/sessions/continue")
    assert len(httpserver.log) == 1
    assert all(sha256_file(p) == h for p, h in before.items())
    assert subject.read(tmp_path / "complete.json")["statuses"] == {"VALID": 2, "ENGINEERING_FAILURE": 1}
    assert (subject.full.limits.RateLimitRelay, subject.full.limits.RateLimitCollector) == old_classes
    assert subject.read(tmp_path / subject.PATCH_DIR / "complete.json")["additional_external_attempts"] == 1


def test_hard_auth_stop_in_real_runner_leaves_pair_pending_not_failed(httpserver, tmp_path, monkeypatch):
    frozen_patch(tmp_path, httpserver, monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    httpserver.expect_request("/v1/chat/completions").respond_with_json(
        {"error": {"message": "Invalid key"}}, status=401
    )
    session = tmp_path / "recovery/sessions/stopped"
    with pytest.raises(ServiceWaitPending, match="NONRETRYABLE_UPSTREAM"):
        subject.run(tmp_path, tmp_path / "absent.env", session)
    assert not (tmp_path / "results/second-pair.json").exists()
    assert not (tmp_path / "responses/second-pair-main-01-01.json").exists()
    assert subject.read(session / "exit.json")["status"] == "STOPPED"
    assert len(httpserver.log) == 1


def test_authorized_isolated_retry_remains_separate_and_auditable(httpserver, tmp_path, monkeypatch):
    root, target = tmp_path / "original", tmp_path / "isolated"
    _, raw = frozen_patch(root, httpserver, monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    old_hash = sha256_file(root / "results/auth-pair.json")
    httpserver.expect_request("/v1/chat/completions").respond_with_json(native.response(raw))
    result = subject.recover_case(root, target, tmp_path / "absent.env", "AUTH")
    assert result["status"] == "VALID"
    assert result["offline_replay_identical"]
    assert sha256_file(root / "results/auth-pair.json") == old_hash
    assert len(httpserver.log) == 1
    with pytest.raises(DedupEvaluationError, match="SERVICE_FRESH_CASE"):
        subject.recover_case(root, target, tmp_path / "absent.env", "AUTH")
    with pytest.raises(DedupEvaluationError, match="SERVICE_CASE"):
        subject.recover_case(root, tmp_path / "wrong", tmp_path / "absent.env", "not-authorized")
    assert not (tmp_path / "wrong").exists()


def test_modified_old_results_prevent_launch(httpserver, tmp_path, monkeypatch):
    frozen_patch(tmp_path, httpserver, monkeypatch)
    (tmp_path / "results/test-pair.json").write_text("{}")
    with pytest.raises(subject.recovery.CheckpointIntegrityError):
        subject.launch(tmp_path, tmp_path / "absent.env")


def test_nohup_wrapper_uses_new_entry_and_stops_safely_without_key(httpserver, tmp_path, monkeypatch):
    frozen_patch(tmp_path, httpserver, monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    launched = subject.launch(tmp_path, tmp_path / "absent.env")
    session = Path(launched["session"])
    try:
        assert os.getsid(launched["pid"]) == launched["pid"]
        deadline = time.monotonic() + 20
        while not (session / "exit.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert subject.read(session / "exit.json")["error_code"] == "EXP6_CREDENTIAL"
        assert not httpserver.log
    finally:
        with suppress(ChildProcessError):
            os.waitpid(launched["pid"], 0)
