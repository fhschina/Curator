# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
import time
from contextlib import suppress
from pathlib import Path

import pytest
from werkzeug.wrappers import Response

from eval.dedup.analysis import checkpoint_preflight as fixtures
from eval.dedup.analysis import exp6_auth_db_recovery as subject
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic
from tests.eval.dedup import test_exp1_reproduction_runtime as native
from tests.eval.dedup.test_exp6_full20k import freeze


def interrupted(root, httpserver):
    rows, main, manifest = freeze(root, two_pairs=True)
    with native.server([main]) as (endpoint, _):
        subject.full.execute(root, rows[0], endpoint, subject.full.runtime.old.coverage_renderer())
    payload = fixtures.synthetic_payload("Authentication recovery record.", "Authentication recovery record.")
    auth_raw = fixtures.synthetic_raw(payload, deltas=("none", "none"))
    auth_row = {"canonical_pair_id": "auth-pair", "review_id": "AUTH"}
    request = subject.full.runtime.old.body(subject.full.runtime.old.main_messages(payload))
    write_json_atomic(root / "inputs/auth-pair.json", {**auth_row, "payload": payload})
    write_json_atomic(root / "main_requests/auth-pair.json", {"body": request, "request_sha256": sha256_json(request)})
    (root / "panel_index.json").write_text(json.dumps([*rows, auth_row]))
    manifest.update(
        population=3,
        endpoint=httpserver.url_for("/v1"),
        artifacts={
            str(p): sha256_file(p)
            for p in [
                root / "panel_index.json",
                root / "smoke_panel.json",
                *(root / "inputs").glob("*.json"),
                *(root / "main_requests").glob("*.json"),
            ]
        },
    )
    manifest.pop("contract_digest")
    manifest["contract_digest"] = sha256_json(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    httpserver.expect_request("/v1/chat/completions").respond_with_json(
        {"error": {"code": "401", "type": "auth_error", "message": subject.transport.BUSY_MESSAGE}}, status=401
    )
    with subject.full.limits.RateLimitRelay(
        profile=subject.recovery.original.TransportProfile(
            min_interval_seconds=0.001,
            max_in_flight=1,
            max_attempts=2,
            retry_base_seconds=0.01,
            retry_cap_seconds=0.02,
            request_deadline_seconds=3,
            max_external_attempts=4,
        ),
        logical_model=subject.full.runtime.old.LOGICAL_MODEL,
        upstream_model=manifest["model"],
        upstream_base_url=manifest["endpoint"],
        upstream_api_key="test-secret",
        timeout_seconds=2,
        expected_generation_parameters=subject.full.runtime.old.GENERATION,
    ) as relay:
        relay.set_context(subject.recovery.original.RelayContext("original", 0, root / "transport_events.jsonl"))
        result = subject.full.execute(root, auth_row, relay.endpoint, subject.full.runtime.old.coverage_renderer())
    assert result["status"] == "ENGINEERING_FAILURE"
    subject.full.check_smoke(root, manifest)
    write_json_atomic(
        root / subject.PATCH_DIR / "health_probe.json",
        {"http_status": 200, "old_key_used": True, "model": manifest["model"], "endpoint": manifest["endpoint"]},
    )
    httpserver.clear()
    return main, auth_raw


def test_append_only_full_continuation_collects_only_pending_and_restores_transport(httpserver, tmp_path, monkeypatch):
    main, _ = interrupted(tmp_path, httpserver)
    original = subject.full.limits.RateLimitRelay
    before = {
        str(p): sha256_file(p)
        for folder in ("requests", "responses", "results")
        for p in (tmp_path / folder).glob("*.json")
    }
    prepared = subject.prepare(tmp_path)
    assert prepared["saved"] == 2
    assert prepared["pending"] == 1
    assert prepared["separate_recovery_case"] == "AUTH"
    httpserver.expect_request("/v1/chat/completions").respond_with_json(native.response(main))
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    subject.run(tmp_path, tmp_path / "absent.env", tmp_path / "recovery/sessions/resume")
    assert len(httpserver.log) == 1
    assert all(sha256_file(p) == h for p, h in before.items())
    assert subject.full.limits.RateLimitRelay is original
    assert subject.read(tmp_path / "complete.json")["statuses"] == {"VALID": 2, "ENGINEERING_FAILURE": 1}
    summary = subject.read(tmp_path / subject.PATCH_DIR / "complete.json")
    assert summary["additional_external_attempts"] == 1
    assert summary["original_saved_results_preserved"] == 2


def test_single_case_recovery_is_fresh_audited_and_never_merges_original_failure(httpserver, tmp_path, monkeypatch):
    root, target = tmp_path / "original", tmp_path / "repaired"
    _, raw = interrupted(root, httpserver)
    subject.prepare(root)
    old_path = root / "results/auth-pair.json"
    old_hash = sha256_file(old_path)
    seen = []

    def respond(request):
        seen.append(json.loads(request.get_data()))
        return Response(json.dumps(native.response(raw)), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    result = subject.recover_case(root, target, tmp_path / "absent.env")
    assert len(seen) == 1
    assert seen[0] == subject.read(root / "main_requests/auth-pair.json")["body"]
    assert result["status"] == "VALID"
    assert result["offline_replay_identical"]
    assert sha256_file(old_path) == old_hash
    assert subject.read(old_path)["status"] == "ENGINEERING_FAILURE"
    assert not (root / "results/second-pair.json").exists()
    with pytest.raises(DedupEvaluationError, match="AUTH_DB_FRESH_CASE"):
        subject.recover_case(root, target, tmp_path / "absent.env")
    assert len(seen) == 1


@pytest.mark.parametrize("changed", ["result", "prefix"])
def test_snapshot_or_transport_prefix_changes_block_recovery(httpserver, tmp_path, changed):
    interrupted(tmp_path, httpserver)
    subject.prepare(tmp_path)
    subject.verify(tmp_path)
    path = tmp_path / ("results/test-pair.json" if changed == "result" else "transport_events.jsonl")
    path.write_text("{}")
    with pytest.raises(subject.recovery.CheckpointIntegrityError):
        subject.verify(tmp_path)


def test_failed_health_check_prevents_freezing_and_network_calls(httpserver, tmp_path):
    interrupted(tmp_path, httpserver)
    path = tmp_path / subject.PATCH_DIR / "health_probe.json"
    health = subject.read(path)
    health["http_status"] = 401
    path.write_text(json.dumps(health))
    with pytest.raises(DedupEvaluationError, match="AUTH_DB_HEALTH"):
        subject.prepare(tmp_path)
    assert not (tmp_path / subject.PATCH_DIR / "manifest.json").exists()
    assert not httpserver.log


def test_transport_override_restores_original_on_exception():
    original = subject.full.limits.RateLimitRelay

    def interrupted_override():
        with subject.transport_override():
            assert subject.full.limits.RateLimitRelay is subject.transport.AuthDatabaseBusyRelay
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        interrupted_override()
    assert subject.full.limits.RateLimitRelay is original


def test_nohup_recovery_launch_detaches_and_missing_key_stops_safely(httpserver, tmp_path, monkeypatch):
    interrupted(tmp_path, httpserver)
    subject.prepare(tmp_path)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    launch = subject.launch(tmp_path, tmp_path / "absent.env")
    session = Path(launch["session"])
    try:
        assert os.getsid(launch["pid"]) == launch["pid"]
        deadline = time.monotonic() + 20
        while not (session / "exit.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert subject.read(session / "exit.json")["error_code"] == "EXP6_CREDENTIAL"
        assert subject.full.status(tmp_path)["running"] is False
        assert not httpserver.log
    finally:
        with suppress(ChildProcessError):
            os.waitpid(launch["pid"], 0)
