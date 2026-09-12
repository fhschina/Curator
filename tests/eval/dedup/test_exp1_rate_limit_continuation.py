# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time
from copy import deepcopy

import pytest
from werkzeug.wrappers import Response

from eval.dedup.analysis import exp1_rate_limit_continuation as subject
from eval.dedup.analysis import exp1_reproduction_runtime as runtime
from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.validation import sha256_file, write_json_atomic, write_text_atomic
from tests.eval.dedup.test_exp1_reproduction_recovery import case
from tests.eval.dedup.test_exp1_reproduction_runtime import response


def relay(httpserver, root):
    out = subject.RateLimitRelay(
        profile=TransportProfile(
            min_interval_seconds=0.001,
            max_in_flight=2,
            max_attempts=2,
            retry_base_seconds=0.001,
            retry_cap_seconds=0.002,
            request_deadline_seconds=3,
            max_external_attempts=20,
        ),
        logical_model=runtime.LOGICAL_MODEL,
        upstream_model=runtime.LOGICAL_MODEL,
        upstream_base_url=httpserver.url_for("/v1"),
        upstream_api_key="local-only-secret",
        timeout_seconds=2,
        expected_generation_parameters=runtime.GENERATION,
    )
    out.set_context(RelayContext("local-rate-test", 0, root / "transport_events.jsonl"))
    return out


def collector(root):
    out = subject.RateLimitCollector(root, root / "session")
    out.retry_base_seconds = 0.01
    out.retry_cap_seconds = 0.02
    return out


def test_repeated_429_then_success_preserves_body_native_pipeline_and_audit(httpserver, tmp_path):
    row, frozen, raw = case()
    calls = []

    def respond(request):
        calls.append((time.monotonic(), json.loads(request.get_data())))
        if len(calls) < 3:
            return Response(
                json.dumps({"error": {"message": "Token rate exceeded", "code": "tpm"}}),
                status=429,
                headers={"Retry-After": "0.03"},
                content_type="application/json",
            )
        return Response(json.dumps(response(raw)), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    collecting = collector(tmp_path)
    with relay(httpserver, tmp_path) as transport, subject.recovery.use_collector(collecting):
        result = runtime.execute_case(tmp_path, row, transport.endpoint, frozen, runtime.coverage_renderer())
    assert result["status"] == "VALID"
    assert result["main_attempts"] == [{"outer": 1, "requests": 1, "status": "VALID"}]
    assert len(calls) == 3
    assert all(body == frozen["body"] for _, body in calls)
    assert calls[1][0] - calls[0][0] >= 0.025
    assert calls[2][0] - calls[1][0] >= 0.025
    assert collecting.rate_retries == 2
    assert len(list((tmp_path / "responses").glob("*.json"))) == 1
    write_json_atomic(tmp_path / "manifest.json", {"sources": {}, "artifacts": {}})
    write_json_atomic(tmp_path / "main_requests.json", [frozen])
    proof = subject.recovery.original.audit(tmp_path, [row], [result])
    assert proof["bound_calls"] == 1
    assert proof["valid_pipeline_replays"] == 1


def test_429_wait_budget_leaves_checkpoint_pending_not_scored_failure(httpserver, tmp_path):
    row, frozen, _ = case()
    httpserver.expect_request("/v1/chat/completions").respond_with_json({"detail": "Rate limit exceeded"}, status=429)
    collecting = collector(tmp_path)
    collecting.max_rate_attempts = 2
    with (
        relay(httpserver, tmp_path) as transport,
        subject.recovery.use_collector(collecting),
        pytest.raises(subject.RateLimitWaitBudget),
    ):
        runtime.execute_case(tmp_path, row, transport.endpoint, frozen, runtime.coverage_renderer())
    assert len(httpserver.log) == 2
    assert (tmp_path / "requests/pair-main-01-01.json").exists()
    assert not (tmp_path / "responses/pair-main-01-01.json").exists()
    assert not (tmp_path / "results/pair.json").exists()


def test_retry_after_beyond_window_is_not_shortened(httpserver, tmp_path):
    row, frozen, _ = case()
    httpserver.expect_request("/v1/chat/completions").respond_with_json(
        {"detail": "Limited"}, status=429, headers={"Retry-After": "1000"}
    )
    collecting = collector(tmp_path)
    collecting.max_wait_window_seconds = 0.1
    with (
        relay(httpserver, tmp_path) as transport,
        subject.recovery.use_collector(collecting),
        pytest.raises(subject.RateLimitWaitBudget),
    ):
        runtime.execute_case(tmp_path, row, transport.endpoint, frozen, runtime.coverage_renderer())
    assert len(httpserver.log) == 1


@pytest.mark.parametrize("status", ["FAILURE", "RECEIVED"])
def test_presaved_failure_and_success_are_not_requeried_or_overwritten(tmp_path, status):
    _, frozen, _ = case()
    write_json_atomic(tmp_path / "requests/pair-main-01-01.json", {k: frozen[k] for k in ("body", "request_sha256")})
    receipt = {"status": status, "request_sha256": frozen["request_sha256"], "error_code": "EXP1_HTTP"}
    path = tmp_path / "responses/pair-main-01-01.json"
    write_json_atomic(path, receipt)
    before = sha256_file(path)
    assert collector(tmp_path)(tmp_path, "pair-main-01-01", frozen["body"], "http://127.0.0.1:1/v1") == receipt
    assert sha256_file(path) == before


def test_full_recovery_runner_and_export_with_rate_limit_continuation(httpserver, tmp_path, monkeypatch):
    import csv
    import io

    row, frozen, raw = case()
    write_json_atomic(tmp_path / "panel_private.json", [row])
    write_json_atomic(tmp_path / "main_requests.json", [frozen])
    public = runtime.common.critic.main_decision(raw, row["payload"])
    primary = subject.recovery.original.benchmark.primary(public)
    label = {
        "canonical_pair_id": "pair",
        "review_id": "R1",
        "stratum_population_n": 10,
        "stratum_sample_n": 10,
        "human_reason_code": "identity_slot",
        "comparison_scoring_eligibility": "INCLUDED",
        **{"human_" + k: v for k, v in primary.items()},
    }
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(label))
    writer.writeheader()
    writer.writerow(label)
    write_text_atomic(tmp_path / "reference_revised_1000.csv", stream.getvalue())
    write_json_atomic(tmp_path / "comparison_exclusions.json", {"excluded_canonical_pair_ids": []})
    write_json_atomic(
        tmp_path / "historical_predictions.json",
        {v: [{"canonical_pair_id": "pair", **primary}] for v in subject.recovery.original.HISTORICAL},
    )
    write_text_atomic(tmp_path / "transport_events.jsonl", "")
    write_json_atomic(tmp_path / "started.json", {})
    manifest = {
        "sources": {},
        "artifacts": {},
        "endpoint": httpserver.url_for("/v1"),
        "model": runtime.LOGICAL_MODEL,
        "max_external_attempts": 20,
    }
    manifest["contract_digest"] = subject.sha256_json(manifest)
    write_json_atomic(tmp_path / "manifest.json", manifest)
    subject.recovery.prepare(tmp_path)
    subject.prepare(tmp_path)
    before = {p: sha256_file(p) for p in tmp_path.glob("*manifest.json")}
    monkeypatch.setenv("NVIDIA_API_KEY", "local-only-secret")
    original_body = deepcopy(frozen["body"])
    httpserver.expect_request("/v1/chat/completions").respond_with_json(response(raw))
    old_collector = subject.recovery.Collector
    with subject.transport_override(tmp_path):
        subject.recovery.run(tmp_path, tmp_path / "absent.env", tmp_path / "recovery/sessions/001")
    assert subject.recovery.Collector is old_collector
    assert len(httpserver.log) == 1
    assert json.loads(httpserver.log[0][0].get_data()) == original_body
    assert all(sha256_file(p) == h for p, h in before.items())
    assert subject.recovery.read(tmp_path / "rate_limit_final.json")["observed_external_attempts"] == 1
    assert subject.recovery.read(tmp_path / "comparison.json")["offline_audit"]["valid_pipeline_replays"] == 1
    assert subject.recovery.status(tmp_path)["exit"]["status"] == "COMPLETE"
