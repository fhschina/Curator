# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from werkzeug.wrappers import Response

from eval.dedup.analysis import exp6_full20k as full
from eval.dedup.judging.service_wait_collector import ServiceWaitCollector, ServiceWaitPending
from eval.dedup.validation import sha256_file
from tests.eval.dedup import test_exp1_reproduction_runtime as native
from tests.eval.dedup.test_exp6_full20k import freeze
from tests.eval.dedup.test_service_wait_relay import relay


def collector(root):
    value = ServiceWaitCollector(root, root / "session")
    value.retry_base_seconds = 0.01
    value.retry_cap_seconds = 0.02
    return value


def pipeline_relay(httpserver, root):
    value = relay(httpserver, root)
    value.logical_model = full.runtime.old.LOGICAL_MODEL
    value.upstream_model = full.runtime.old.LOGICAL_MODEL
    value.expected_generation_parameters = full.runtime.old.GENERATION
    return value


def test_repeated_5xx_eventually_recovers_without_judge_semantic_retry(httpserver, tmp_path):
    rows, raw, _ = freeze(tmp_path)
    calls = []

    def respond(request):
        calls.append((time.monotonic(), json.loads(request.get_data())))
        if len(calls) <= 4:
            return Response(
                json.dumps({"error": {"message": "EngineCore unavailable"}}),
                status=500,
                content_type="application/json",
            )
        return Response(json.dumps(native.response(raw)), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    collecting = collector(tmp_path)
    with pipeline_relay(httpserver, tmp_path) as proxy, full.recovery.use_collector(collecting):
        result = full.execute(tmp_path, rows[0], proxy.endpoint, full.runtime.old.coverage_renderer())
    assert result["status"] == "VALID"
    assert result["main_attempts"] == [{"outer": 1, "requests": 1, "status": "VALID"}]
    assert len(calls) == 5
    assert all(body == calls[0][1] for _, body in calls)
    assert collecting.service_retries == 2
    assert calls[2][0] - calls[1][0] >= 0.009
    assert calls[4][0] - calls[3][0] >= 0.019
    assert full.replay_results(tmp_path, rows)["offline_replay_identical"]


def test_service_budget_exhaustion_leaves_failed_and_queued_pairs_pending(httpserver, tmp_path):
    rows, _, _ = freeze(tmp_path, two_pairs=True)
    httpserver.expect_request("/v1/chat/completions").respond_with_json(
        {"error": {"message": "unavailable"}}, status=500
    )
    collecting = collector(tmp_path)
    collecting.max_service_submissions = 2
    with pipeline_relay(httpserver, tmp_path) as proxy, full.recovery.use_collector(collecting):
        with pytest.raises(ServiceWaitPending, match="BUDGET_EXHAUSTED"):
            full.execute(tmp_path, rows[0], proxy.endpoint, full.runtime.old.coverage_renderer())
        with pytest.raises(ServiceWaitPending):
            full.execute(tmp_path, rows[1], proxy.endpoint, full.runtime.old.coverage_renderer())
    assert len(httpserver.log) == 4
    assert not list((tmp_path / "results").glob("*.json"))
    assert not list((tmp_path / "responses").glob("*.json"))
    assert (tmp_path / "requests/test-pair-main-01-01.json").exists()
    assert collecting.cancelled.is_set()


def test_true_auth_failure_stops_without_semantic_retry_or_queued_failure(httpserver, tmp_path):
    rows, _, _ = freeze(tmp_path, two_pairs=True)
    httpserver.expect_request("/v1/chat/completions").respond_with_json(
        {"error": {"message": "Invalid key"}}, status=401
    )
    collecting = collector(tmp_path)
    with pipeline_relay(httpserver, tmp_path) as proxy, full.recovery.use_collector(collecting):
        with pytest.raises(ServiceWaitPending, match="NONRETRYABLE_UPSTREAM"):
            full.execute(tmp_path, rows[0], proxy.endpoint, full.runtime.old.coverage_renderer())
        with pytest.raises(ServiceWaitPending):
            full.execute(tmp_path, rows[1], proxy.endpoint, full.runtime.old.coverage_renderer())
    assert len(httpserver.log) == 1
    assert not list((tmp_path / "results").glob("*.json"))


def test_interrupted_pending_stage_resumes_identical_request_and_preserves_success(httpserver, tmp_path):
    rows, raw, _ = freeze(tmp_path)
    httpserver.expect_request("/v1/chat/completions").respond_with_json(
        {"error": {"message": "unavailable"}}, status=503
    )
    first = collector(tmp_path)
    first.max_service_submissions = 1
    with (
        pipeline_relay(httpserver, tmp_path) as proxy,
        full.recovery.use_collector(first),
        pytest.raises(ServiceWaitPending),
    ):
        full.execute(tmp_path, rows[0], proxy.endpoint, full.runtime.old.coverage_renderer())
    path = tmp_path / "requests/test-pair-main-01-01.json"
    before = sha256_file(path)
    httpserver.clear()
    httpserver.expect_request("/v1/chat/completions").respond_with_json(native.response(raw))
    second = collector(tmp_path)
    with pipeline_relay(httpserver, tmp_path) as proxy, full.recovery.use_collector(second):
        result = full.execute(tmp_path, rows[0], proxy.endpoint, full.runtime.old.coverage_renderer())
        replayed = full.execute(tmp_path, rows[0], proxy.endpoint, full.runtime.old.coverage_renderer())
    assert result == replayed
    assert len(httpserver.log) == 1
    assert sha256_file(path) == before
    assert second.retransmitted == 1


def test_retry_after_exceeding_wait_budget_stops_without_shortening_it(httpserver, tmp_path):
    rows, _, _ = freeze(tmp_path)
    httpserver.expect_request("/v1/chat/completions").respond_with_json(
        {"error": {"message": "limited"}}, status=429, headers={"Retry-After": "999"}
    )
    collecting = collector(tmp_path)
    collecting.max_rate_wait_seconds = 0.1
    with (
        pipeline_relay(httpserver, tmp_path) as proxy,
        full.recovery.use_collector(collecting),
        pytest.raises(ServiceWaitPending, match="RATE_LIMIT_BUDGET"),
    ):
        full.execute(tmp_path, rows[0], proxy.endpoint, full.runtime.old.coverage_renderer())
    assert len(httpserver.log) == 1
    assert not list((tmp_path / "results").glob("*.json"))


def test_shared_cancel_wakes_another_waiter_promptly(httpserver, tmp_path):
    rows, _, _ = freeze(tmp_path)
    httpserver.expect_request("/v1/chat/completions").respond_with_json({"error": {"message": "down"}}, status=500)
    collecting = collector(tmp_path)
    collecting.retry_base_seconds = collecting.retry_cap_seconds = 60
    with (
        pipeline_relay(httpserver, tmp_path) as proxy,
        full.recovery.use_collector(collecting),
        ThreadPoolExecutor() as pool,
    ):
        future = pool.submit(full.execute, tmp_path, rows[0], proxy.endpoint, full.runtime.old.coverage_renderer())
        deadline = time.monotonic() + 3
        while not collecting.waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        assert collecting.waiting
        with pytest.raises(ServiceWaitPending):
            collecting.stop_pending("peer stopped")
        with pytest.raises(ServiceWaitPending, match="peer stopped"):
            future.result(timeout=1)
    assert not list((tmp_path / "results").glob("*.json"))
