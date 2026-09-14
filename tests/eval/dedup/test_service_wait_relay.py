# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest
import requests
from werkzeug.wrappers import Response

from eval.dedup.judging.auth_database_busy_relay import BUSY_MESSAGE
from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.judging.service_wait_relay import CONTRACT, ServiceWaitRelay


def relay(httpserver, root, *, budget=30):
    value = ServiceWaitRelay(
        profile=TransportProfile(
            min_interval_seconds=0.001,
            max_in_flight=2,
            max_attempts=2,
            retry_base_seconds=0.005,
            retry_cap_seconds=0.01,
            request_deadline_seconds=3,
            max_external_attempts=budget,
        ),
        logical_model="logical",
        upstream_model="upstream",
        upstream_base_url=httpserver.url_for("/v1"),
        upstream_api_key="test-only-secret",
        timeout_seconds=2,
        expected_generation_parameters={"temperature": 0},
    )
    value.service_cooldown_seconds = 0.01
    value.set_context(RelayContext("test", 0, root / "transport_events.jsonl"))
    return value


@pytest.mark.parametrize(
    ("statuses", "message", "expected"),
    [
        ([500, 500], "engine failed", ("WAIT", "SERVICE_UNAVAILABLE")),
        ([502, 504], "connection failed", ("WAIT", "SERVICE_UNAVAILABLE")),
        ([503, 503], "unavailable", ("WAIT", "SERVICE_UNAVAILABLE")),
        ([401, 401], BUSY_MESSAGE, ("WAIT", "SERVICE_UNAVAILABLE")),
        ([429], "limited", ("WAIT", "RATE_LIMIT")),
        ([401], "Invalid API key", ("STOP", "NONRETRYABLE_UPSTREAM")),
        ([403], "permission denied", ("STOP", "NONRETRYABLE_UPSTREAM")),
        ([400], "invalid request", ("STOP", "NONRETRYABLE_UPSTREAM")),
    ],
)
def test_control_handoff_preserves_upstream_status_and_identical_retry_body(
    httpserver, tmp_path, statuses, message, expected
):
    action, reason = expected
    calls = []

    def respond(request):
        calls.append(json.loads(request.get_data()))
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        return Response(
            json.dumps({"error": {"message": message, "type": "auth_error", "code": str(status)}}),
            status=status,
            content_type="application/json",
        )

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    body = {"model": "logical", "temperature": 0}
    with relay(httpserver, tmp_path) as proxy:
        response = requests.post(proxy.endpoint + "/chat/completions", json=body, timeout=5)
        assert response.status_code == 503
        assert response.json()["service_wait_contract"] == CONTRACT
        assert response.json()["action"] == action
        assert response.json()["reason"] == reason
        assert len(calls) == len(statuses)
        assert all(c == {**body, "model": "upstream"} for c in calls)
        assert (proxy._circuit_reason is None) is (action == "WAIT")
    events = [json.loads(s) for s in (tmp_path / "transport_events.jsonl").read_text().splitlines()]
    assert [e["upstream_http_status"] for e in events] == statuses
    assert "test-only-secret" not in json.dumps(events)


@pytest.mark.parametrize("blocked", ["circuit", "budget"])
def test_queued_admission_failure_is_control_stop_without_external_request(httpserver, tmp_path, blocked):
    with relay(httpserver, tmp_path) as proxy:
        if blocked == "circuit":
            proxy._open_circuit("nonretryable_upstream_error")
        else:
            proxy._external_attempts = proxy.profile.max_external_attempts
        response = requests.post(
            proxy.endpoint + "/chat/completions", json={"model": "logical", "temperature": 0}, timeout=5
        )
        assert response.json()["action"] == "STOP"
        assert response.json()["reason"] in {"TRANSPORT_CIRCUIT_OPEN", "TRANSPORT_ATTEMPT_BUDGET"}
    assert not httpserver.log


def test_success_response_is_not_wrapped_or_modified(httpserver, tmp_path):
    raw = {"choices": [{"finish_reason": "stop", "message": {"content": "unchanged"}}]}
    httpserver.expect_request("/v1/chat/completions").respond_with_json(raw)
    with relay(httpserver, tmp_path) as proxy:
        response = requests.post(
            proxy.endpoint + "/chat/completions", json={"model": "logical", "temperature": 0}, timeout=5
        )
    assert response.status_code == 200
    assert response.json() == raw
