# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time

import pytest
import requests
from werkzeug.wrappers import Response

from eval.dedup.judging.auth_database_busy_relay import (
    BASE_CONTRACT,
    BUSY_MESSAGE,
    CONTRACT,
    AuthDatabaseBusyRelay,
    is_database_busy_auth,
)
from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.request_relay import RelayContext


def busy_body():
    return {"error": {"type": "auth_error", "code": "401", "message": BUSY_MESSAGE}}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, busy_body(), True),
        (403, busy_body(), False),
        (200, busy_body(), False),
        (401, {"error": {"type": "auth_error", "code": "401", "message": "Invalid API key"}}, False),
        (401, {"error": {"type": "auth_error", "code": "401", "message": "too many clients already"}}, False),
        (401, {"error": {"type": "auth_error", "code": "401", "message": BUSY_MESSAGE + "; revoked key"}}, False),
        (401, {"message": BUSY_MESSAGE}, False),
        (401, [busy_body()], False),
        (401, {"error": "invalid key"}, False),
        (401, {"error": {"type": "invalid_key", "code": "401", "message": BUSY_MESSAGE}}, False),
    ],
)
def test_only_exact_structured_upstream_database_error_is_retryable(status, body, expected):
    assert is_database_busy_auth(status, json.dumps(body).encode()) is expected
    assert is_database_busy_auth(status, b"not JSON") is False


@pytest.mark.parametrize(
    ("statuses", "busy"),
    [
        ([401, 200], True),
        ([401, 401], True),
        ([401], False),
        ([403], True),
        ([500, 200], True),
        ([429], True),
        ([200], True),
    ],
)
def test_real_http_retry_boundary_preserves_requests_and_actual_statuses(httpserver, tmp_path, statuses, busy):
    calls = []

    def respond(request):
        calls.append((time.monotonic(), json.loads(request.get_data())))
        value = busy_body() if busy else {"error": {"type": "auth_error", "code": "401", "message": "Invalid API key"}}
        return Response(
            json.dumps(value),
            status=statuses[min(len(calls) - 1, len(statuses) - 1)],
            headers={"Retry-After": "0.03"},
            content_type="application/json",
        )

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    path = tmp_path / "events.jsonl"
    body = {"model": "logical", "temperature": 0, "messages": [{"role": "user", "content": "unchanged"}]}
    with AuthDatabaseBusyRelay(
        profile=TransportProfile(
            min_interval_seconds=0.001,
            max_in_flight=2,
            max_attempts=2,
            retry_base_seconds=0.01,
            retry_cap_seconds=0.02,
            request_deadline_seconds=3,
            max_external_attempts=10,
        ),
        logical_model="logical",
        upstream_model="upstream",
        upstream_base_url=httpserver.url_for("/v1"),
        upstream_api_key="test-only-secret",
        timeout_seconds=2,
        expected_generation_parameters={"temperature": 0},
    ) as relay:
        relay.set_context(RelayContext("test", 0, path))
        response = requests.post(relay.endpoint + "/chat/completions", json=body, timeout=5)
        assert len(calls) == len(statuses)
        assert response.status_code == (400 if statuses == [401, 401] else statuses[-1])
        assert all(value == {**body, "model": "upstream"} for _, value in calls)
        if len(calls) == 2:
            assert calls[1][0] - calls[0][0] >= 0.025
        if statuses == [429]:
            assert response.json()["transport_contract"] == BASE_CONTRACT
            assert relay._circuit_reason is None
        elif statuses[-1] >= 400:
            assert relay._circuit_reason is not None
            requests.post(relay.endpoint + "/chat/completions", json=body, timeout=5)
            assert len(calls) == len(statuses)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    external = [e for e in events if e["external_request"]]
    assert [e["upstream_http_status"] for e in external] == statuses
    assert all(e["transport_patch_contract"] == CONTRACT for e in external)
    assert external[0]["auth_database_busy"] is (statuses[0] == 401 and busy)
    assert "test-only-secret" not in path.read_text()
