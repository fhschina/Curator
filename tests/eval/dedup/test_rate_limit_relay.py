# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time

import pytest
import requests
from werkzeug.wrappers import Response

from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.rate_limit_relay import CONTRACT, RateLimitRelay, error_diagnostic
from eval.dedup.judging.request_relay import RelayContext


@pytest.mark.parametrize("statuses", [(429,), (500, 200), (500, 500), (401,), (400,)])
def test_error_handoff_and_unchanged_non429_retry_boundary(httpserver, tmp_path, statuses):
    calls = []

    def respond(request):
        calls.append((time.monotonic(), json.loads(request.get_data())))
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        return Response(
            json.dumps({"error": {"message": "token budget exceeded", "code": "tpm_limit"}}),
            status=status,
            headers={"Retry-After": "0.03", "X-RateLimit-Limit-Tokens": "10000"},
            content_type="application/json",
        )

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    path = tmp_path / "events.jsonl"
    with RateLimitRelay(
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
        upstream_api_key="local-secret",
        timeout_seconds=2,
        expected_generation_parameters={"temperature": 0},
    ) as relay:
        relay.set_context(RelayContext("test", 0, path))
        response = requests.post(
            relay.endpoint + "/chat/completions", json={"model": "logical", "temperature": 0}, timeout=5
        )
        assert len(calls) == len(statuses)
        assert response.status_code == (400 if statuses == (500, 500) else statuses[-1])
        if statuses == (429,):
            assert relay._circuit_reason is None
            assert response.json()["transport_contract"] == CONTRACT
            assert response.json()["retry_after_seconds"] == 0.03
        elif statuses[-1] != 200:
            assert relay._circuit_reason is not None
        if len(calls) > 1:
            assert calls[0][1] == calls[1][1]
            assert calls[1][0] - calls[0][0] >= 0.025
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["http_status"] for e in events] == list(statuses)
    assert events[0]["provider_diagnostic"]["error"]["code"] == "tpm_limit"
    assert "local-secret" not in path.read_text()


def test_diagnostics_redact_credentials_and_ignore_non_json_and_unknown_headers():
    raw = json.dumps(
        {
            "error": {
                "message": "Bearer private-secret nvapi-another-token private-secret",
                "code": "RATE_LIMIT",
                "credentials": "not-allowed",
            }
        }
    ).encode()
    out = error_diagnostic(
        raw,
        {
            "Authorization": "Bearer private-secret",
            "Set-Cookie": "private-cookie",
            "X-RateLimit-Remaining-Tokens": "0",
            "X-Request-ID": "id-123",
        },
        "private-secret",
    )
    encoded = json.dumps(out)
    for secret in ("private-secret", "nvapi-another-token", "private-cookie", "not-allowed"):
        assert secret not in encoded
    assert out["headers"]["x-ratelimit-remaining-tokens"] == "0"
    assert (
        error_diagnostic(b"<html>private page</html>", {}, "secret")["error"]["format"]
        == "NON_JSON_BODY_NOT_PERSISTED"
    )
