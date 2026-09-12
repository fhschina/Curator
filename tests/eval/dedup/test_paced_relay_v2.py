# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time

import pytest
import requests
from werkzeug.wrappers import Response

from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.paced_relay_v2 import TRANSPORT_CONTRACT, PacedRelayV2
from eval.dedup.judging.request_relay import RelayContext


@pytest.mark.parametrize("statuses", [(500, 200), (500, 500), (400,), (401,), (429, 200), (503, 200)])
def test_real_bounded_transport_recovery_preserves_order_budget_and_circuit(httpserver, tmp_path, statuses):
    calls = []

    def respond(request):
        calls.append((time.monotonic(), json.loads(request.get_data())))
        return Response("{}", status=statuses[min(len(calls) - 1, len(statuses) - 1)], content_type="application/json")

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    path = tmp_path / "events.jsonl"
    with PacedRelayV2(
        profile=TransportProfile(
            min_interval_seconds=0.01,
            max_in_flight=1,
            max_attempts=2,
            retry_base_seconds=0.04,
            retry_cap_seconds=0.05,
            request_deadline_seconds=3,
            max_external_attempts=2,
        ),
        logical_model="logical",
        upstream_model="upstream",
        upstream_base_url=httpserver.url_for("/v1"),
        upstream_api_key="local-secret",
        timeout_seconds=3,
        expected_generation_parameters={"temperature": 0},
    ) as relay:
        relay.set_context(RelayContext("test-v2", 0, path))
        body = {
            "model": "logical",
            "temperature": 0,
            "schema": {"properties": {"explanation": {}, "a_loss_span_id": {}}},
        }
        response = requests.post(relay.endpoint + "/chat/completions", json=body, timeout=5)
        assert response.status_code == (200 if statuses[-1] == 200 else statuses[0] if len(statuses) == 1 else 400)
        assert len(calls) == len(statuses)
        if len(statuses) == 2:
            assert calls[1][0] - calls[0][0] >= 0.035
            assert calls[0][1] == calls[1][1]
        assert list(calls[0][1]["schema"]["properties"]) == ["explanation", "a_loss_span_id"]
        assert requests.post(relay.endpoint + "/chat/completions", json=body, timeout=5).status_code == 400
        assert len(calls) == len(statuses)
    text = path.read_text()
    assert "local-secret" not in text
    events = [e for e in map(json.loads, text.splitlines()) if e.get("external_request")]
    assert [e["http_status"] for e in events] == list(statuses)
    assert all(e["transport_contract"] == TRANSPORT_CONTRACT for e in events)
    assert [e["upstream_attempt"] for e in events] == list(range(1, len(statuses) + 1))
