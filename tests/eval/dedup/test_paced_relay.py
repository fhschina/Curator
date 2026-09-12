# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from itertools import pairwise

import pytest
from werkzeug.wrappers import Response

from eval.dedup.judging.paced_relay import PacedRelay, TransportProfile, retry_after_seconds
from eval.dedup.judging.request_relay import RelayContext


def make_relay(httpserver, **profile):
    return PacedRelay(
        profile=TransportProfile(
            min_interval_seconds=0.01,
            retry_base_seconds=0.01,
            retry_cap_seconds=0.02,
            request_deadline_seconds=3,
            **profile,
        ),
        logical_model="logical",
        upstream_model="upstream",
        upstream_base_url=httpserver.url_for("/v1"),
        upstream_api_key="private-test-secret",
        timeout_seconds=3,
        expected_generation_parameters={"temperature": 0, "max_tokens": 16},
    )


def send(relay, *, text="private-input", **changes):
    body = {
        "model": "logical",
        "temperature": 0,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": text}],
        **changes,
    }
    request = urllib.request.Request(  # noqa: S310 - local test relay
        relay.endpoint + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_retry_after_numeric_date_and_invalid_values():
    now = datetime(2026, 9, 11, tzinfo=UTC)
    assert retry_after_seconds("2.5", now=now) == 2.5
    assert retry_after_seconds(format_datetime(now + timedelta(seconds=30)), now=now) == 30
    for invalid in (None, "bad", "-1", "nan", "inf"):
        assert retry_after_seconds(invalid, now=now) == 0


def test_429_then_success_preserves_body_and_honors_header_and_counts_both_attempts(httpserver, tmp_path):
    calls = []

    def respond(request):
        calls.append((time.monotonic(), json.loads(request.get_data()), request.headers["Authorization"]))
        return Response(
            "limited" if len(calls) == 1 else "original-response",
            status=429 if len(calls) == 1 else 200,
            headers={"Retry-After": "0.08"},
            content_type="application/json",
        )

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    path = tmp_path / "events.jsonl"
    with make_relay(httpserver) as relay:
        relay.set_context(RelayContext("main", 1, path))
        status, body = send(relay)
    assert status == 200
    assert body == b"original-response"
    assert len(calls) == 2
    assert calls[1][0] - calls[0][0] >= 0.075
    assert calls[0][1] == calls[1][1]
    assert calls[0][1]["model"] == "upstream"
    assert calls[0][2] == "Bearer private-test-secret"
    text = path.read_text()
    assert "private-test-secret" not in text
    assert "private-input" not in text
    events = list(map(json.loads, text.splitlines()))
    assert [e["http_status"] for e in events] == [429, 200]
    assert [e["upstream_attempt"] for e in events] == [1, 2]
    assert events[0]["request_hash"] == events[1]["request_hash"]


def test_actual_parallel_requests_are_spaced_and_admission_persists_across_stages(httpserver, tmp_path):
    starts = []

    def respond(_request):
        starts.append(time.monotonic())
        return Response("{}", content_type="application/json")

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    with make_relay(httpserver, max_in_flight=2) as relay:
        relay.set_context(RelayContext("main", 1, tmp_path / "main.jsonl"))
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: send(relay, text=str(i)), range(4)))
        relay.set_context(RelayContext("critic", 1, tmp_path / "critic.jsonl"))
        results.append(send(relay))
    assert all(status == 200 for status, _ in results)
    assert len(starts) == 5
    assert all(b - a >= 0.007 for a, b in pairwise(starts))


@pytest.mark.parametrize("upstream_status", [429, 503, 401])
def test_exhaustion_and_auth_errors_stop_further_upstream_requests(httpserver, tmp_path, upstream_status):
    httpserver.expect_request("/v1/chat/completions").respond_with_data("failure", status=upstream_status)
    with make_relay(httpserver, max_attempts=2) as relay:
        relay.set_context(RelayContext("main", 1, tmp_path / "events.jsonl"))
        status, _ = send(relay)
        expected = 1 if upstream_status == 401 else 2
        assert len(httpserver.log) == expected
        assert status == (401 if upstream_status == 401 else 400)
        assert send(relay)[0] == 400
        assert len(httpserver.log) == expected


def test_changed_generation_settings_never_reach_upstream(httpserver, tmp_path):
    with make_relay(httpserver) as relay:
        relay.set_context(RelayContext("main", 1, tmp_path / "events.jsonl"))
        assert send(relay, temperature=0.5)[0] == 400
    assert not httpserver.log
    events = list(map(json.loads, (tmp_path / "events.jsonl").read_text().splitlines()))
    assert all(not e["external_request"] for e in events)


def test_retry_after_beyond_deadline_is_not_shortened_into_an_early_retry(httpserver, tmp_path):
    httpserver.expect_request("/v1/chat/completions").respond_with_data(
        "limited", status=429, headers={"Retry-After": "100"}
    )
    with make_relay(httpserver) as relay:
        relay.set_context(RelayContext("main", 1, tmp_path / "events.jsonl"))
        assert send(relay)[0] == 400
    assert len(httpserver.log) == 1


def test_global_external_budget_cannot_be_bypassed_by_new_logical_requests(httpserver, tmp_path):
    httpserver.expect_request("/v1/chat/completions").respond_with_data("{}", content_type="application/json")
    with make_relay(httpserver, max_external_attempts=1) as relay:
        relay.set_context(RelayContext("main", 1, tmp_path / "events.jsonl"))
        assert send(relay)[0] == 200
        assert send(relay)[0] == 400
    assert len(httpserver.log) == 1
