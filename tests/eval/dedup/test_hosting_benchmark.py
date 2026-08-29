# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from eval.dedup.hosting_benchmark.artifacts import complete_marker, next_attempt_root
from eval.dedup.hosting_benchmark.config import load_config
from eval.dedup.hosting_benchmark.metrics import _endpoint_concurrency_summary, percentile
from eval.dedup.hosting_benchmark.relay import (
    LoopbackThreadingHTTPServer,
    RelayContext,
    RelayTarget,
    RequestRelay,
    audit_paired_request_events,
    canonical_request_hash,
    select_successful_initial_request_events,
)
from eval.dedup.hosting_benchmark.runner import _persist_dynamic_preflight_report, _persist_idempotent_report
from eval.dedup.hosting_benchmark.workload import allocate_blocks, provision_model_checkpoint
from eval.dedup.validation import DedupEvaluationError


def _workload_rows() -> list[dict[str, Any]]:
    rows = []
    for track in ("5a", "5b"):
        for index in range(3_100):
            rows.append(
                {
                    "canonical_pair_id": f"{track}-{index:04d}",
                    "track": track,
                    "stratification_tokens": index,
                }
            )
    return rows


def test_workload_allocation_is_balanced_disjoint_and_deterministic() -> None:
    first = allocate_blocks(
        _workload_rows(),
        seed=26082701,
        concurrencies=(1, 2, 4, 8),
        repeats=3,
        pairs_per_block=500,
        warmup=100,
    )
    second = allocate_blocks(
        _workload_rows(),
        seed=26082701,
        concurrencies=(1, 2, 4, 8),
        repeats=3,
        pairs_per_block=500,
        warmup=100,
    )

    assert first == second
    warmup, blocks = first
    assert len(warmup) == 100
    assert len(blocks) == 12
    assert Counter(block["endpoint_order"][0] for block in blocks) == {"local": 6, "hub": 6}
    measured_ids = []
    for block in blocks:
        assert len(block["rows"]) == 500
        assert Counter(row["track"] for row in block["rows"]) == {"5a": 250, "5b": 250}
        assert Counter(row["stratum"] for row in block["rows"]) == {
            f"{track}:q{quintile}": 50 for track in ("5a", "5b") for quintile in range(1, 6)
        }
        measured_ids.extend(row["canonical_pair_id"] for row in block["rows"])
    assert len(set(measured_ids)) == 6_000
    assert not set(measured_ids) & {row["canonical_pair_id"] for row in warmup}


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([], 0.95) is None
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_frozen_config_has_no_rpm_limiter() -> None:
    config_path = Path(__file__).parents[3] / "eval" / "dedup" / "hosting_benchmark" / "qwen38_fp8.json"
    config = load_config(config_path)

    assert config.workload.concurrencies == (1, 2, 4, 8)
    assert config.timeout_seconds == 600
    assert "requests_per_minute" not in config_path.read_text(encoding="utf-8")


def test_incomplete_block_uses_a_new_attempt_directory(tmp_path: Path) -> None:
    first = next_attempt_root(tmp_path)
    (first / "partial.jsonl").write_text("partial\n", encoding="utf-8")

    assert complete_marker(tmp_path) is None
    second = next_attempt_root(tmp_path)
    (second / "complete.json").write_text("{}\n", encoding="utf-8")

    assert first.name == "attempt-01"
    assert second.name == "attempt-02"
    assert complete_marker(tmp_path) == second / "complete.json"


def test_quota_limited_attempt_is_not_a_complete_marker(tmp_path: Path) -> None:
    first = next_attempt_root(tmp_path)
    (first / "quota_limited.json").write_text('{"status":"quota_limited"}\n', encoding="utf-8")

    assert complete_marker(tmp_path) is None
    second = next_attempt_root(tmp_path)
    (second / "complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")

    assert complete_marker(tmp_path) == second / "complete.json"


def test_idempotent_report_ignores_only_declared_volatile_fields(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    first = {"checked_at_utc": "first", "storage_free_bytes": 100, "status": "pass"}
    second = {"checked_at_utc": "second", "storage_free_bytes": 90, "status": "pass"}

    assert (
        _persist_idempotent_report(
            path,
            first,
            volatile_fields=frozenset({"checked_at_utc", "storage_free_bytes"}),
        )
        == first
    )
    assert (
        _persist_idempotent_report(
            path,
            second,
            volatile_fields=frozenset({"checked_at_utc", "storage_free_bytes"}),
        )
        == first
    )


def test_dynamic_preflight_report_is_restart_safe(tmp_path: Path) -> None:
    first = {"checked_at_utc": "first", "status": "pass"}
    second = {"checked_at_utc": "second", "status": "pass"}

    assert _persist_dynamic_preflight_report(tmp_path, first) == first
    assert _persist_dynamic_preflight_report(tmp_path, second) == first


def test_paired_request_audit_uses_pinned_tokens_and_reports_provider_drift() -> None:
    local = [
        {"request_hash": "a", "prompt_tokens_local": 10, "usage": {"prompt_tokens": 10}},
        {"request_hash": "b", "prompt_tokens_local": 20, "usage": {"prompt_tokens": 20}},
    ]
    hub = [
        {"request_hash": "a", "prompt_tokens_local": 10, "usage": {"prompt_tokens": 11}},
        {"request_hash": "b", "prompt_tokens_local": 20, "usage": {"prompt_tokens": 20}},
    ]

    audit = audit_paired_request_events("local", local, "hub", hub)

    assert audit["canonical_prompt_token_equality"] is True
    assert audit["provider_prompt_token_usage_equality"] is False
    assert audit["provider_prompt_token_usage_mismatched_requests"] == 1
    assert audit["provider_prompt_token_usage_delta_counts"] == {"1": 1}

    hub[0]["prompt_tokens_local"] = 12
    with pytest.raises(DedupEvaluationError, match="HOSTING_CANONICAL_PROMPT_TOKEN_MISMATCH"):
        audit_paired_request_events("local", local, "hub", hub)


def test_initial_request_selection_excludes_failed_retry_and_preserves_multiplicity() -> None:
    def event(status: int, request_hash: str = "duplicate") -> dict[str, Any]:
        return {
            "outer_attempt": 1,
            "message_count": 2,
            "http_status": status,
            "request_hash": request_hash,
            "prompt_tokens_local": 10,
            "usage": {"prompt_tokens": 10},
        }

    local_attempts = [event(200), event(200)]
    hub_attempts = [event(500), event(200), event(200)]
    local = select_successful_initial_request_events(
        local_attempts,
        expected_count=2,
        block_id="block",
    )
    hub = select_successful_initial_request_events(
        hub_attempts,
        expected_count=2,
        block_id="block",
    )

    assert local == select_successful_initial_request_events(
        local_attempts,
        expected_count=2,
        block_id="block",
    )
    assert len(hub) == 2
    assert audit_paired_request_events("local", local, "hub", hub)["request_count"] == 2


def test_initial_request_selection_requires_full_successful_workload() -> None:
    events = [
        {
            "outer_attempt": 1,
            "message_count": 2,
            "http_status": 500,
            "request_hash": "failed",
        }
    ]

    with pytest.raises(DedupEvaluationError, match="HOSTING_REQUEST_ACCOUNTING_MISMATCH"):
        select_successful_initial_request_events(events, expected_count=1, block_id="block")


def test_model_provision_reuses_a_matching_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    existing = {
        "schema_version": "hosting-model-provision-v1",
        "model_id": "model-id",
        "revision": "revision",
        "resolved_path": str(model_path),
        "provisioned_at_utc": "first",
    }
    (model_path / ".hosting-benchmark-revision.json").write_text(
        json.dumps(existing) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_kwargs: str(model_path))
    config = SimpleNamespace(
        model=SimpleNamespace(
            local_model_path=model_path,
            local_model_id="model-id",
            local_model_revision="revision",
        ),
        payload=SimpleNamespace(tokenizer_cache_root=tmp_path / "cache"),
    )

    assert provision_model_checkpoint(config) == existing


def test_endpoint_metrics_include_goodput_retries_and_token_rates(tmp_path: Path) -> None:
    markers = []
    for repeat in range(1, 4):
        attempt = tmp_path / f"block-{repeat}" / "attempt-01"
        (attempt / "events").mkdir(parents=True)
        terminals = [
            {"canonical_pair_id": f"pair-{repeat}-a", "record_type": "result", "attempts": 1, "retried": False},
            {"canonical_pair_id": f"pair-{repeat}-b", "record_type": "result", "attempts": 2, "retried": True},
        ]
        events = [
            {
                "duration_seconds": 1.0,
                "http_status": 200,
                "error_type": None,
                "outstanding_at_submit": 1,
                "prompt_tokens_local": 10,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
            {
                "duration_seconds": 2.0,
                "http_status": 200,
                "error_type": None,
                "outstanding_at_submit": 2,
                "prompt_tokens_local": 12,
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        ]
        (attempt / "terminal.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in terminals),
            encoding="utf-8",
        )
        (attempt / "events" / "outer-attempt-01.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events),
            encoding="utf-8",
        )
        markers.append(
            {
                "endpoint": "local",
                "concurrency": 2,
                "attempt_root": str(attempt.relative_to(tmp_path)),
                "duration_seconds": 2.0,
                "goodput_pairs_per_second": 1.0,
            }
        )

    summary = _endpoint_concurrency_summary(tmp_path, "local", 2, markers)

    assert summary["pairs"] == 6
    assert summary["aggregate_goodput_pairs_per_second"] == 1.0
    assert summary["pair_attempts"] == 9
    assert summary["retried_pairs"] == 3
    assert summary["raw_requests_per_wall_second"] == 1.0
    assert summary["prompt_token_count_source"] == "pinned_client_tokenizer"  # noqa: S105
    assert summary["provider_reported_prompt_tokens"] == 66
    assert summary["total_tokens_per_wall_second"] == 14.0
    assert summary["completion_tokens_p50"] == 3.0


class _Upstream:
    def __init__(self, *, status: int = 200, reasoning_content: str | None = None) -> None:
        self.status = status
        self.reasoning_content = reasoning_content
        self.requests: list[dict[str, Any]] = []
        self.authorization: list[str | None] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                outer.authorization.append(self.headers.get("Authorization"))
                message = {"content": "{}"}
                if outer.reasoning_content is not None:
                    message["reasoning_content"] = outer.reasoning_content
                response = {
                    "choices": [{"message": message}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
                body = json.dumps(response).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self.server = LoopbackThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _post(endpoint: str, request: dict[str, Any]) -> int:
    call = urllib.request.Request(  # noqa: S310 - test endpoint is a loopback HTTP server
        endpoint + "/chat/completions",
        data=json.dumps(request).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(call) as response:  # noqa: S310
            response.read()
            return response.status
    except urllib.error.HTTPError as exc:
        exc.read()
        return exc.code


def test_relay_hashes_before_model_rewrite_and_redacts_request_content(tmp_path: Path) -> None:
    request = {
        "model": "logical-model",
        "messages": [{"role": "user", "content": "sensitive-document"}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 16,
        "stream": False,
    }
    events = tmp_path / "events.jsonl"
    with (
        _Upstream() as upstream,
        RequestRelay(
            logical_model="logical-model",
            max_model_len=128,
            prompt_token_counter=lambda messages: len(messages) + 6,
        ) as relay,
    ):
        relay.set_route(
            target=RelayTarget(
                label="hub",
                base_url=upstream.base_url,
                model="provider-model",
                api_key="secret-token",  # pragma: allowlist secret
            ),
            context=RelayContext(
                endpoint="hub",
                block_id="block",
                outer_attempt=1,
                events_path=events,
                queued_at_utc="2026-08-27T00:00:00+00:00",
            ),
        )
        assert _post(relay.endpoint, request) == 200

    assert upstream.requests[0]["model"] == "provider-model"
    assert upstream.authorization == ["Bearer secret-token"]  # pragma: allowlist secret
    event = json.loads(events.read_text())
    assert event["request_hash"] == canonical_request_hash(request)
    assert event["usage"]["prompt_tokens"] == 7
    assert "sensitive-document" not in events.read_text()
    assert "secret-token" not in events.read_text()


def test_relay_preserves_429_and_rejects_context_overflow_before_forwarding(tmp_path: Path) -> None:
    request = {
        "model": "logical-model",
        "messages": [{"role": "user", "content": "text"}],
        "max_tokens": 16,
        "stream": False,
    }
    events = tmp_path / "events.jsonl"
    with (
        _Upstream(status=429) as upstream,
        RequestRelay(
            logical_model="logical-model",
            max_model_len=20,
            prompt_token_counter=lambda _messages: 4,
        ) as relay,
    ):
        relay.set_route(
            target=RelayTarget(label="hub", base_url=upstream.base_url, model="provider-model"),
            context=RelayContext(
                endpoint="hub",
                block_id="quota",
                outer_attempt=1,
                events_path=events,
                queued_at_utc="2026-08-27T00:00:00+00:00",
            ),
        )
        assert _post(relay.endpoint, request) == 429
        relay.set_route(
            target=RelayTarget(label="hub", base_url=upstream.base_url, model="provider-model"),
            context=RelayContext(
                endpoint="hub",
                block_id="overflow",
                outer_attempt=1,
                events_path=events,
                queued_at_utc="2026-08-27T00:00:00+00:00",
            ),
        )
        overflow = {**request, "max_tokens": 17}
        assert _post(relay.endpoint, overflow) == 400

    assert len(upstream.requests) == 1
    event_rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [row["http_status"] for row in event_rows] == [429, 400]
    assert event_rows[1]["error_type"] == "context_overflow"


def test_relay_rejects_generation_drift_and_flags_visible_thinking(tmp_path: Path) -> None:
    request = {
        "model": "logical-model",
        "messages": [{"role": "user", "content": "text"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    events = tmp_path / "events.jsonl"
    expected = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with (
        _Upstream(reasoning_content="unexpected") as upstream,
        RequestRelay(
            logical_model="logical-model",
            max_model_len=128,
            expected_generation_parameters=expected,
        ) as relay,
    ):
        context = RelayContext(
            endpoint="hub",
            block_id="generation-contract",
            outer_attempt=1,
            events_path=events,
            queued_at_utc="2026-08-27T00:00:00+00:00",
        )
        relay.set_route(
            target=RelayTarget(label="hub", base_url=upstream.base_url, model="provider-model"),
            context=context,
        )
        assert _post(relay.endpoint, {**request, "temperature": 0.1}) == 400
        relay.set_route(
            target=RelayTarget(label="hub", base_url=upstream.base_url, model="provider-model"),
            context=context,
        )
        assert _post(relay.endpoint, request) == 200

    assert len(upstream.requests) == 1
    event_rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert event_rows[0]["generation_parameters_valid"] is False
    assert event_rows[1]["generation_parameters_valid"] is True
    assert event_rows[1]["thinking_content_observed"] is True
