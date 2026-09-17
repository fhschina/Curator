# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Credential-safe upstream pacing with bounded transport retries and per-attempt receipts."""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler

from eval.dedup.core.validation import require, sha256_json
from eval.dedup.judging.request_relay import RelayContext, RequestRelay

TRANSPORT_CONTRACT = "dedup-paced-transport-v1"


@dataclass(frozen=True)
class TransportProfile:
    min_interval_seconds: float = 2.0
    max_in_flight: int = 4
    max_attempts: int = 4
    retry_base_seconds: float = 10.0
    retry_cap_seconds: float = 60.0
    request_deadline_seconds: float = 480.0
    max_external_attempts: int = 5000

    def __post_init__(self) -> None:
        require(
            all(
                math.isfinite(v) and v > 0
                for v in (
                    self.min_interval_seconds,
                    self.retry_base_seconds,
                    self.retry_cap_seconds,
                    self.request_deadline_seconds,
                )
            )
            and all(
                type(v) is int and v > 0 for v in (self.max_in_flight, self.max_attempts, self.max_external_attempts)
            )
            and self.retry_base_seconds <= self.retry_cap_seconds < self.request_deadline_seconds,
            "TRANSPORT_PROFILE",
            "finite positive pacing and bounded retry/deadline settings required",
        )


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float:
    if not value:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            delay = (date - (now or datetime.now(UTC))).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


class PacedRelay(RequestRelay):
    def __init__(self, *, profile: TransportProfile, **kwargs):
        super().__init__(**kwargs)
        self.profile = profile
        self._admission = threading.Condition()
        self._slots = threading.BoundedSemaphore(profile.max_in_flight)
        self._next_start = 0.0
        self._cooldown_until = 0.0
        self._external_attempts = 0
        self._circuit_reason: str | None = None
        self._stopping = threading.Event()

    def stop(self) -> None:
        self._stopping.set()
        with self._admission:
            self._admission.notify_all()
        super().stop()

    def _open_circuit(self, reason: str) -> None:
        with self._admission:
            self._circuit_reason = reason
            self._admission.notify_all()

    def _admit(self, deadline: float) -> tuple[int, float]:
        queued = time.monotonic()
        while not self._slots.acquire(timeout=0.1):
            if time.monotonic() >= deadline or self._stopping.is_set() or self._circuit_reason:
                raise TimeoutError("transport admission unavailable")
        try:
            with self._admission:
                while True:
                    now = time.monotonic()
                    require(
                        not self._stopping.is_set() and self._circuit_reason is None,
                        "TRANSPORT_CIRCUIT_OPEN",
                        "transport has stopped admitting requests",
                    )
                    require(
                        self._external_attempts < self.profile.max_external_attempts,
                        "TRANSPORT_ATTEMPT_BUDGET",
                        "external request budget exhausted",
                    )
                    if now >= deadline:
                        raise TimeoutError("transport admission deadline exceeded")  # noqa: TRY301
                    delay = max(self._next_start, self._cooldown_until) - now
                    if delay <= 0:
                        self._next_start = now + self.profile.min_interval_seconds
                        self._external_attempts += 1
                        return self._external_attempts, now - queued
                    self._admission.wait(timeout=min(delay, deadline - now, 1.0))
        except BaseException:
            self._slots.release()
            raise

    def _forward(self, body: dict, context: RelayContext, sequence: int, outstanding: int) -> tuple[int, str, bytes]:
        deadline = time.monotonic() + self.profile.request_deadline_seconds
        request_hash = sha256_json(body)
        forwarded = {**body, "model": self.upstream_model}
        data = json.dumps(forwarded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for attempt in range(1, self.profile.max_attempts + 1):
            external_sequence, queued = self._admit(deadline)
            started = time.monotonic()
            status, upstream_status, error_type, retry_after = 502, None, None, 0.0
            content_type, response_body = "application/json", b'{"error":{"message":"upstream transport unavailable"}}'
            try:
                request = urllib.request.Request(  # noqa: S310 - frozen upstream URL, never selected by input text
                    self.upstream_base_url.rstrip("/") + "/chat/completions",
                    data=data,
                    headers={"Authorization": f"Bearer {self._upstream_api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(  # noqa: S310 - configured upstream only
                        request, timeout=min(self.timeout_seconds, max(0.001, deadline - started))
                    ) as response:
                        response_body, status = response.read(), response.status
                        upstream_status = status
                        content_type = response.headers.get("Content-Type", "application/json")
                except urllib.error.HTTPError as exc:
                    response_body, status, upstream_status = exc.read(), exc.code, exc.code
                    content_type = exc.headers.get("Content-Type", "application/json")
                    retry_after = retry_after_seconds(exc.headers.get("Retry-After"))
                    error_type = "http_error"
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    error_type = type(exc).__name__
                retryable = status in {429, 502, 503, 504}
                delay = (
                    max(
                        retry_after,
                        min(self.profile.retry_cap_seconds, self.profile.retry_base_seconds * 2 ** (attempt - 1)),
                    )
                    if retryable
                    else 0.0
                )
                if retryable:
                    with self._admission:
                        self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)
                        self._admission.notify_all()
                self._write_event(
                    context,
                    {
                        "schema_version": "dedup-paced-upstream-event-v1",
                        "transport_contract": TRANSPORT_CONTRACT,
                        "sequence": external_sequence,
                        "logical_request_sequence": sequence,
                        "upstream_attempt": attempt,
                        "block_id": context.block_id,
                        "outer_attempt": context.outer_attempt,
                        "outstanding_at_submit": outstanding,
                        "request_hash": request_hash,
                        "external_request": True,
                        "started_at_utc": datetime.fromtimestamp(
                            time.time() - (time.monotonic() - started), UTC
                        ).isoformat(),
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "duration_seconds": time.monotonic() - started,
                        "admission_wait_seconds": queued,
                        "http_status": status,
                        "upstream_http_status": upstream_status,
                        "retry_after_seconds": retry_after,
                        "cooldown_seconds": delay,
                        "error_type": error_type,
                    },
                )
            finally:
                self._slots.release()
            if not retryable:
                if status >= 400:
                    self._open_circuit("nonretryable_upstream_error")
                return status, content_type, response_body
            if attempt == self.profile.max_attempts or time.monotonic() + delay >= deadline:
                self._open_circuit("bounded_transport_retries_exhausted")
                break
        # Nonretryable local error prevents an SDK from starting another unbounded 429 retry loop.
        return (
            400,
            "application/json",
            b'{"error":{"message":"bounded transport retries exhausted; inspect upstream receipts"}}',
        )

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        context = None
        status, content_type, response_body = (
            400,
            "application/json",
            b'{"error":{"message":"transport request rejected"}}',
        )
        try:
            sequence, outstanding, context = self._begin()
            require(handler.path == "/v1/chat/completions", "TRANSPORT_PATH", "unsupported relay path")
            body = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", "0"))))
            require(
                isinstance(body, dict)
                and body.get("model") == self.logical_model
                and body.get("stream") is not True
                and all(body.get(k) == v for k, v in self.expected_generation_parameters.items()),
                "TRANSPORT_REQUEST_CONTRACT",
                "request differs from the frozen model or generation settings",
            )
            status, content_type, response_body = self._forward(body, context, sequence, outstanding)
        except Exception as exc:  # noqa: BLE001 - never expose credential, request text, or upstream exception details
            if context is not None:
                self._write_event(
                    context,
                    {
                        "schema_version": "dedup-paced-local-event-v1",
                        "external_request": False,
                        "block_id": context.block_id,
                        "outer_attempt": context.outer_attempt,
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "http_status": 400,
                        "error_type": type(exc).__name__,
                    },
                )
        finally:
            if context is not None:
                self._finish()
            self._send(handler, status, content_type, response_body)
