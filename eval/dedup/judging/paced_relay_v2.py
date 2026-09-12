# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Versioned bounded HTTP 500 recovery without changing historical relay behavior."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

from eval.dedup.judging.paced_relay import PacedRelay, retry_after_seconds
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.validation import sha256_json

TRANSPORT_CONTRACT = "dedup-paced-transport-v2"


class PacedRelayV2(PacedRelay):
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
                retryable = status in {429, 500, 502, 503, 504}
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
