# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Separate transient service unavailability from terminal semantic judgments."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

from eval.dedup.analysis.exp1_reproduction_recovery import error_code
from eval.dedup.judging.auth_database_busy_relay import AuthDatabaseBusyRelay, is_database_busy_auth
from eval.dedup.judging.paced_relay import retry_after_seconds
from eval.dedup.judging.rate_limit_relay import CONTRACT as BASE_CONTRACT
from eval.dedup.judging.rate_limit_relay import error_diagnostic
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.validation import DedupEvaluationError, sha256_json

CONTRACT = "dedup-service-unavailable-wait-v1"


class ServiceWaitRelay(AuthDatabaseBusyRelay):
    service_cooldown_seconds = 60.0

    @staticmethod
    def control(action: str, reason: str, delay: float, diagnostic: dict | None = None) -> tuple[int, str, bytes]:
        return (
            503,
            "application/json",
            json.dumps(
                {
                    "service_wait_contract": CONTRACT,
                    "action": action,
                    "reason": reason,
                    "retry_after_seconds": delay,
                    "provider_diagnostic": diagnostic,
                }
            ).encode(),
        )

    def _forward(self, body: dict, context: RelayContext, sequence: int, outstanding: int) -> tuple[int, str, bytes]:
        deadline = time.monotonic() + self.profile.request_deadline_seconds
        request_hash = sha256_json(body)
        data = json.dumps({**body, "model": self.upstream_model}, ensure_ascii=False, separators=(",", ":")).encode()
        for attempt in range(1, self.profile.max_attempts + 1):
            try:
                external_sequence, queued = self._admit(deadline)
            except (DedupEvaluationError, TimeoutError) as exc:
                return self.control("STOP", error_code(exc), 0.0)
            started = time.monotonic()
            status, upstream_status, error_type, retry_after = 502, None, None, 0.0
            content_type, response_body = "application/json", b'{"error":{"message":"upstream transport unavailable"}}'
            diagnostic = None
            try:
                request = urllib.request.Request(  # noqa: S310 - frozen configured upstream only
                    self.upstream_base_url.rstrip("/") + "/chat/completions",
                    data=data,
                    headers={"Authorization": f"Bearer {self._upstream_api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(  # noqa: S310 - frozen configured upstream only
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
                    diagnostic = error_diagnostic(response_body, dict(exc.headers), self._upstream_api_key)
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    error_type = type(exc).__name__
                database_busy = is_database_busy_auth(status, response_body)
                retryable = status in {429, 500, 502, 503, 504} or database_busy
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
                        "transport_contract": BASE_CONTRACT,
                        "transport_patch_contract": CONTRACT,
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
                        "auth_database_busy": database_busy,
                        **({"provider_diagnostic": diagnostic} if diagnostic is not None else {}),
                    },
                )
            finally:
                self._slots.release()
            if status == 429:
                return self.control("WAIT", "RATE_LIMIT", delay, diagnostic)
            if not retryable:
                if status >= 400:
                    self._open_circuit("nonretryable_upstream_error")
                    return self.control("STOP", "NONRETRYABLE_UPSTREAM", 0.0, diagnostic)
                return status, content_type, response_body
            if attempt == self.profile.max_attempts or time.monotonic() + delay >= deadline:
                delay = max(delay, self.service_cooldown_seconds)
                with self._admission:
                    self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)
                    self._admission.notify_all()
                return self.control("WAIT", "SERVICE_UNAVAILABLE", delay, diagnostic)
        return self.control("STOP", "UNEXPECTED_TRANSPORT_EXIT", 0.0)
