# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Rate-limit-aware transport used by both Hub and local release runs."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from eval.dedup.core.validation import require, sha256_json, write_json_atomic
from eval.dedup.judging.rate_limit_relay import CONTRACT, RateLimitRelay
from eval.dedup.runtime import state


class RateLimitWaitBudget(BaseException):
    """Leave a pair pending instead of converting provider throttling into a semantic result."""


class RateLimitCollector(state.Collector):
    retry_base_seconds = 60.0
    retry_cap_seconds = 300.0
    max_rate_attempts = 60
    max_wait_window_seconds = 6 * 3600

    def __init__(self, root: Path, session: Path):
        super().__init__(root, session)
        self.waiting: dict[str, dict] = {}
        self.rate_retries = 0

    def heartbeat(self) -> dict:
        value = super().heartbeat()
        with self.lock:
            return {**value, "rate_limit_waiting": dict(self.waiting), "rate_limit_retries": self.rate_retries}

    def __call__(self, root: Path, key: str, request: dict, endpoint: str) -> dict:
        receipt_path = root / "responses" / f"{key}.json"
        if receipt_path.exists():
            return super().__call__(root, key, request, endpoint)
        state.intact(root.resolve() == self.root and Path(key).name == key)
        request_path = root / "requests" / f"{key}.json"
        saved = {"body": request, "request_sha256": sha256_json(request)}
        existed = request_path.exists()
        if existed:
            state.intact(state.read(request_path) == saved)
        else:
            write_json_atomic(request_path, saved)
        with self.lock:
            state.intact(key not in self.active)
            self.active[key] = state.now()
            self.retransmitted += int(existed)
        started = time.monotonic()
        for attempt in range(1, self.max_rate_attempts + 1):
            with self.lock:
                self.submitted += 1
                state.append_event(
                    self.session / "collection_events.jsonl",
                    {
                        "event": "SUBMIT",
                        "key": key,
                        "request_sha256": saved["request_sha256"],
                        "interrupted_request_retransmission": existed and attempt == 1,
                        "rate_limit_retry": attempt > 1,
                        "rate_attempt": attempt,
                    },
                )
            receipt = {"request_sha256": saved["request_sha256"], "status": "FAILURE"}
            limited = None
            try:
                response = requests.post(endpoint + "/chat/completions", json=request, timeout=660)
                receipt["http_status"] = response.status_code
                if response.status_code == 429:
                    limited = response.json()
                    state.intact(limited.get("transport_contract") == CONTRACT)
                else:
                    require(response.status_code == 200, "DEDUP_HTTP", "request failed; retain the failure")
                    receipt.update(status="RECEIVED", raw_response=response.json())
            except Exception as exc:  # noqa: BLE001 - preserve credential-safe receipt shape
                receipt["error_code"] = state.error_code(exc)
            if limited is None:
                write_json_atomic(receipt_path, receipt)
                with self.lock:
                    state.append_event(self.session / "collection_events.jsonl", {"event": "SAVED", "key": key})
                    self.active.pop(key)
                return receipt
            delay = max(
                float(limited["retry_after_seconds"]),
                min(self.retry_cap_seconds, self.retry_base_seconds * 2 ** min(attempt - 1, 10)),
            )
            with self.lock:
                self.rate_retries += 1
                state.append_event(
                    self.session / "collection_events.jsonl",
                    {"event": "RATE_LIMIT_WAIT", "key": key, "rate_attempt": attempt, "wait_seconds": delay},
                )
                self.waiting[key] = {
                    "until_utc": datetime.fromtimestamp(time.time() + delay, UTC).isoformat(),
                    "rate_attempt": attempt,
                }
            if attempt == self.max_rate_attempts or time.monotonic() - started + delay > self.max_wait_window_seconds:
                raise RateLimitWaitBudget("RATE_LIMIT_WAIT_BUDGET_EXHAUSTED_PAIR_PENDING")
            until = time.monotonic() + delay
            while time.monotonic() < until:
                threading.Event().wait(min(30, max(0, until - time.monotonic())))
            with self.lock:
                self.waiting.pop(key)
        raise RateLimitWaitBudget("RATE_LIMIT_WAIT_BUDGET_EXHAUSTED_PAIR_PENDING")


__all__ = ["RateLimitCollector", "RateLimitRelay", "RateLimitWaitBudget"]
