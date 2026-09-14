# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bounded service waits with shared cancellation and append-only checkpoints."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import requests

from eval.dedup.analysis import exp1_rate_limit_continuation as old
from eval.dedup.judging.service_wait_relay import CONTRACT
from eval.dedup.validation import sha256_json, write_json_atomic

recovery = old.recovery


class ServiceWaitPending(BaseException):
    """Escape native semantic correction, leaving uncompleted pairs pending."""


class ServiceWaitCollector(old.RateLimitCollector):
    max_service_submissions = 6
    max_service_wait_seconds = 1800.0
    max_rate_wait_seconds = 21600.0

    def __init__(self, root: Path, session: Path):
        super().__init__(root, session)
        self.cancelled = threading.Event()
        self.service_retries = 0
        self.stop_reason: str | None = None

    def heartbeat(self) -> dict:
        value = super().heartbeat()
        with self.lock:
            return {**value, "service_retries": self.service_retries, "stop_reason": self.stop_reason}

    def stop_pending(self, reason: str) -> None:
        with self.lock:
            if self.stop_reason is None:
                self.stop_reason = reason
                recovery.append_event(
                    self.session / "collection_events.jsonl", {"event": "STOP_PENDING", "reason": reason}
                )
        self.cancelled.set()
        raise ServiceWaitPending(reason)

    def __call__(self, root: Path, key: str, request: dict, endpoint: str) -> dict:
        recovery.intact(root.resolve() == self.root and Path(key).name == key)
        if (root / "responses" / (key + ".json")).exists():
            return recovery.Collector.__call__(self, root, key, request, endpoint)
        if self.cancelled.is_set():
            raise ServiceWaitPending(self.stop_reason)
        saved = {"body": request, "request_sha256": sha256_json(request)}
        path = root / "requests" / (key + ".json")
        existed = path.exists()
        write_json_atomic(path, saved)
        with self.lock:
            recovery.intact(key not in self.active)
            self.active[key] = recovery.now()
            self.retransmitted += int(existed)
        counts, waited = {"SERVICE_UNAVAILABLE": 0, "RATE_LIMIT": 0}, {"SERVICE_UNAVAILABLE": 0.0, "RATE_LIMIT": 0.0}
        attempt = 0
        try:
            while not self.cancelled.is_set():
                attempt += 1
                with self.lock:
                    self.submitted += 1
                    recovery.append_event(
                        self.session / "collection_events.jsonl",
                        {
                            "event": "SUBMIT",
                            "key": key,
                            "request_sha256": saved["request_sha256"],
                            "submission": attempt,
                            "interrupted_request_retransmission": existed and attempt == 1,
                        },
                    )
                try:
                    response = requests.post(endpoint + "/chat/completions", json=request, timeout=660)
                except requests.RequestException:
                    self.stop_pending("LOCAL_RELAY_CONNECTION_FAILED_OUTCOME_UNKNOWN")
                try:
                    value = response.json()
                except ValueError:
                    self.stop_pending("LOCAL_RELAY_INVALID_JSON")
                if response.status_code == 200:
                    receipt = {
                        "request_sha256": saved["request_sha256"],
                        "http_status": 200,
                        "status": "RECEIVED",
                        "raw_response": value,
                    }
                    write_json_atomic(root / "responses" / (key + ".json"), receipt)
                    with self.lock:
                        recovery.append_event(self.session / "collection_events.jsonl", {"event": "SAVED", "key": key})
                    return receipt
                if not isinstance(value, dict) or value.get("service_wait_contract") != CONTRACT:
                    self.stop_pending("UNRECOGNIZED_RELAY_FAILURE")
                if value.get("action") == "STOP":
                    self.stop_pending(str(value.get("reason", "TRANSPORT_STOP")))
                reason = value.get("reason")
                if value.get("action") != "WAIT" or reason not in counts:
                    self.stop_pending("INVALID_WAIT_CONTROL")
                delay = value.get("retry_after_seconds")
                if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
                    self.stop_pending("INVALID_WAIT_DELAY")
                counts[reason] += 1
                delay = max(
                    delay, min(self.retry_cap_seconds, self.retry_base_seconds * 2 ** min(counts[reason] - 1, 10))
                )
                limit = self.max_service_submissions if reason == "SERVICE_UNAVAILABLE" else self.max_rate_attempts
                budget = (
                    self.max_service_wait_seconds if reason == "SERVICE_UNAVAILABLE" else self.max_rate_wait_seconds
                )
                if counts[reason] >= limit or waited[reason] + delay > budget:
                    self.stop_pending(reason + "_BUDGET_EXHAUSTED")
                waited[reason] += delay
                with self.lock:
                    self.service_retries += reason == "SERVICE_UNAVAILABLE"
                    self.rate_retries += reason == "RATE_LIMIT"
                    self.waiting[key] = {"reason": reason, "wait_seconds": delay, "submission": attempt}
                    recovery.append_event(
                        self.session / "collection_events.jsonl",
                        {
                            "event": "SERVICE_WAIT",
                            "key": key,
                            "reason": reason,
                            "wait_seconds": delay,
                            "submission": attempt,
                            "provider_diagnostic": value.get("provider_diagnostic"),
                        },
                    )
                if self.cancelled.wait(delay):
                    raise ServiceWaitPending(self.stop_reason)
                with self.lock:
                    self.waiting.pop(key, None)
            raise ServiceWaitPending(self.stop_reason)
        finally:
            with self.lock:
                self.active.pop(key, None)
                self.waiting.pop(key, None)
