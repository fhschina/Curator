# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Continue the same frozen exp1 experiment while treating temporary 429s as waiting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import requests

from eval.dedup.analysis import exp1_reproduction_recovery as recovery
from eval.dedup.judging.rate_limit_relay import CONTRACT, RateLimitRelay
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

HERE = Path(__file__).resolve()
PROTOCOL = HERE.with_suffix(".md")


class RateLimitWaitBudget(BaseException):
    """Leave the pair pending rather than manufacture a semantic failure from a 429."""


class RateLimitCollector(recovery.Collector):
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
        receipt_path = root / "responses" / (key + ".json")
        if receipt_path.exists():
            return super().__call__(root, key, request, endpoint)
        recovery.intact(root.resolve() == self.root and Path(key).name == key)
        request_path = root / "requests" / (key + ".json")
        saved = {"body": request, "request_sha256": sha256_json(request)}
        existed = request_path.exists()
        if existed:
            recovery.intact(recovery.read(request_path) == saved)
        else:
            write_json_atomic(request_path, saved)
        with self.lock:
            recovery.intact(key not in self.active)
            self.active[key] = recovery.now()
            self.retransmitted += int(existed)
        started = time.monotonic()
        for attempt in range(1, self.max_rate_attempts + 1):
            with self.lock:
                self.submitted += 1
                recovery.append_event(
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
                    recovery.intact(limited.get("transport_contract") == CONTRACT)
                else:
                    require(response.status_code == 200, "EXP1_HTTP", "request failed; retain the failure")
                    receipt.update(status="RECEIVED", raw_response=response.json())
            except Exception as exc:  # noqa: BLE001 - preserve the original credential-safe receipt shape
                receipt["error_code"] = recovery.error_code(exc)
            if limited is None:
                write_json_atomic(receipt_path, receipt)
                with self.lock:
                    recovery.append_event(self.session / "collection_events.jsonl", {"event": "SAVED", "key": key})
                    self.active.pop(key)
                return receipt
            delay = max(
                float(limited["retry_after_seconds"]),
                min(self.retry_cap_seconds, self.retry_base_seconds * 2 ** min(attempt - 1, 10)),
            )
            with self.lock:
                self.rate_retries += 1
                recovery.append_event(
                    self.session / "collection_events.jsonl",
                    {
                        "event": "RATE_LIMIT_WAIT",
                        "key": key,
                        "rate_attempt": attempt,
                        "wait_seconds": delay,
                        "provider_diagnostic": limited.get("provider_diagnostic"),
                    },
                )
                self.waiting[key] = {
                    "until_utc": datetime.fromtimestamp(time.time() + delay, UTC).isoformat(),
                    "rate_attempt": attempt,
                }
            if attempt == self.max_rate_attempts or time.monotonic() - started + delay > self.max_wait_window_seconds:
                raise RateLimitWaitBudget("RATE_LIMIT_WAIT_BUDGET_EXHAUSTED_PAIR_PENDING")
            # Chunk the wait so process heartbeats remain independently observable.
            until = time.monotonic() + delay
            while time.monotonic() < until:
                threading.Event().wait(min(30, max(0, until - time.monotonic())))
            with self.lock:
                self.waiting.pop(key)
        raise RateLimitWaitBudget("RATE_LIMIT_WAIT_BUDGET_EXHAUSTED_PAIR_PENDING")


def prepare(root: Path) -> dict:
    with recovery.exclusive(root):
        recovery.verify(root)
        require(
            not recovery.status(root)["running"] and not (root / "complete.json").exists(),
            "RATE_LIMIT_STATE",
            "inactive incomplete run required",
        )
        require(not (root / "rate_limit_manifest.json").exists(), "RATE_LIMIT_FROZEN", "already frozen")
        inventory = recovery.inventory(root)
        source = [
            HERE,
            PROTOCOL,
            Path(sys.modules[RateLimitRelay.__module__].__file__).resolve(),
            HERE.parents[3] / "tests/eval/dedup/test_rate_limit_relay.py",
            HERE.parents[3] / "tests/eval/dedup/test_exp1_rate_limit_continuation.py",
        ]
        snapshot = [p for folder in ("requests", "responses", "results") for p in (root / folder).glob("*.json")]
        prefix = (root / "transport_events.jsonl").read_bytes()
        manifest = {
            "contract_version": CONTRACT,
            "at_utc": recovery.now(),
            "semantic_changes": False,
            "original_manifest_sha256": sha256_file(root / "manifest.json"),
            "recovery_manifest_sha256": sha256_file(root / "recovery_manifest.json"),
            "sources": {str(p): sha256_file(p) for p in source},
            "artifacts": {str(p): sha256_file(p) for p in snapshot},
            "initial_inventory": inventory,
            "transport_prefix_bytes": len(prefix),
            "transport_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "transport_changes": {
                "temporary_429_is_pending": True,
                "wait_base_seconds": 60,
                "wait_cap_seconds": 300,
                "retry_after_never_shortened": True,
                "max_rate_attempts_per_stage": 60,
                "max_stage_wait_window_seconds": 21600,
                "workers": 2,
                "max_external_attempts_unchanged": 24000,
            },
            "existing_failures_preserved": [
                recovery.read(p)["review_id"]
                for p in (root / "results").glob("*.json")
                if recovery.read(p)["status"] != "VALID"
            ],
        }
        manifest["contract_digest"] = sha256_json(manifest)
        write_json_atomic(root / "rate_limit_manifest.json", manifest)
        return {
            "completed": len(inventory["completed_pair_ids"]),
            "existing_failures_preserved": manifest["existing_failures_preserved"],
        }


def verify(root: Path) -> None:
    recovery.original.reference.verify_freeze(root / "rate_limit_manifest.json")
    manifest = recovery.read(root / "rate_limit_manifest.json")
    recovery.intact(
        sha256_file(root / "manifest.json") == manifest["original_manifest_sha256"]
        and sha256_file(root / "recovery_manifest.json") == manifest["recovery_manifest_sha256"]
    )
    with (root / "transport_events.jsonl").open("rb") as stream:
        prefix = stream.read(manifest["transport_prefix_bytes"])
    recovery.intact(hashlib.sha256(prefix).hexdigest() == manifest["transport_prefix_sha256"])


@contextmanager
def transport_override(root: Path) -> Iterator[None]:
    old_relay, old_collector = recovery.original.PacedRelayV2, recovery.Collector
    old_verify, old_export = recovery.verify, recovery.original.export

    def check(path: Path) -> dict:
        verify(path)
        return old_verify(path)

    def export(path: Path) -> dict:
        rows = recovery.events(path / "transport_events.jsonl")
        selected = [e for e in rows if e.get("transport_contract") == CONTRACT]
        write_json_atomic(
            path / "rate_limit_final.json",
            {
                "transport_contract": CONTRACT,
                "semantic_changes": False,
                "observed_external_attempts": len(selected),
                "rate_limit_responses": sum(e["http_status"] == 429 for e in selected),
                "provider_diagnostics": [e for e in selected if "provider_diagnostic" in e],
                "accounting_note": "Original transport_retries counts within-relay retries only. Repeated 429 requests and waits are additionally recorded here and in collection_events; no 429 is silently converted to a correct Judge answer.",
            },
        )
        return old_export(path)

    check(root)
    recovery.original.PacedRelayV2, recovery.Collector = RateLimitRelay, RateLimitCollector
    recovery.verify, recovery.original.export = check, export
    try:
        yield
    finally:
        recovery.original.PacedRelayV2, recovery.Collector = old_relay, old_collector
        recovery.verify, recovery.original.export = old_verify, old_export


def launch(root: Path, env_file: Path) -> dict:
    with recovery.exclusive(root):
        recovery.verify(root)
        verify(root)
        require(
            not recovery.status(root)["running"] and not (root / "complete.json").exists(),
            "RATE_LIMIT_LAUNCH",
            "must not duplicate an active or complete run",
        )
        session = root / "recovery/sessions" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        name = "exp1-rate-wait-" + sha256_json(str(root))[:12]
        write_json_atomic(
            session / "launch.json",
            {
                "at_utc": recovery.now(),
                "tmux_session": name,
                "transport_contract": CONTRACT,
                "credential_persisted": False,
            },
        )
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                name,
                "-c",
                str(HERE.parents[3]),
                sys.executable,
                "-m",
                "eval.dedup.analysis.exp1_rate_limit_continuation",
                "run",
                "--root",
                str(root),
                "--env-file",
                str(env_file),
                "--session",
                str(session),
            ],
            check=True,
            capture_output=True,
        )
        return {"session": str(session), "tmux_session": name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "run", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "run":
        require(args.env_file is not None and args.session is not None, "RATE_LIMIT_ARGS", "paths required")
        with (args.session / "run.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                with transport_override(root):
                    recovery.run(root, args.env_file.resolve(), args.session)
            except BaseException as exc:  # noqa: BLE001 - never log credential-bearing exception messages
                print(json.dumps({"error_code": recovery.error_code(exc)}), flush=True)
                return 1
        return 0
    if args.command == "prepare":
        result = prepare(root)
    elif args.command == "launch":
        require(args.env_file is not None, "RATE_LIMIT_ARGS", "existing env file required")
        result = launch(root, args.env_file.resolve())
    else:
        result = recovery.status(root)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
