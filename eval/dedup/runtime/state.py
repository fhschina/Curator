# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Append-only run state, locking, and resumable request collection."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import requests

from eval.dedup.core.validation import DedupEvaluationError, require, sha256_json, write_json_atomic


def now() -> str:
    return datetime.now(UTC).isoformat()


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def error_code(exc: BaseException) -> str:
    return exc.issue.code if isinstance(exc, DedupEvaluationError) else type(exc).__name__


class CheckpointIntegrityError(BaseException):
    """Keep checkpoint corruption outside Judge retry and scoring paths."""


def intact(condition: bool) -> None:
    if not condition:
        raise CheckpointIntegrityError("DEDUP_CHECKPOINT_INTEGRITY")


@contextmanager
def exclusive(root: Path) -> Iterator[None]:
    directory = root / "recovery"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("DEDUP_RUN_ALREADY_ACTIVE") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def append_event(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps({"at_utc": now(), **value}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class Collector:
    """Persist every request and response while allowing interrupted runs to resume."""

    def __init__(self, root: Path, session: Path):
        self.root, self.session = root.resolve(), session
        self.lock = threading.Lock()
        self.active: dict[str, str] = {}
        self.replayed = self.submitted = self.retransmitted = 0

    def __call__(self, root: Path, key: str, request: dict, endpoint: str) -> dict:
        intact(root.resolve() == self.root and Path(key).name == key)
        request_path, receipt_path = (root / folder / f"{key}.json" for folder in ("requests", "responses"))
        saved = {"body": request, "request_sha256": sha256_json(request)}
        existed = request_path.exists()
        if existed:
            intact(read(request_path) == saved)
        else:
            intact(not receipt_path.exists())
            write_json_atomic(request_path, saved)
        if receipt_path.exists():
            receipt = read(receipt_path)
            intact(receipt["request_sha256"] == saved["request_sha256"])
            with self.lock:
                self.replayed += 1
            return receipt
        with self.lock:
            intact(key not in self.active)
            self.active[key] = now()
            self.submitted += 1
            self.retransmitted += int(existed)
            append_event(
                self.session / "collection_events.jsonl",
                {
                    "event": "SUBMIT",
                    "key": key,
                    "request_sha256": saved["request_sha256"],
                    "interrupted_request_retransmission": existed,
                },
            )
        receipt = {"request_sha256": saved["request_sha256"], "status": "FAILURE"}
        try:
            response = requests.post(endpoint + "/chat/completions", json=request, timeout=660)
            receipt["http_status"] = response.status_code
            require(response.status_code == 200, "DEDUP_HTTP", "request failed; retain the failure")
            receipt.update(status="RECEIVED", raw_response=response.json())
        except Exception as exc:  # noqa: BLE001 - do not persist credential-bearing exception messages
            receipt["error_code"] = error_code(exc)
        write_json_atomic(receipt_path, receipt)
        with self.lock:
            append_event(self.session / "collection_events.jsonl", {"event": "SAVED", "key": key})
            self.active.pop(key)
        return receipt

    def heartbeat(self) -> dict:
        with self.lock:
            return {
                "active_requests": dict(self.active),
                "checkpoint_calls_replayed": self.replayed,
                "logical_calls_submitted": self.submitted,
                "interrupted_requests_retransmitted": self.retransmitted,
            }


def process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def status(root: Path) -> dict:
    sessions = sorted((root / "recovery/sessions").glob("*"))
    info = {
        "completed": len(list((root / "results").glob("*.json"))),
        "complete": (root / "complete.json").exists(),
        "running": False,
    }
    if sessions:
        session = sessions[-1]
        info["session"] = str(session)
        if (session / "started.json").exists():
            started = read(session / "started.json")
            info["running"] = (
                not (session / "exit.json").exists()
                and started["process_start"] is not None
                and process_start(started["pid"]) == started["process_start"]
            )
        beats = sorted((session / "heartbeats").glob("*.json"))
        if beats:
            info["last_heartbeat"] = read(beats[-1])
        if (session / "exit.json").exists():
            info["exit"] = read(session / "exit.json")
    return info


@contextmanager
def use_collector(collector: Callable[[Path, str, dict, str], dict]) -> Iterator[None]:
    """Temporarily route immutable contract calls through a replay/resume collector."""
    from eval.dedup.runtime import contract

    original = contract.collect
    contract.collect = collector
    try:
        yield
    finally:
        contract.collect = original
