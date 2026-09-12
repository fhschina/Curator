# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Append-only, same-run recovery around the frozen exp1 semantic runtime."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import requests

from eval.dedup.analysis import exp1_reproduction as original
from eval.dedup.analysis import exp1_reproduction_runtime as runtime
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic

HERE = Path(__file__).resolve()
PROTOCOL = HERE.with_suffix(".md")


def now() -> str:
    return datetime.now(UTC).isoformat()


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def error_code(exc: BaseException) -> str:
    return exc.issue.code if isinstance(exc, DedupEvaluationError) else type(exc).__name__


class CheckpointIntegrityError(BaseException):
    """Never turn recovery integrity violations into Judge retries or scored failures."""


def intact(condition: bool) -> None:
    if not condition:
        raise CheckpointIntegrityError("EXP1_CHECKPOINT_INTEGRITY")


@contextmanager
def exclusive(root: Path) -> Iterator[None]:
    directory = root / "recovery"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("EXP1_RECOVERY_ALREADY_RUNNING") from None
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


def inventory(root: Path) -> dict:
    rows = read(root / "panel_private.json")
    indexed = original._index(rows, "recovery panel")
    completed, missing, partial = [], [], []
    accounted = set()
    requests_by_pair = {pid: [] for pid in indexed}
    for path in (root / "requests").glob("*.json"):
        pid = path.stem.rsplit("-main-", 1)[0] if "-main-" in path.stem else path.stem.rsplit("-", 1)[0]
        intact(pid in indexed)
        requests_by_pair[pid].append(path)
        request = read(path)
        intact(request["request_sha256"] == sha256_json(request["body"]))
        receipt_path = root / "responses" / path.name
        if receipt_path.exists():
            intact(read(receipt_path)["request_sha256"] == request["request_sha256"])
        else:
            missing.append(path.name)
    intact(
        {p.name for p in (root / "responses").glob("*.json")}
        <= {p.name for paths in requests_by_pair.values() for p in paths}
    )
    intact({p.stem for p in (root / "results").glob("*.json")} <= set(indexed))
    for pid, row in indexed.items():
        path = root / "results" / (pid + ".json")
        if not path.exists():
            if requests_by_pair[pid]:
                partial.append(row["review_id"])
            continue
        result = read(path)
        intact(result["canonical_pair_id"] == pid and result["review_id"] == row["review_id"])
        names = {pid + "-" + stage["stage"] + ".json" for stage in result["stages"]}
        intact(len(names) == len(result["stages"]) and names == {p.name for p in requests_by_pair[pid]})
        for stage in result["stages"]:
            receipt_path = root / "responses" / (pid + "-" + stage["stage"] + ".json")
            intact(receipt_path.exists() and sha256_file(receipt_path) == stage["response_sha256"])
        completed.append(pid)
        accounted.update(names)
    return {
        "population": len(rows),
        "completed_pair_ids": completed,
        "partial_review_ids": partial,
        "missing_receipts": sorted(missing),
        "completed_bound_calls": len(accounted),
    }


def prepare(root: Path) -> dict:
    original.reference.verify_freeze(root / "manifest.json")
    require(
        (root / "started.json").exists() and not (root / "complete.json").exists(),
        "EXP1_RECOVERY_STATE",
        "interrupted incomplete run required",
    )
    with exclusive(root):
        require(not (root / "recovery_manifest.json").exists(), "EXP1_RECOVERY_FROZEN", "already prepared")
        state = inventory(root)
        paths = [HERE, PROTOCOL, HERE.parents[3] / "tests/eval/dedup/test_exp1_reproduction_recovery.py"]
        artifacts = [
            p
            for folder in ("requests", "responses", "results", "progress_snapshots")
            for p in (root / folder).glob("*.json")
        ]
        old_events = (root / "transport_events.jsonl").read_bytes()
        manifest = {
            "schema_version": "dedup-exp1-recovery-v1",
            "at_utc": now(),
            "original_contract_digest": read(root / "manifest.json")["contract_digest"],
            "original_manifest_sha256": sha256_file(root / "manifest.json"),
            "semantic_changes": False,
            "historical_cache_reused": False,
            "same_run_checkpoint_continuation": True,
            "initial_inventory": state,
            "original_transport_prefix_bytes": len(old_events),
            "original_transport_prefix_sha256": hashlib.sha256(old_events).hexdigest(),
            "unknown_original_inflight_attempts_upper_bound": 2 * len(state["missing_receipts"]),
            "sources": {str(p): sha256_file(p) for p in paths},
            "artifacts": {str(p): sha256_file(p) for p in artifacts},
        }
        manifest["contract_digest"] = sha256_json(manifest)
        write_json_atomic(root / "recovery_manifest.json", manifest)
        return {"completed": len(state["completed_pair_ids"]), **state}


def verify(root: Path) -> dict:
    original.reference.verify_freeze(root / "manifest.json")
    original.reference.verify_freeze(root / "recovery_manifest.json")
    manifest = read(root / "recovery_manifest.json")
    intact(sha256_file(root / "manifest.json") == manifest["original_manifest_sha256"])
    with (root / "transport_events.jsonl").open("rb") as stream:
        prefix = stream.read(manifest["original_transport_prefix_bytes"])
    intact(hashlib.sha256(prefix).hexdigest() == manifest["original_transport_prefix_sha256"])
    return manifest


class Collector:
    def __init__(self, root: Path, session: Path):
        self.root, self.session = root.resolve(), session
        self.lock = threading.Lock()
        self.active: dict[str, str] = {}
        self.replayed = self.submitted = self.retransmitted = 0

    def __call__(self, root: Path, key: str, request: dict, endpoint: str) -> dict:
        intact(root.resolve() == self.root and Path(key).name == key)
        request_path, receipt_path = (root / d / (key + ".json") for d in ("requests", "responses"))
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
        # Match the frozen collect() transport and receipt shape, except for same-run resume.
        receipt = {"request_sha256": saved["request_sha256"], "status": "FAILURE"}
        try:
            response = requests.post(endpoint + "/chat/completions", json=request, timeout=660)
            receipt["http_status"] = response.status_code
            require(response.status_code == 200, "EXP1_HTTP", "request failed; retain the failure")
            receipt.update(status="RECEIVED", raw_response=response.json())
        except Exception as exc:  # noqa: BLE001 - do not log credential-bearing exception messages
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


@contextmanager
def use_collector(collector: Collector) -> Iterator[None]:
    previous = runtime.collect
    runtime.collect = collector
    try:
        yield
    finally:
        runtime.collect = previous


def collect_pending(root: Path, session: Path, relay: original.PacedRelayV2, collector: Collector) -> None:
    state = inventory(root)
    done = set(state["completed_pair_ids"])
    rows = read(root / "panel_private.json")
    pending = iter(row for row in rows if row["canonical_pair_id"] not in done)
    frozen = original._index(read(root / "main_requests.json"), "recovery initial requests")
    render = runtime.coverage_renderer()
    with use_collector(collector), ThreadPoolExecutor(max_workers=2) as pool:
        active = set()

        def submit() -> None:
            row = next(pending, None)
            if row is not None:
                active.add(
                    pool.submit(
                        runtime.execute_case, root, row, relay.endpoint, frozen[row["canonical_pair_id"]], render
                    )
                )

        submit()
        submit()
        while active:
            ready, active = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                result = future.result()
                done.add(result["canonical_pair_id"])
                progress = {
                    "completed": len(done),
                    "population": len(rows),
                    "review_id": result["review_id"],
                    "status": result["status"],
                }
                write_json_atomic(session / "progress" / f"{len(done):04d}.json", progress)
                print(json.dumps(progress), flush=True)
            require(relay._circuit_reason is None, "EXP1_RECOVERY_CIRCUIT", "stop; do not consume remaining pairs")
            for _ in ready:
                submit()


def run(root: Path, env_file: Path, session: Path) -> None:
    from eval.dedup.cli import _load_repository_env

    with exclusive(root):
        manifest = verify(root)
        require(not (root / "complete.json").exists(), "EXP1_RECOVERY_COMPLETE", "already complete")
        require(not (session / "started.json").exists(), "EXP1_RECOVERY_SESSION", "fresh execution session required")
        write_json_atomic(
            session / "started.json",
            {
                "at_utc": now(),
                "pid": os.getpid(),
                "process_start": process_start(os.getpid()),
                "recovery_digest": manifest["contract_digest"],
            },
        )
        collector = Collector(root, session)
        stopped = threading.Event()

        def heartbeat() -> None:
            sequence = 0
            while not stopped.is_set():
                write_json_atomic(
                    session / "heartbeats" / f"{sequence:06d}.json",
                    {
                        "at_utc": now(),
                        "pid": os.getpid(),
                        "completed": len(list((root / "results").glob("*.json"))),
                        **collector.heartbeat(),
                    },
                )
                sequence += 1
                stopped.wait(30)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            _load_repository_env(env_file)
            credential = os.environ.get("NVIDIA_API_KEY", "").strip()
            require(bool(credential), "EXP1_CREDENTIAL", "existing NVIDIA credential required")
            runtime.main_toolchain()
            frozen = read(root / "manifest.json")
            observed = sum(e.get("external_request", False) for e in events(root / "transport_events.jsonl"))
            sessions = sorted((root / "recovery/sessions").glob("*"))
            submitted_before = sum(
                e["event"] == "SUBMIT" for s in sessions for e in events(s / "collection_events.jsonl")
            )
            observed_before = sum(
                e.get("external_request", False) and e.get("block_id", "").startswith("exp1-recovery-")
                for e in events(root / "transport_events.jsonl")
            )
            # A killed HTTP call may have reached the provider without leaving a completion event.
            reserve = manifest["unknown_original_inflight_attempts_upper_bound"] + max(
                0, 2 * submitted_before - observed_before
            )
            budget = frozen["max_external_attempts"] - observed - reserve
            require(budget > 0, "EXP1_RECOVERY_BUDGET", "global attempt budget exhausted")
            profile = original.TransportProfile(
                min_interval_seconds=2,
                max_in_flight=2,
                max_attempts=2,
                retry_base_seconds=10,
                retry_cap_seconds=40,
                request_deadline_seconds=640,
                max_external_attempts=budget,
            )
            with original.PacedRelayV2(
                profile=profile,
                logical_model=runtime.LOGICAL_MODEL,
                upstream_base_url=frozen["endpoint"],
                upstream_model=frozen["model"],
                upstream_api_key=credential,
                timeout_seconds=600,
                expected_generation_parameters=runtime.GENERATION,
            ) as relay:
                relay.set_context(
                    original.RelayContext("exp1-recovery-" + session.name, 0, root / "transport_events.jsonl")
                )
                collect_pending(root, session, relay, collector)
            verify(root)
            recovery_summary = {
                "at_utc": now(),
                "original_results_preserved": len(manifest["initial_inventory"]["completed_pair_ids"]),
                "semantic_changes": False,
                "same_run_checkpoint_continuation": True,
                "continuous_uninterrupted_execution": False,
                "original_inflight_unknown_attempts_upper_bound": manifest[
                    "unknown_original_inflight_attempts_upper_bound"
                ],
                "accounting_note": "Original export counts observed HTTP completion events only; interrupted submissions may have reached the provider. Recovery retransmissions are separate from semantic retries.",
                "sessions": [str(s) for s in sessions],
                "recovery_artifacts": {str(p): sha256_file(p) for s in sessions for p in s.rglob("*.jsonl")},
                "recovery_retransmissions": sum(
                    e.get("interrupted_request_retransmission", False)
                    for s in sessions
                    for e in events(s / "collection_events.jsonl")
                ),
            }
            if (root / "recovery_final.json").exists():
                previous = read(root / "recovery_final.json")
                intact(all(sha256_file(p) == digest for p, digest in previous["recovery_artifacts"].items()))
            else:
                write_json_atomic(root / "recovery_final.json", recovery_summary)
            original.export(root)
            write_json_atomic(session / "exit.json", {"at_utc": now(), "status": "COMPLETE", "exit_code": 0})
        except BaseException as exc:
            write_json_atomic(
                session / "exit.json",
                {"at_utc": now(), "status": "STOPPED", "exit_code": 1, "error_code": error_code(exc)},
            )
            raise
        finally:
            stopped.set()
            thread.join(timeout=5)


def process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def status(root: Path) -> dict:
    sessions = sorted((root / "recovery/sessions").glob("*"))
    info = {
        "completed": len(list((root / "results").glob("*.json"))),
        "population": 1000,
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


def launch(root: Path, env_file: Path) -> dict:
    with exclusive(root):
        verify(root)
        require(
            not status(root)["running"] and not (root / "complete.json").exists(),
            "EXP1_RECOVERY_LAUNCH",
            "must not duplicate an active or complete run",
        )
        session = root / "recovery/sessions" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        name = "exp1-recovery-" + sha256_json(str(root))[:12]
        write_json_atomic(
            session / "launch.json", {"at_utc": now(), "tmux_session": name, "credential_persisted": False}
        )
        subprocess.run(  # noqa: S603 - fixed executable and argv, never shell interpolation
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
                "eval.dedup.analysis.exp1_reproduction_recovery",
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
        return {"session": str(session), "tmux_session": name, "started": "VERIFY_WITH_STATUS"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "run", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "run":
        require(args.env_file is not None and args.session is not None, "EXP1_RECOVERY_ARGS", "paths required")
        with (args.session / "run.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                run(root, args.env_file.resolve(), args.session)
            except BaseException as exc:  # noqa: BLE001 - persist only credential-safe error codes
                print(json.dumps({"error_code": error_code(exc)}), flush=True)
                return 1
        return 0
    if args.command == "prepare":
        result = prepare(root)
    elif args.command == "launch":
        require(args.env_file is not None, "EXP1_RECOVERY_ARGS", "existing env file required")
        result = launch(root, args.env_file.resolve())
    else:
        result = status(root)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
