# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Publish immutable Exp5 snapshots; finalize only after audited 20k completion."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from eval.dedup.analysis import exp5_dashboard as report
from eval.dedup.analysis import exp5_full20k as experiment
from eval.dedup.validation import require, sha256_file, write_json_atomic


def verify_snapshot(root: Path, snapshot: Path) -> dict:
    root, snapshot = root.resolve(), snapshot.resolve()
    require(snapshot.is_relative_to(root / "presentation"), "EXP5_PUBLISH_SCOPE", "run-scoped presentation only")
    record = experiment.read(snapshot / "snapshot.json")
    manifest = experiment.read(root / "manifest.json")
    require(
        record["source_contract_digest"] == manifest["contract_digest"], "EXP5_PUBLISH_CONTRACT", "same frozen run"
    )
    for path, digest in {**record["result_bindings"], **record["files"]}.items():
        require(
            Path(path).resolve().is_relative_to(root) and sha256_file(Path(path)) == digest,
            "EXP5_PUBLISH_BINDING",
            "snapshot and referenced results are unchanged",
        )
    summary = experiment.read(snapshot / "reports/comparison.json")
    require(
        summary["population"] == manifest["population"]
        and summary["collected"] <= summary["population"]
        and summary["complete"] == (not record["preview"]),
        "EXP5_PUBLISH_COUNTS",
        "honest snapshot scope",
    )
    require(
        all(
            (snapshot / "reports" / name).is_file()
            for name in (
                "pair_explorer_exp5.html",
                "comparison.html",
                "comparison.json",
                "v05_exp5_pairs.csv",
                "RESULTS.md",
            )
        ),
        "EXP5_PUBLISH_ARTIFACTS",
        "all promised presentation artifacts",
    )
    if summary["complete"]:
        complete = experiment.read(root / "complete.json")
        require(
            complete["contract_digest"] == manifest["contract_digest"]
            and complete["population"] == summary["collected"] == summary["population"]
            and complete["fresh_full20k"] is True,
            "EXP5_PUBLISH_COMPLETION",
            "audited complete 20k only",
        )
        for path, digest in complete["artifacts"].items():
            require(sha256_file(Path(path)) == digest, "EXP5_PUBLISH_COMPLETION_BINDING", "audited results unchanged")
    return summary


def publish(root: Path, snapshot: Path) -> dict:
    summary = verify_snapshot(root, snapshot)
    current = root / "presentation/current"
    require(not current.exists() or current.is_symlink(), "EXP5_CURRENT_TYPE", "never overwrite a user directory")
    if current.is_symlink():
        require(
            current.resolve().is_relative_to((root / "presentation").resolve()),
            "EXP5_CURRENT_SCOPE",
            "only replace this experiment's existing presentation pointer",
        )
        if current.resolve() == snapshot.resolve():
            return {"snapshot": str(snapshot), "complete": summary["complete"], "collected": summary["collected"]}
    temporary = current.with_name(f".current-{os.getpid()}")
    require(not temporary.exists() and not temporary.is_symlink(), "EXP5_POINTER_TEMP", "unique publication pointer")
    temporary.symlink_to(snapshot.resolve())
    temporary.replace(current)
    experiment.recovery.append_event(
        root / "presentation/publications.jsonl",
        {
            "snapshot": str(snapshot),
            "snapshot_sha256": sha256_file(snapshot / "snapshot.json"),
            "complete": summary["complete"],
            "collected": summary["collected"],
        },
    )
    return {"snapshot": str(snapshot), "complete": summary["complete"], "collected": summary["collected"]}


def watch(root: Path, session: Path, *, preview_every: int = 5000, poll_seconds: float = 30) -> None:
    require(preview_every > 0 and poll_seconds > 0, "EXP5_WATCH_INTERVAL", "positive snapshot and poll intervals")
    root = root.resolve()
    session.mkdir(parents=True, exist_ok=True)
    with (root / "presentation/finalizer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experiment.verify(root)
        sources = {
            str(Path(module.__file__).resolve()): sha256_file(Path(module.__file__))
            for module in (report, report.view, sys.modules[__name__])
        }
        write_json_atomic(
            session / "started.json",
            {
                "at_utc": experiment.recovery.now(),
                "pid": os.getpid(),
                "process_start": experiment.recovery.process_start(os.getpid()),
                "sources": sources,
            },
        )
        try:
            while True:
                for path, digest in sources.items():
                    require(sha256_file(Path(path)) == digest, "EXP5_EXPORTER_CHANGED", "frozen report producer")
                state = experiment.recovery.status(root)
                experiment.recovery.append_event(
                    session / "progress.jsonl", {k: state[k] for k in ("completed", "complete", "running")}
                )
                if state["complete"]:
                    final = root / "presentation/final-v1"
                    if not final.exists():
                        report.build(root, final)
                    result = publish(root, final)
                    write_json_atomic(
                        session / "exit.json",
                        {
                            "at_utc": experiment.recovery.now(),
                            "status": "PUBLISHED_FINAL",
                            **result,
                        },
                    )
                    return
                require(
                    state["running"],
                    "EXP5_COLLECTION_STOPPED",
                    "collector is not live; do not restart or fabricate completion",
                )
                milestone = state["completed"] // preview_every * preview_every
                snapshot = root / "presentation" / f"progress-{milestone:05d}"
                if milestone and not snapshot.exists():
                    report.build(root, snapshot, preview=True)
                    publish(root, snapshot)
                threading.Event().wait(poll_seconds)
        except BaseException as exc:
            write_json_atomic(
                session / "exit.json",
                {
                    "at_utc": experiment.recovery.now(),
                    "status": "STOPPED",
                    "error_code": experiment.recovery.error_code(exc),
                },
            )
            raise


def launch(root: Path) -> dict:
    root = root.resolve()
    name = "exp5-presentation-finalizer"
    require(
        subprocess.run(["tmux", "has-session", "-t", name], capture_output=True, check=False).returncode != 0,  # noqa: S603 - fixed tmux session lookup
        "EXP5_FINALIZER_RUNNING",
        "existing worker must not be duplicated",
    )
    session = (
        root
        / "presentation/finalizer"
        / experiment.recovery.datetime.now(experiment.recovery.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    write_json_atomic(session / "launch.json", {"at_utc": experiment.recovery.now(), "tmux_session": name})
    subprocess.run(  # noqa: S603 - fixed executable and argument vector
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            name,
            "-c",
            str(Path(__file__).resolve().parents[3]),
            sys.executable,
            "-m",
            "eval.dedup.analysis.exp5_presentation_finalize",
            "watch",
            "--root",
            str(root),
            "--session",
            str(session),
        ],
        capture_output=True,
        check=True,
    )
    return {"session": str(session), "tmux_session": name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish", "launch", "watch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    if args.command == "watch":
        require(args.session is not None, "EXP5_WATCH_SESSION", "session path required")
        with (args.session / "run.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                watch(args.root, args.session)
            except BaseException as exc:  # noqa: BLE001 - safe exception codes only
                print(json.dumps({"error_code": experiment.recovery.error_code(exc)}), flush=True)
                return 1
        return 0
    if args.command == "publish":
        require(args.snapshot is not None, "EXP5_PUBLISH_SNAPSHOT", "explicit snapshot required")
        result = publish(args.root.resolve(), args.snapshot.resolve())
    else:
        result = launch(args.root)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
