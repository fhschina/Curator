from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from eval.dedup.runtime import background_tokens, preparation, release
from tests.eval.dedup.handoff.test_paths import make_inputs
from tests.eval.dedup.runtime import test_preparation as preparation_tests
from tests.eval.dedup.runtime.test_preparation import offline_collector

small_config = preparation_tests.small_config

if TYPE_CHECKING:
    from collections.abc import Callable

REAL_POPEN = subprocess.Popen
STALL_WORKER = """
import signal, sys, time
from pathlib import Path
from eval.dedup.handoff import corpus
from eval.dedup.runtime import background_tokens
original = corpus.iter_corpus_batches
def blocked(*args, **kwargs):
    for batch in original(*args, **kwargs):
        yield batch
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        destination = Path(sys.argv[sys.argv.index('--destination') + 1])
        (destination / 'blocked').touch()
        time.sleep(60)
corpus.iter_corpus_batches = blocked
raise SystemExit(background_tokens.main())
"""


def stalled_worker(argv: list[str], **kwargs) -> subprocess.Popen:
    if isinstance(argv, list) and argv[2:4] == ["-m", "eval.dedup.runtime.background_tokens"]:
        argv = [*argv[:2], "-c", STALL_WORKER, *argv[4:]]
    return REAL_POPEN(argv, **kwargs)


def wait_until(predicate: Callable[[], bool], *, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            message = "process did not reach the expected state"
            raise AssertionError(message)
        time.sleep(0.02)


@pytest.fixture
def prepared(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, dict]:
    request.getfixturevalue("small_config")
    root = tmp_path / "run"
    preparation.prepare(root, input_paths=make_inputs(tmp_path / "input"))
    return root, release.verify(root)


def test_real_worker_reuses_sample_counts_without_changing_frozen_files(prepared: tuple[Path, dict]) -> None:
    root, manifest = prepared
    task = background_tokens.BackgroundTokens(root, root / "recovery/sessions/real", manifest)
    try:
        task.start()
        assert task.process is not None
        assert task.process.wait(timeout=15) == 0
        progress = release.read(task.destination / "worker.json")
        assert progress["status"] == "COMPLETED"
        selected = set(pq.read_table(root / "sample_documents.parquet", columns=["doc_id"])["doc_id"].to_pylist())
        table = pq.read_table(next(task.destination.glob("*.parquet")))
        ids = table["doc_id"].to_pylist()
        assert set(ids) == set(range(24)) - selected
        assert len(ids) == len(set(ids)) == progress["counted"]
        assert progress["reused_sample_documents"] == len(selected)
        assert table["token_count"].to_pylist() == [9] * len(ids)
        assert progress["tokenizer"] == manifest["tokenizer"]
        release.verify(root)
    finally:
        task.stop()


def test_stuck_real_scan_is_killed_within_limit_and_cleanup_is_idempotent(
    prepared: tuple[Path, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = prepared
    monkeypatch.setattr(
        background_tokens, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": stalled_worker})
    )
    task = background_tokens.BackgroundTokens(root, root / "recovery/sessions/stuck", manifest)
    try:
        task.start()
        worker = task.process
        wait_until(lambda: (task.destination / "blocked").exists())
        assert release.read(task.destination / "worker.json")["status"] == "RUNNING"
        assert list(task.destination.glob("*.tmp"))
        started = time.monotonic()
        task.stop()
        assert time.monotonic() - started < background_tokens.STOP_TIMEOUT_SECONDS
        assert worker.poll() == -signal.SIGKILL
        task.stop()
        release.verify(root)
        assert not (root / "requests").exists()
    finally:
        task.stop()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
def test_parent_cancellation_and_death_leave_no_running_statistics_child(
    prepared: tuple[Path, dict],
    signum: int,
) -> None:
    root, manifest = prepared
    session = root / "recovery/sessions/parent"
    script = f"""
import json, subprocess, time
from pathlib import Path
from eval.dedup.runtime import background_tokens
original = subprocess.Popen
def child(argv, **kwargs):
    return original([*argv[:2], '-c', {STALL_WORKER!r}, *argv[4:]], **kwargs)
background_tokens.subprocess.Popen = child
root, session = Path({str(root)!r}), Path({str(session)!r})
with background_tokens.supervise(root, session, json.loads({json.dumps(manifest)!r})) as task:
    while not (task.destination / 'blocked').exists():
        time.sleep(.02)
    (task.destination / 'parent-ready').write_text(str(task.process.pid))
    time.sleep(60)
"""
    parent = REAL_POPEN([sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    destination = root / "background_tokens/parent"
    child_pid = None
    try:
        wait_until(lambda: (destination / "parent-ready").exists(), timeout=15)
        child_pid = int((destination / "parent-ready").read_text())
        started = time.monotonic()
        parent.send_signal(signum)
        parent.wait(timeout=8)

        def stopped() -> bool:
            try:
                return Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
            except FileNotFoundError:
                return True

        wait_until(stopped, timeout=2)
        assert time.monotonic() - started < 6
        release.verify(root)
        assert not (root / "requests").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


@pytest.mark.parametrize("mode", ["unfinished", "failed", "circuit"])
def test_hub_runtime_completes_audit_while_statistics_are_unfinished_or_failed(
    prepared: tuple[Path, dict],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    root, manifest = prepared
    tasks = []
    original_start = background_tokens.BackgroundTokens.start
    if mode in {"unfinished", "circuit"}:
        monkeypatch.setattr(
            background_tokens, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": stalled_worker})
        )
    else:
        corpus = release.read(root / "preparation/corpus_manifest.json")
        Path(corpus["shards"][0]["resolved_path"]).unlink()

    def start(task: background_tokens.BackgroundTokens) -> None:
        original_start(task)
        tasks.append(task)
        if mode in {"unfinished", "circuit"}:
            wait_until(lambda: (task.destination / "blocked").exists())
        else:
            assert task.process.wait(timeout=10) != 0

    monkeypatch.setattr(background_tokens.BackgroundTokens, "start", start)
    collect = offline_collector(root)
    calls = []

    def collector(*args) -> dict:
        calls.append(args[1])
        return collect(*args)

    collector.heartbeat = dict

    class Relay:
        endpoint = "fixture://judge"
        _circuit_reason = "forced_failure" if mode == "circuit" else None

        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def set_context(self, context: object) -> None:
            pass

        def __exit__(self, *args) -> None:
            assert tasks[-1].process is None

    monkeypatch.setattr(release.transport, "RateLimitRelay", Relay)
    monkeypatch.setattr(release.transport, "RateLimitCollector", lambda *_args: collector)
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture")
    session = root / "recovery/sessions/fixture"
    if mode == "circuit":
        from eval.dedup.core.validation import DedupEvaluationError

        with pytest.raises(DedupEvaluationError, match="V07_CIRCUIT"):
            release.run(root, root / "missing-env", session)
        assert tasks[0].process is None
        assert release.read(session / "exit.json")["status"] == "STOPPED"
        Relay._circuit_reason = None
        session = root / "recovery/sessions/resumed"
    release.run(root, root / "missing-env", session)
    report = release.audit(root)
    assert report["sample_report"]["documents"] == manifest["sample_document_count"]
    assert report["offline_replay_identical"]
    assert len(calls) == len(set(calls)) == manifest["population"]
    before = list(calls)
    release.collect_rows(root, session, release.read(root / "panel_index.json"), Relay(), collector, 2, "RESUME")
    assert calls == before
    release.verify(root)


def test_failure_stops_background_before_waiting_for_other_judge_threads(
    prepared: tuple[Path, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = prepared
    monkeypatch.setattr(
        background_tokens, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": stalled_worker})
    )
    task = background_tokens.BackgroundTokens(root, root / "recovery/sessions/failure", manifest)
    other_started = threading.Event()
    release_other = threading.Event()
    try:
        task.start()
        worker = task.process
        wait_until(lambda: (task.destination / "blocked").exists())

        def execute(_root: Path, row: dict, *_args) -> dict:
            if row["canonical_pair_id"] == "failure":
                assert other_started.wait(timeout=5)
                message = "Judge failure"
                raise RuntimeError(message)
            other_started.set()
            assert release_other.wait(timeout=10)
            return {"canonical_pair_id": "other", "review_id": "other", "status": "VALID"}

        def abort() -> None:
            task.stop()
            assert worker.poll() == -signal.SIGKILL
            release_other.set()

        monkeypatch.setattr(release, "execute", execute)
        with pytest.raises(RuntimeError, match="Judge failure"):
            release.collect_rows(
                root,
                root / "session",
                [{"canonical_pair_id": "failure"}, {"canonical_pair_id": "other"}],
                SimpleNamespace(endpoint="fixture://judge", _circuit_reason=None),
                lambda *_args: {},
                2,
                "FULL20K",
                on_abort=abort,
            )
        assert release_other.is_set()
    finally:
        release_other.set()
        task.stop()
