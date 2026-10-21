# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disposable, resource-bounded token statistics owned by one Judge session."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from eval.dedup.core.validation import require
from eval.dedup.runtime import state

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

FLOW = "selected_documents_v1"
STOP_TIMEOUT_SECONDS = 5.0
MAX_ROWS = 1_000_000
MAX_SECONDS = 3600


def _write_progress(path: Path, value: dict) -> None:
    """Atomically replace disposable progress, outside the immutable artifact tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".progress-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class BackgroundTokens:
    """Own only a statistics child; cleanup never touches Judge checkpoints."""

    def __init__(self, root: Path, session: Path, manifest: dict) -> None:
        self.root = root
        self.destination = root / "background_tokens" / session.name
        self.enabled = manifest.get("preparation_flow") == FLOW
        self.process: subprocess.Popen | None = None
        self.started = False

    def record(self, status: str, **details) -> None:
        with suppress(OSError):
            _write_progress(self.destination / "supervisor.json", {"status": status, **details})

    def start(self) -> None:
        if not self.enabled or self.started:
            return
        self.started = True
        try:
            self.destination.mkdir(parents=True, exist_ok=False)
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "-1",
                "TOKENIZERS_PARALLELISM": "false",
                "RAYON_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            argv = [
                sys.executable,
                "-u",
                "-m",
                "eval.dedup.runtime.background_tokens",
                "--root",
                str(self.root),
                "--destination",
                str(self.destination),
                "--parent-pid",
                str(os.getpid()),
            ]
            with (self.destination / "worker.log").open("wb") as log:
                self.process = subprocess.Popen(  # noqa: S603 - fixed module and argv, no shell
                    argv,
                    cwd=Path(__file__).resolve().parents[3],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.record("STARTED", pid=self.process.pid, process_start=state.process_start(self.process.pid))
        except Exception as exc:  # noqa: BLE001 - optional statistics never block Judge execution
            self.record("ABANDONED", error_code=type(exc).__name__)

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=min(2.0, max(0.0, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
            self.record("STOPPED", returncode=process.returncode)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.record("ABANDONED", error_code=type(exc).__name__)
        finally:
            self.process = None


@contextmanager
def supervise(root: Path, session: Path, manifest: dict) -> Iterator[BackgroundTokens]:
    task = BackgroundTokens(root, session, manifest)
    previous = {}

    def cancel(signum: int, frame: FrameType | None) -> None:
        task.stop()
        original = previous[signum]
        if callable(original):
            original(signum, frame)
        elif original != signal.SIG_IGN:
            raise SystemExit(128 + signum)

    try:
        if task.enabled and threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, cancel)
        task.start()
        yield task
    finally:
        task.stop()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _bind_parent(parent_pid: int) -> None:
    require(sys.platform == "linux", "BACKGROUND_PLATFORM", "statistics worker requires Linux")
    # Register before importing native tokenizer libraries; also close the parent-death race.
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0, "BACKGROUND_PARENT", "parent-death signal")
    if os.getppid() != parent_pid:
        raise SystemExit(1)
    os.nice(19)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if executable := shutil.which("ionice"):
        subprocess.run(  # noqa: S603 - fixed executable and process ID
            [executable, "-c", "3", "-p", str(os.getpid())],
            check=True,
            timeout=1,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def scan(root: Path, destination: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from eval.dedup.core.config import TokenizerConfig
    from eval.dedup.handoff.corpus import TokenCounter, iter_corpus_batches
    from eval.dedup.pair_construction.outcomes import _length_bucket

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    manifest = state.read(root / "manifest.json")
    expected = manifest["tokenizer"]
    counter = TokenCounter(
        TokenizerConfig(
            expected["kind"],
            expected["model_id"],
            expected["requested_revision"],
            root / "preparation/tokenizer",
        ),
        local_files_only=True,
    )
    require(counter.contract() == expected, "BACKGROUND_TOKENIZER", "same frozen tokenizer required")
    # Native libraries reserve large virtual mappings; bound additional allocation after setup.
    virtual_bytes = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = virtual_bytes + 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (min(limit, hard) if hard != resource.RLIM_INFINITY else limit, hard))
    sample = pq.read_table(root / manifest["sample_documents"], columns=["doc_id"])
    selected = set(sample["doc_id"].to_pylist())
    corpus = state.read(root / "preparation/corpus_manifest.json")
    inventory = state.read(root / "preparation/inputs_manifest.json")
    for shard in corpus["shards"]:
        path = Path(shard["resolved_path"])
        stat = path.stat()
        before = inventory["files"][str(path)]
        require(
            (stat.st_size, stat.st_mtime_ns) == (before["size_bytes"], before["mtime_ns"]),
            "INPUT_CHANGED",
            "background input changed",
        )
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    counted = scanned = parts = 0
    writer = None
    part_rows = 0
    schema = pa.schema(
        [
            ("doc_id", pa.int64()),
            ("char_count", pa.int64()),
            ("token_count", pa.int64()),
            ("length_bucket", pa.string()),
        ]
    )

    def progress(status: str) -> dict:
        value = {
            "status": status,
            "counted": counted,
            "scanned": scanned,
            "parts": parts,
            "reused_sample_documents": len(selected),
            "input_manifest_sha256": manifest["input_manifest_sha256"],
            "tokenizer": expected,
        }
        _write_progress(destination / "worker.json", value)
        return value

    try:
        for batch in iter_corpus_batches(corpus, columns=("text",), batch_size=128):
            scanned += batch.num_rows
            rows = [row for row in batch.to_pylist() if row["doc_id"] not in selected][: MAX_ROWS - counted]
            if rows:
                texts = [row["text"] for row in rows]
                require(all(text is not None for text in texts), "NULL_TEXT", "background text must not be null")
                counts = counter.count_many(texts, batch_size=128)
                table = pa.Table.from_pylist(
                    [
                        {
                            "doc_id": row["doc_id"],
                            "char_count": len(text),
                            "token_count": count,
                            "length_bucket": _length_bucket(count),
                        }
                        for row, text, count in zip(rows, texts, counts, strict=True)
                    ],
                    schema=schema,
                )
                if writer is None:
                    writer = pq.ParquetWriter(destination / f"part-{parts:05d}.tmp", schema, compression="zstd")
                writer.write_table(table)
                part_rows += len(rows)
                counted += len(rows)
                if part_rows >= 8192:
                    writer.close()
                    writer = None
                    (destination / f"part-{parts:05d}.tmp").replace(destination / f"part-{parts:05d}.parquet")
                    parts += 1
                    part_rows = 0
            progress("RUNNING")
            if counted >= MAX_ROWS or time.monotonic() - started >= MAX_SECONDS:
                return progress("LIMIT_REACHED")
            time.sleep(0.02)
        if writer is not None:
            writer.close()
            writer = None
            (destination / f"part-{parts:05d}.tmp").replace(destination / f"part-{parts:05d}.parquet")
            parts += 1
        return progress("COMPLETED")
    finally:
        if writer is not None:
            writer.close()


def status(root: Path, session: Path) -> dict | None:
    destination = root / "background_tokens" / session.name
    if not destination.exists():
        return None
    value = {}
    for name in ("worker", "supervisor"):
        with suppress(OSError, ValueError):
            value[name] = state.read(destination / (name + ".json"))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        _bind_parent(args.parent_pid)
        scan(args.root, args.destination)
        return 0
    except Exception as exc:  # noqa: BLE001 - disposable partial statistics are never required
        with suppress(OSError):
            _write_progress(
                args.destination / "worker.json", {"status": "ABANDONED", "error_code": state.error_code(exc)}
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
