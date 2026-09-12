# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fresh, resumable exploratory 20k collection using the frozen Exp5 runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import exp1_conflict_evidence_fix as runtime
from eval.dedup.analysis import exp1_rate_limit_continuation as limits
from eval.dedup.analysis import exp1_reproduction_recovery as recovery
from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V2, _semantic_diff_packet, assert_blind_payload
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

VERSION = "v0.6.2.33-exp5"
HERE = Path(__file__).resolve()
CHECKPOINT = Path("/raid/hfang/ihb/runs/v06233-exp1-conflict-evidence-fix1-check-v1")
PRIOR = Path("/raid/hfang/ihb/runs/v06233-exp1-fresh-full1000-v1")
PAYLOAD_SOURCE = Path("/raid/hfang/ihb/runs/v0.6.1")
SOURCE = Path("/raid/hfang/dedup_eval_runs/dedup-full-20260813T220949Z-d4c37bb483/v0_run")
read = recovery.read


def upgrade_payload(payload: dict) -> dict:
    """Reuse visible text/windows, adding the existing deterministic Exp1 span packet."""
    result = deepcopy(payload)
    result["payload_schema_version"] = VISIBLE_PAYLOAD_V2
    result["semantic_diff_evidence"] = _semantic_diff_packet(
        result["document_a"]["text"],
        result["document_b"]["text"],
        truncated=result["long_document_evidence"]["truncated"],
    )
    assert_blind_payload(result)
    return result


def verify(root: Path) -> dict:
    manifest = read(root / "manifest.json")
    require(
        manifest["contract_digest"] == sha256_json({k: v for k, v in manifest.items() if k != "contract_digest"}),
        "EXP5_MANIFEST",
        "frozen manifest digest",
    )
    for paths in (manifest["sources"], manifest["artifacts"]):
        for path, digest in paths.items():
            recovery.intact(Path(path).is_file() and sha256_file(Path(path)) == digest)
    return manifest


def prepare(root: Path) -> dict:
    import pyarrow.parquet as pq

    require(not root.exists(), "EXP5_FRESH_ROOT", "new run root; no cross-run answer reuse")
    checkpoint = read(CHECKPOINT / "manifest.json")
    sources = {p: digest for p, digest in checkpoint["sources"].items() if Path(p).is_relative_to(HERE.parents[3])}
    for path, digest in sources.items():
        recovery.intact(sha256_file(Path(path)) == digest)
    extra = [
        HERE,
        HERE.with_suffix(".md"),
        HERE.parents[3] / "tests/eval/dedup/test_exp5_full20k.py",
        CHECKPOINT / "manifest.json",
        CHECKPOINT / "complete.json",
        PRIOR / "panel_private.json",
        PRIOR / "manifest.json",
        PAYLOAD_SOURCE / "run_manifest.json",
        PAYLOAD_SOURCE / "data/judge_payloads.jsonl",
        SOURCE / "data/candidate_pairs.parquet",
        SOURCE / "data/pair_provenance.parquet",
        SOURCE / "data/judge_results.jsonl",
        SOURCE / "logs/judge_errors.jsonl",
        SOURCE / "data/document_outcomes.parquet",
        SOURCE / "manifests/sut_run_manifest.json",
        Path(limits.__file__),
        Path(recovery.__file__),
        Path(sys.modules[limits.RateLimitRelay.__module__].__file__),
    ]
    runtime.old.main_toolchain()
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and name.startswith(
            ("eval.", "data_designer.engine.models", "data_designer.engine.column_generators.utils.prompt_renderer")
        ):
            extra.append(Path(path).resolve())
    sources.update({str(p): sha256_file(p) for p in extra})
    expected = {r["canonical_pair_id"] for r in pq.read_table(SOURCE / "data/candidate_pairs.parquet").to_pylist()}
    tracks: dict[str, set] = {}
    for row in pq.read_table(
        SOURCE / "data/pair_provenance.parquet", columns=["canonical_pair_id", "track"]
    ).to_pylist():
        tracks.setdefault(row["canonical_pair_id"], set()).add(row["track"])
    require(
        len(expected) == 20000
        and set(tracks) == expected
        and all(len(v) == 1 for v in tracks.values())
        and Counter(next(iter(v)) for v in tracks.values()) == {"5a": 10000, "5b": 10000},
        "EXP5_POPULATION",
        "the original 20k population and 10k/10k tracks",
    )
    prior = {r["canonical_pair_id"]: r for r in read(PRIOR / "panel_private.json")}
    root.mkdir(mode=0o700, parents=True)
    artifacts, rows, seen, matched = {}, [], set(), 0
    with (PAYLOAD_SOURCE / "data/judge_payloads.jsonl").open() as stream:
        for position, line in enumerate(stream, 1):
            source = json.loads(line)
            key = source["canonical_pair_id"]
            recovery.intact(key in expected and key not in seen and Path(key).name == key)
            seen.add(key)
            payload = upgrade_payload(source["payload"])
            if key in prior:
                recovery.intact(payload == prior[key]["payload"])
                matched += 1
            row = {
                "canonical_pair_id": key,
                "review_id": prior[key]["review_id"] if key in prior else f"P{position:05d}",
            }
            rows.append(row)
            request = runtime.old.body(runtime.old.main_messages(payload))
            values = {
                "inputs": {**row, "payload": payload},
                "main_requests": {"body": request, "request_sha256": sha256_json(request)},
            }
            for folder, value in values.items():
                path = root / folder / (key + ".json")
                write_json_atomic(path, value)
                artifacts[str(path)] = sha256_file(path)
            if position % 1000 == 0:
                print(json.dumps({"prepared": position, "development_payload_matches": matched}), flush=True)
    recovery.intact(seen == expected and matched == len(prior) == 1000)
    write_json_atomic(root / "panel_index.json", rows)
    write_text_atomic(root / "protocol.md", HERE.with_suffix(".md").read_text())
    for path in (root / "panel_index.json", root / "protocol.md"):
        artifacts[str(path)] = sha256_file(path)
    original = read(PRIOR / "manifest.json")
    manifest = {
        "version": VERSION,
        "runtime_version": runtime.VERSION,
        "mode": "USER_AUTHORIZED_EXPLORATORY_FULL20K_NOT_RELEASE",
        "population": len(rows),
        "at_utc": recovery.now(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_run": str(SOURCE),
        "payload_source": str(PAYLOAD_SOURCE),
        "checkpoint": str(CHECKPOINT),
        "development_payload_matches": matched,
        "old_answers_reused": False,
        "independent_holdout_passed": False,
        "release_eligible": False,
        "minhash_diagnostics": "UNAVAILABLE_MISSING_CONTRACT",
        "model": original["model"],
        "endpoint": original["endpoint"],
        "generation": runtime.old.GENERATION,
        "workers": 2,
        "min_interval_seconds": 2,
        "max_external_attempts": 120000,
        "sources": sources,
        "artifacts": artifacts,
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {k: v for k, v in manifest.items() if k not in {"sources", "artifacts"}}


def execute(root: Path, row: dict, endpoint: str, render: Callable) -> dict:
    key = row["canonical_pair_id"]
    return runtime.execute_case(
        root, read(root / "inputs" / (key + ".json")), endpoint, read(root / "main_requests" / (key + ".json")), render
    )


def audit(root: Path) -> dict:
    manifest = verify(root)
    rows, seen = read(root / "panel_index.json"), set()
    recovery.intact(len(rows) == manifest["population"])
    recovery.intact({p.stem for p in (root / "results").glob("*.json")} == {r["canonical_pair_id"] for r in rows})

    def replay(directory: Path, key: str, body: dict, endpoint: str) -> dict:
        recovery.intact(directory == root and key not in seen and endpoint == "offline://exp5")
        seen.add(key)
        request, receipt = (read(root / folder / (key + ".json")) for folder in ("requests", "responses"))
        recovery.intact(
            request == {"body": body, "request_sha256": sha256_json(body)}
            and receipt["request_sha256"] == request["request_sha256"]
        )
        return receipt

    statuses, repairs, corrections = Counter(), 0, 0
    render = runtime.old.coverage_renderer()
    with recovery.use_collector(replay):
        for row in rows:
            result = execute(root, row, "offline://exp5", render)
            statuses[result["status"]] += 1
            repairs += int("coverage_evidence_repair" in result)
            corrections += int(any(a["requests"] > 1 for a in result["main_attempts"]))
    for folder in ("requests", "responses"):
        recovery.intact({p.stem for p in (root / folder).glob("*.json")} == seen)
    completion = {
        "version": VERSION,
        "population": len(rows),
        "statuses": dict(statuses),
        "saved_calls_replayed": len(seen),
        "evidence_repair_pairs": repairs,
        "main_format_correction_pairs": corrections,
        "fresh_full20k": len(rows) == 20000,
        "release_eligible": False,
        "contract_digest": manifest["contract_digest"],
        "artifacts": {
            str(p): sha256_file(p)
            for folder in ("requests", "responses", "results")
            for p in (root / folder).glob("*.json")
        },
    }
    write_json_atomic(root / "complete.json", completion)
    return {k: v for k, v in completion.items() if k != "artifacts"}


def run(root: Path, env_file: Path, session: Path) -> None:
    from eval.dedup.cli import _load_repository_env

    with recovery.exclusive(root):
        manifest = verify(root)
        require(not (root / "complete.json").exists(), "EXP5_COMPLETE", "completed run is immutable")
        write_json_atomic(
            session / "started.json",
            {"at_utc": recovery.now(), "pid": os.getpid(), "process_start": recovery.process_start(os.getpid())},
        )
        collector = limits.RateLimitCollector(root, session)
        stopped = threading.Event()

        def heartbeat() -> None:
            sequence = 0
            while not stopped.is_set():
                write_json_atomic(
                    session / "heartbeats" / f"{sequence:06d}.json",
                    {
                        "at_utc": recovery.now(),
                        "completed": len(list((root / "results").glob("*.json"))),
                        "population": manifest["population"],
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
            require(bool(credential), "EXP5_CREDENTIAL", "existing NVIDIA key required")
            used = sum(bool(e.get("external_request")) for e in recovery.events(root / "transport_events.jsonl"))
            budget = manifest["max_external_attempts"] - used
            require(budget > 0, "EXP5_BUDGET", "bounded external request budget")
            profile = recovery.original.TransportProfile(
                min_interval_seconds=manifest["min_interval_seconds"],
                max_in_flight=manifest["workers"],
                max_attempts=2,
                retry_base_seconds=10,
                retry_cap_seconds=40,
                request_deadline_seconds=640,
                max_external_attempts=budget,
            )
            with limits.RateLimitRelay(
                profile=profile,
                logical_model=runtime.old.LOGICAL_MODEL,
                upstream_base_url=manifest["endpoint"],
                upstream_model=manifest["model"],
                upstream_api_key=credential,
                timeout_seconds=600,
                expected_generation_parameters=runtime.old.GENERATION,
            ) as relay:
                relay.set_context(
                    recovery.original.RelayContext("exp5-" + session.name, 0, root / "transport_events.jsonl")
                )
                rows = iter(
                    r
                    for r in read(root / "panel_index.json")
                    if not (root / "results" / (r["canonical_pair_id"] + ".json")).exists()
                )
                render = runtime.old.coverage_renderer()
                with recovery.use_collector(collector), ThreadPoolExecutor(max_workers=manifest["workers"]) as pool:
                    active = set()

                    def submit() -> None:
                        row = next(rows, None)
                        if row is not None:
                            active.add(pool.submit(execute, root, row, relay.endpoint, render))

                    for _ in range(manifest["workers"]):
                        submit()
                    while active:
                        ready, active = wait(active, return_when=FIRST_COMPLETED)
                        for future in ready:
                            result = future.result()
                            progress = {
                                "at_utc": recovery.now(),
                                "review_id": result["review_id"],
                                "status": result["status"],
                                "completed": len(list((root / "results").glob("*.json"))),
                            }
                            write_json_atomic(session / "progress" / (result["canonical_pair_id"] + ".json"), progress)
                            print(json.dumps(progress), flush=True)
                        require(
                            relay._circuit_reason is None, "EXP5_CIRCUIT", "stop without consuming remaining pairs"
                        )
                        for _ in ready:
                            submit()
            print(json.dumps(audit(root)), flush=True)
            write_json_atomic(session / "exit.json", {"at_utc": recovery.now(), "status": "COMPLETE", "exit_code": 0})
        except BaseException as exc:
            write_json_atomic(
                session / "exit.json",
                {
                    "at_utc": recovery.now(),
                    "status": "STOPPED",
                    "error_code": recovery.error_code(exc),
                    "exit_code": 1,
                },
            )
            raise
        finally:
            stopped.set()
            thread.join(timeout=5)


def launch(root: Path, env_file: Path) -> dict:
    with recovery.exclusive(root):
        verify(root)
        require(
            not (root / "complete.json").exists() and not recovery.status(root)["running"],
            "EXP5_LAUNCH",
            "no duplicate active or completed run",
        )
        session = root / "recovery/sessions" / recovery.datetime.now(recovery.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        name = "exp5-full20k-" + sha256_json(str(root))[:12]
        write_json_atomic(
            session / "launch.json", {"at_utc": recovery.now(), "tmux_session": name, "credential_persisted": False}
        )
        subprocess.run(  # noqa: S603 - fixed executable and argument vector
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
                "eval.dedup.analysis.exp5_full20k",
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
    parser.add_argument("command", choices=("prepare", "launch", "run", "status", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "run":
        require(args.env_file is not None and args.session is not None, "EXP5_ARGS", "env and session required")
        with (args.session / "run.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                run(root, args.env_file.resolve(), args.session)
            except BaseException as exc:  # noqa: BLE001 - persist only credential-safe error codes
                print(json.dumps({"error_code": recovery.error_code(exc)}), flush=True)
                return 1
        return 0
    if args.command == "prepare":
        value = prepare(root)
    elif args.command == "launch":
        require(args.env_file is not None, "EXP5_ARGS", "existing env file required")
        value = launch(root, args.env_file.resolve())
    elif args.command == "audit":
        value = audit(root)
    else:
        value = {**recovery.status(root), "population": read(root / "manifest.json")["population"]}
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
