# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fresh Exp6 repetition-fix2 collection with a smoke gate and nohup continuation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from eval.dedup.analysis import exp1_rate_limit_continuation as limits
from eval.dedup.analysis import exp1_reproduction_recovery as recovery
from eval.dedup.analysis import exp5_full20k as baseline
from eval.dedup.analysis import exp6_repetition_fix2 as runtime
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

HERE = Path(__file__).resolve()
SOURCE = Path("/raid/hfang/ihb/runs/v06233-exp5-full20k-v1")
CHECKPOINT = Path("/raid/hfang/ihb/runs/v06233-exp6-main-repetition-fix2-check-v1")
VERSION = runtime.VERSION
POPULATION = 20000
SMOKE_SIZE = 24
REQUIRED_SMOKE_STAGES = {"main", "coverage", "subject", "verifier"}
KNOWN_CASES = ("P01329", "P09549", "P16061", "P06727", "P14187")
read = recovery.read


def select_smoke(rows: list[dict], controls: list[str], history: dict[str, dict]) -> list[dict]:
    """Use frozen execution mechanisms, never reference labels or new predictions."""
    by_review = {r["review_id"]: r for r in rows}
    by_key = {r["canonical_pair_id"]: r for r in rows}
    selected = [{**by_review[name], "sampling_reason": "historical_failure_or_repetition"} for name in KNOWN_CASES]
    buckets = defaultdict(list)
    for key in controls:
        saved = history[key]
        stages = {s["stage"] for s in saved["stages"]}
        kind = (
            "native_correction"
            if any(a["requests"] > 1 for a in saved["main_attempts"])
            else "verifier"
            if "verifier" in stages
            else "subject"
            if "subject" in stages
            else "coverage"
            if "coverage" in stages
            else saved["public"]["relation_type"]
        )
        buckets[kind].append(key)
    for kind, keys in sorted(buckets.items()):
        selected.extend({**by_key[key], "sampling_reason": kind} for key in keys[:2])
    seen = {r["canonical_pair_id"] for r in selected}
    for key in controls:
        if len(selected) >= SMOKE_SIZE:
            break
        if key not in seen:
            selected.append({**by_key[key], "sampling_reason": "normal_path_protection"})
            seen.add(key)
    require(len(selected) == len(seen) == SMOKE_SIZE, "EXP6_SMOKE_PANEL", "unique frozen smoke population")
    return selected


def verify(root: Path) -> dict:
    manifest = baseline.verify(root)
    require(
        manifest["version"] == manifest["runtime_version"] == runtime.VERSION
        and manifest["generation"] == runtime.old.GENERATION
        and manifest["old_answers_reused"] is False,
        "EXP6_RUNTIME_BINDING",
        "the frozen repetition-fix2 runtime and fresh answers are required",
    )
    return manifest


def prepare(root: Path) -> dict:
    require(not root.exists(), "EXP6_FRESH_ROOT", "new run root; no historical answers may be copied")
    original = baseline.verify(SOURCE)
    checkpoint = baseline.verify(CHECKPOINT)
    recovery.intact(checkpoint["runtime_version"] == runtime.VERSION)
    rows = read(SOURCE / "panel_index.json")
    recovery.intact(len(rows) == len({r["canonical_pair_id"] for r in rows}) == POPULATION)
    controls = checkpoint["offline_control_ids"]
    history = {key: read(SOURCE / "results" / (key + ".json")) for key in controls}
    history_bindings = read(SOURCE / "complete.json")["artifacts"]
    for key in controls:
        path = SOURCE / "results" / (key + ".json")
        recovery.intact(sha256_file(path) == history_bindings[str(path)])
    smoke = select_smoke(rows, controls, history)
    sources = {**original["sources"], **checkpoint["sources"]}
    extras = [HERE, HERE.with_suffix(".md"), HERE.parents[3] / "tests/eval/dedup/test_exp6_full20k.py"]
    sources.update({str(p): sha256_file(p) for p in extras})
    root.mkdir(parents=True, mode=0o700)
    artifacts = {}
    for index, row in enumerate(rows, 1):
        key = row["canonical_pair_id"]
        recovery.intact(Path(key).name == key)
        for folder in ("inputs", "main_requests"):
            source = SOURCE / folder / (key + ".json")
            target = root / folder / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            recovery.intact(sha256_file(target) == original["artifacts"][str(source)])
            artifacts[str(target)] = sha256_file(target)
        payload = read(root / "inputs" / (key + ".json"))
        frozen = read(root / "main_requests" / (key + ".json"))
        recovery.intact(all(payload[k] == v for k, v in row.items()))
        recovery.intact(frozen["request_sha256"] == sha256_json(frozen["body"]))
        recovery.intact(frozen["body"] == runtime.old.body(frozen["body"]["messages"]))
        if index % 1000 == 0:
            print(json.dumps({"prepared": index}), flush=True)
    write_json_atomic(root / "panel_index.json", rows)
    write_json_atomic(root / "smoke_panel.json", smoke)
    write_text_atomic(root / "protocol.md", HERE.with_suffix(".md").read_text())
    for path in (
        root / "panel_index.json",
        root / "smoke_panel.json",
        root / "protocol.md",
        SOURCE / "manifest.json",
        SOURCE / "complete.json",
        CHECKPOINT / "manifest.json",
        CHECKPOINT / "completion.json",
    ):
        artifacts[str(path)] = sha256_file(path)
    manifest = {
        "version": VERSION,
        "runtime_version": runtime.VERSION,
        "mode": "USER_AUTHORIZED_FRESH_FULL20K_WITH_SMOKE_GATE_NOT_RELEASE",
        "at_utc": recovery.now(),
        "population": len(rows),
        "smoke_size": len(smoke),
        "required_smoke_stages": sorted(REQUIRED_SMOKE_STAGES),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_run": str(SOURCE),
        "source_contract_digest": original["contract_digest"],
        "checkpoint": str(CHECKPOINT),
        "checkpoint_contract_digest": checkpoint["contract_digest"],
        "development_payload_matches": original["development_payload_matches"],
        "old_answers_reused": False,
        "smoke_included_in_population": True,
        "independent_holdout_passed": False,
        "release_eligible": False,
        "minhash_diagnostics": "UNAVAILABLE_MISSING_CONTRACT",
        "model": original["model"],
        "endpoint": original["endpoint"],
        "generation": runtime.old.GENERATION,
        "workers": original["workers"],
        "min_interval_seconds": original["min_interval_seconds"],
        "max_external_attempts": original["max_external_attempts"],
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


def replay_results(root: Path, rows: list[dict]) -> dict:
    """Replay into a disposable result directory; historical results are never rewritten."""
    statuses, stages = Counter(), Counter()
    repairs, corrections, unresolved = Counter(), 0, 0
    seen = set()
    bindings = {}
    render = runtime.old.coverage_renderer()
    with tempfile.TemporaryDirectory(prefix="exp6-replay-") as directory:
        target = Path(directory)
        for folder in ("inputs", "main_requests", "responses"):
            (target / folder).symlink_to(root / folder, target_is_directory=True)

        def replay(path: Path, key: str, body: dict, endpoint: str) -> dict:
            recovery.intact(path == target and key not in seen and endpoint == "offline://exp6")
            request, receipt = (read(root / folder / (key + ".json")) for folder in ("requests", "responses"))
            recovery.intact(request == {"body": body, "request_sha256": sha256_json(body)})
            recovery.intact(receipt["request_sha256"] == request["request_sha256"])
            seen.add(key)
            return receipt

        with recovery.use_collector(replay):
            for row in rows:
                key = row["canonical_pair_id"]
                path = root / "results" / (key + ".json")
                recovery.intact(path.is_file())
                original = read(path)
                recovery.intact(original["version"] == VERSION)
                before = set(seen)
                expected = {key + "-" + stage["stage"] for stage in original["stages"]}
                recovery.intact(len(expected) == len(original["stages"]))
                for stage in original["stages"]:
                    receipt_path = root / "responses" / (key + "-" + stage["stage"] + ".json")
                    recovery.intact(sha256_file(receipt_path) == stage["response_sha256"])
                result = execute(target, row, "offline://exp6", render)
                recovery.intact(result == original and seen - before == expected)
                statuses[result["status"]] += 1
                unresolved += result["public"]["same_duplicate_group"] == "UNRESOLVED"
                stages.update("main" if s["stage"].startswith("main-") else s["stage"] for s in result["stages"])
                corrections += sum(a["requests"] for a in result["main_attempts"]) > 1
                for name in ("coverage_evidence_repair", "coverage_anchor_id_repair", "main_repetition_repair"):
                    repairs[name] += name in result
                bindings[str(path)] = sha256_file(path)
    for key in seen:
        for folder in ("requests", "responses"):
            path = root / folder / (key + ".json")
            bindings[str(path)] = sha256_file(path)
    return {
        "population": len(rows),
        "statuses": dict(statuses),
        "stages": dict(stages),
        "semantic_unresolved": unresolved,
        "main_retry_pairs": corrections,
        "repairs": dict(repairs),
        "saved_calls_replayed": len(seen),
        "replayed_keys": sorted(seen),
        "artifacts": bindings,
        "offline_replay_identical": True,
    }


def check_smoke(root: Path, manifest: dict) -> dict:
    rows = read(root / "smoke_panel.json")
    report = replay_results(root, rows)
    recovery.intact(len(rows) == manifest["smoke_size"])
    missing = sorted(set(manifest["required_smoke_stages"]) - report["stages"].keys())
    passed = report["statuses"] == {"VALID": len(rows)} and not missing
    value = {
        **report,
        "passed": passed,
        "missing_required_stages": missing,
        "contract_digest": manifest["contract_digest"],
        "gate_kind": "ENGINEERING_NOT_SEMANTIC_ACCURACY",
    }
    path = root / "smoke_complete.json"
    if path.exists():
        recovery.intact(read(path) == value)
    else:
        write_json_atomic(path, value)
    require(passed, "EXP6_SMOKE_FAILED", "stop before the remaining population; preserve all smoke results")
    return {k: v for k, v in value.items() if k not in {"artifacts", "replayed_keys"}}


def audit(root: Path) -> dict:
    manifest = verify(root)
    rows = read(root / "panel_index.json")
    recovery.intact(len(rows) == manifest["population"])
    recovery.intact({p.stem for p in (root / "results").glob("*.json")} == {r["canonical_pair_id"] for r in rows})
    check_smoke(root, manifest)
    report = replay_results(root, rows)
    for folder in ("requests", "responses"):
        recovery.intact({p.stem for p in (root / folder).glob("*.json")} == set(report["replayed_keys"]))
    report.pop("replayed_keys")
    transport = [e for e in recovery.events(root / "transport_events.jsonl") if e.get("external_request")]
    value = {
        **report,
        "version": VERSION,
        "contract_digest": manifest["contract_digest"],
        "fresh_full20k": len(rows) == POPULATION,
        "release_eligible": False,
        "external_attempts": len(transport),
        "http_statuses": dict(Counter(e.get("http_status") for e in transport)),
    }
    path = root / "complete.json"
    if path.exists():
        recovery.intact(read(path) == json.loads(json.dumps(value)))
    else:
        write_json_atomic(path, value)
    return {k: v for k, v in value.items() if k != "artifacts"}


def collect_rows(
    root: Path,
    session: Path,
    rows: list[dict],
    relay: limits.RateLimitRelay,
    collector: limits.RateLimitCollector,
    workers: int,
    phase: str,
) -> None:
    pending = iter(r for r in rows if not (root / "results" / (r["canonical_pair_id"] + ".json")).exists())
    render = runtime.old.coverage_renderer()
    with recovery.use_collector(collector), ThreadPoolExecutor(max_workers=workers) as pool:
        active = set()

        def submit() -> None:
            row = next(pending, None)
            if row is not None:
                active.add(pool.submit(execute, root, row, relay.endpoint, render))

        for _ in range(workers):
            submit()
        while active:
            ready, active = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                result = future.result()
                progress = {
                    "at_utc": recovery.now(),
                    "phase": phase,
                    "review_id": result["review_id"],
                    "status": result["status"],
                    "completed": len(list((root / "results").glob("*.json"))),
                }
                write_json_atomic(session / "progress" / (result["canonical_pair_id"] + ".json"), progress)
                print(json.dumps(progress), flush=True)
            require(relay._circuit_reason is None, "EXP6_CIRCUIT", "stop without consuming remaining pairs")
            for _ in ready:
                submit()


def run(root: Path, env_file: Path, session: Path, *, smoke_only: bool = False) -> None:
    from eval.dedup.cli import _load_repository_env

    with recovery.exclusive(root):
        require(not (session / "started.json").exists(), "EXP6_SESSION", "fresh execution session required")
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
                        "population": POPULATION,
                        **collector.heartbeat(),
                    },
                )
                sequence += 1
                stopped.wait(30)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            manifest = verify(root)
            require(not (root / "complete.json").exists(), "EXP6_COMPLETE", "completed run is immutable")
            _load_repository_env(env_file)
            credential = os.environ.get("NVIDIA_API_KEY", "").strip()
            require(bool(credential), "EXP6_CREDENTIAL", "existing NVIDIA key required")
            used = sum(bool(e.get("external_request")) for e in recovery.events(root / "transport_events.jsonl"))
            budget = manifest["max_external_attempts"] - used
            require(budget > 0, "EXP6_BUDGET", "bounded external request budget")
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
                    recovery.original.RelayContext("exp6-" + session.name, 0, root / "transport_events.jsonl")
                )
                smoke = read(root / "smoke_panel.json")
                smoke_ids = {r["canonical_pair_id"] for r in smoke}
                if not (root / "smoke_complete.json").exists():
                    recovery.intact({p.stem for p in (root / "results").glob("*.json")} <= smoke_ids)
                    collect_rows(root, session, smoke, relay, collector, manifest["workers"], "SMOKE")
                print(json.dumps({"event": "SMOKE_GATE_PASSED", **check_smoke(root, manifest)}), flush=True)
                if not smoke_only:
                    print(json.dumps({"event": "FULL20K_CONTINUATION", "runtime_version": VERSION}), flush=True)
                    collect_rows(
                        root,
                        session,
                        read(root / "panel_index.json"),
                        relay,
                        collector,
                        manifest["workers"],
                        "FULL20K",
                    )
            if not smoke_only:
                print(json.dumps(audit(root)), flush=True)
            write_json_atomic(
                session / "exit.json",
                {"at_utc": recovery.now(), "status": "SMOKE_PASSED" if smoke_only else "COMPLETE", "exit_code": 0},
            )
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


def status(root: Path) -> dict:
    info = recovery.status(root)
    info["population"] = read(root / "manifest.json")["population"]
    if "session" in info:
        session = Path(info["session"])
        if not (session / "started.json").exists() and (session / "launch.json").exists():
            launched = read(session / "launch.json")
            info["running"] = (
                not (session / "exit.json").exists()
                and launched["process_start"] is not None
                and recovery.process_start(launched["pid"]) == launched["process_start"]
            )
    if (root / "smoke_complete.json").exists():
        info["smoke_passed"] = read(root / "smoke_complete.json")["passed"]
    return info


def launch(root: Path, env_file: Path, *, smoke_only: bool = False) -> dict:
    with recovery.exclusive(root):
        verify(root)
        require(
            not (root / "complete.json").exists() and not status(root)["running"],
            "EXP6_LAUNCH",
            "no duplicate active or completed run",
        )
        if (root / "smoke_complete.json").exists():
            require(read(root / "smoke_complete.json")["passed"], "EXP6_SMOKE_FAILED", "failed gate is immutable")
        session = root / "recovery/sessions" / recovery.datetime.now(recovery.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        session.mkdir(parents=True)
        argv = [
            "nohup",
            sys.executable,
            "-u",
            "-m",
            "eval.dedup.analysis.exp6_full20k",
            "run",
            "--root",
            str(root),
            "--env-file",
            str(env_file),
            "--session",
            str(session),
        ]
        if smoke_only:
            argv.append("--smoke-only")
        with (session / "run.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(  # noqa: S603 - fixed executable and argv, no shell or credentials
                argv,
                cwd=HERE.parents[3],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        value = {
            "at_utc": recovery.now(),
            "pid": process.pid,
            "process_start": recovery.process_start(process.pid),
            "launcher": "nohup",
            "session": str(session),
            "log": str(session / "run.log"),
            "smoke_only": smoke_only,
            "credential_persisted": False,
        }
        write_json_atomic(session / "launch.json", value)
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "run", "status", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            value = prepare(root)
        elif args.command in {"launch", "run"}:
            require(args.env_file is not None, "EXP6_ARGS", "existing env file required")
            if args.command == "launch":
                value = launch(root, args.env_file.resolve(), smoke_only=args.smoke_only)
            else:
                require(args.session is not None, "EXP6_ARGS", "execution session required")
                run(root, args.env_file.resolve(), args.session.resolve(), smoke_only=args.smoke_only)
                return 0
        elif args.command == "audit":
            with recovery.exclusive(root):
                value = audit(root)
        else:
            value = status(root)
        print(json.dumps(value), flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - do not print potentially credential-bearing exception messages
        print(json.dumps({"error_code": recovery.error_code(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
