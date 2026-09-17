# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepare, execute, replay, and audit immutable Judge v0.7 runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from eval.dedup.core.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic
from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.runtime import JUDGE_CONTRACT_VERSION, TOOL_VERSION, contract, execution, state, transport

HERE = Path(__file__).resolve()
RELEASE = HERE.parents[1] / "release"
PROTOCOL = RELEASE / "README.md"
SMOKE_PANEL = RELEASE / "smoke_panel.json"
RELEASE_MANIFEST = RELEASE / "manifest.json"
VERSION = TOOL_VERSION
POPULATION = 20000
SMOKE_SIZE = 24
REQUIRED_SMOKE_STAGES = {"main", "coverage", "subject", "verifier"}
FULL_MODE = "FRESH_FULL20K_WITH_SMOKE_GATE"
SMOKE_ONLY_MODE = "FRESH_SMOKE_ONLY"
read = state.read


def _configured_path(explicit: Path | None, environment_variable: str) -> Path | None:
    value = explicit or (Path(raw) if (raw := os.environ.get(environment_variable)) else None)
    return value.expanduser().resolve() if value is not None else None


def _verify_manifest_digest(root: Path, *, error_code: str) -> dict:
    manifest = read(root / "manifest.json")
    require(
        manifest.get("contract_digest")
        == sha256_json({key: value for key, value in manifest.items() if key != "contract_digest"}),
        error_code,
        "frozen manifest digest",
    )
    return manifest


def _bound_digest(bindings: dict[str, str], relative_path: Path, *, error_code: str) -> str:
    parts = relative_path.parts
    matches = {digest for path, digest in bindings.items() if Path(path).parts[-len(parts) :] == parts}
    require(len(matches) == 1, error_code, f"one digest for {relative_path.as_posix()}")
    return next(iter(matches))


def _verify_bound_file(root: Path, relative_path: Path, bindings: dict[str, str], *, error_code: str) -> Path:
    path = root / relative_path
    require(
        path.is_file() and sha256_file(path) == _bound_digest(bindings, relative_path, error_code=error_code),
        error_code,
        f"unchanged {relative_path.as_posix()}",
    )
    return path


def _current_sources(*manifests: dict, extras: list[Path]) -> dict[str, str]:
    """Rebind historical source paths to this checkout and Python environment."""
    repository = HERE.parents[3]
    candidates = {path.resolve() for path in extras}
    site_roots = [Path(path).resolve() for path in sys.path if path and Path(path).name == "site-packages"]
    for manifest in manifests:
        for raw_path in manifest.get("sources", {}):
            normalized = Path(raw_path).as_posix()
            if "/Curator/" in normalized:
                candidate = repository / normalized.split("/Curator/", 1)[1]
                if candidate.is_file():
                    candidates.add(candidate.resolve())
            elif "/site-packages/" in normalized:
                relative = Path(normalized.split("/site-packages/", 1)[1])
                for site_root in site_roots:
                    candidate = site_root / relative
                    if candidate.is_file():
                        candidates.add(candidate.resolve())
                        break
    return {str(path): sha256_file(path) for path in sorted(candidates)}


def frozen_smoke(rows: list[dict]) -> list[dict]:
    """Bind the release-owned smoke panel to the frozen 20K population."""
    by_key = {r["canonical_pair_id"]: r for r in rows}
    template = read(SMOKE_PANEL)
    require(
        isinstance(template, list)
        and len(template) == SMOKE_SIZE
        and len({item.get("canonical_pair_id") for item in template}) == SMOKE_SIZE,
        "V07_SMOKE_PANEL",
        "unique release-owned smoke panel",
    )
    selected = []
    for item in template:
        require(
            set(item) == {"canonical_pair_id", "review_id", "sampling_reason"}
            and item["canonical_pair_id"] in by_key
            and item["review_id"] == by_key[item["canonical_pair_id"]]["review_id"]
            and isinstance(item["sampling_reason"], str)
            and item["sampling_reason"],
            "V07_SMOKE_PANEL",
            "smoke rows bound to the frozen 20K population",
        )
        selected.append({**by_key[item["canonical_pair_id"]], "sampling_reason": item["sampling_reason"]})
    return selected


def verify(root: Path) -> dict:
    manifest = _verify_manifest_digest(root, error_code="DEDUP_MANIFEST")
    tool_version = manifest.get("tool_version", manifest.get("version"))
    judge_contract_version = manifest.get("judge_contract_version", manifest.get("runtime_version"))
    require(
        tool_version in {JUDGE_CONTRACT_VERSION, TOOL_VERSION}
        and judge_contract_version == JUDGE_CONTRACT_VERSION
        and manifest["generation"] == contract.GENERATION
        and manifest["old_answers_reused"] is False,
        "DEDUP_RUNTIME_BINDING",
        "a supported immutable v0.7 run is required",
    )
    return manifest


def prepare(root: Path, *, source: Path, smoke_only: bool = False) -> dict:
    require(not root.exists(), "V07_FRESH_ROOT", "new run root; no historical answers may be copied")
    source = source.expanduser().resolve()
    original = _verify_manifest_digest(source, error_code="V07_SOURCE_MANIFEST")
    _verify_bound_file(
        source,
        Path("panel_index.json"),
        original["artifacts"],
        error_code="V07_SOURCE_ARTIFACT",
    )
    source_completion = read(source / "complete.json")
    state.intact(source_completion["contract_digest"] == original["contract_digest"])
    source_rows = read(source / "panel_index.json")
    state.intact(len(source_rows) == len({r["canonical_pair_id"] for r in source_rows}) == POPULATION)
    smoke = frozen_smoke(source_rows)
    if smoke_only:
        by_key = {row["canonical_pair_id"]: row for row in source_rows}
        rows = [by_key[row["canonical_pair_id"]] for row in smoke]
    else:
        rows = source_rows
    extras = [
        HERE,
        RELEASE_MANIFEST,
        PROTOCOL,
        SMOKE_PANEL,
        Path(contract.__file__),
        Path(execution.__file__),
        Path(state.__file__),
        Path(transport.__file__),
        *sorted((RELEASE / "prompts").glob("*")),
    ]
    sources = _current_sources(original, extras=extras)
    root.mkdir(parents=True, mode=0o700)
    artifacts = {}
    for index, row in enumerate(rows, 1):
        key = row["canonical_pair_id"]
        state.intact(Path(key).name == key)
        for folder in ("inputs", "main_requests"):
            relative = Path(folder) / (key + ".json")
            source_path = _verify_bound_file(
                source,
                relative,
                original["artifacts"],
                error_code="V07_SOURCE_ARTIFACT",
            )
            target = root / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source_path, target)
            state.intact(sha256_file(target) == sha256_file(source_path))
            artifacts[str(target)] = sha256_file(target)
        payload = read(root / "inputs" / (key + ".json"))
        frozen = read(root / "main_requests" / (key + ".json"))
        state.intact(all(payload[k] == v for k, v in row.items()))
        state.intact(frozen["request_sha256"] == sha256_json(frozen["body"]))
        state.intact(frozen["body"] == contract.body(frozen["body"]["messages"]))
        if index % 1000 == 0:
            print(json.dumps({"prepared": index}), flush=True)
    write_json_atomic(root / "panel_index.json", rows)
    write_json_atomic(root / "smoke_panel.json", smoke)
    write_text_atomic(root / "protocol.md", PROTOCOL.read_text())
    provenance = root / "provenance"
    provenance.mkdir()
    provenance_files = {
        provenance / "source_manifest.json": source / "manifest.json",
        provenance / "source_complete.json": source / "complete.json",
    }
    for target, source_path in provenance_files.items():
        shutil.copyfile(source_path, target)
    for path in (
        root / "panel_index.json",
        root / "smoke_panel.json",
        root / "protocol.md",
        *provenance_files,
    ):
        artifacts[str(path)] = sha256_file(path)
    manifest = {
        "version": TOOL_VERSION,
        "runtime_version": JUDGE_CONTRACT_VERSION,
        "tool_version": TOOL_VERSION,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "mode": SMOKE_ONLY_MODE if smoke_only else FULL_MODE,
        "distribution": "FHSCHINA_FORK_DEDUP_EVAL_ONLY",
        "at_utc": state.now(),
        "population": len(rows),
        "source_population": len(source_rows),
        "smoke_size": len(smoke),
        "required_smoke_stages": sorted(REQUIRED_SMOKE_STAGES),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_run": str(source),
        "source_contract_digest": original["contract_digest"],
        "smoke_panel_source": str(SMOKE_PANEL),
        "smoke_panel_source_sha256": sha256_file(SMOKE_PANEL),
        "development_payload_matches": original["development_payload_matches"],
        "old_answers_reused": False,
        "smoke_included_in_population": True,
        "independent_holdout_passed": False,
        "release_eligible": False,
        "minhash_diagnostics": "UNAVAILABLE_MISSING_CONTRACT",
        "model": original["model"],
        "endpoint": original["endpoint"],
        "generation": contract.GENERATION,
        "workers": original["workers"],
        "min_interval_seconds": original["min_interval_seconds"],
        "max_external_attempts": original["max_external_attempts"],
        "sources": sources,
        "source_digest": sha256_json(sources),
        "contract_lineage": {
            "original_release_commit": "ef17d9b6527532521580781570942728509629bd",
            "judge_contract_version": JUDGE_CONTRACT_VERSION,
        },
        "artifacts": artifacts,
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {k: v for k, v in manifest.items() if k not in {"sources", "artifacts"}}


def require_execution_mode(manifest: dict, *, smoke_only: bool) -> None:
    require(
        manifest.get("mode") != SMOKE_ONLY_MODE or smoke_only,
        "DEDUP_SMOKE_ONLY_ROOT",
        "a smoke-only prepared root cannot continue as a full run",
    )


def execute(root: Path, row: dict, endpoint: str, render: Callable) -> dict:
    key = row["canonical_pair_id"]
    return execution.execute_case(
        root, read(root / "inputs" / (key + ".json")), endpoint, read(root / "main_requests" / (key + ".json")), render
    )


def replay_results(root: Path, rows: list[dict]) -> dict:
    """Replay into a disposable result directory; historical results are never rewritten."""
    statuses, stages = Counter(), Counter()
    repairs, corrections, unresolved = Counter(), 0, 0
    seen = set()
    bindings = {}
    render = contract.coverage_renderer()
    with tempfile.TemporaryDirectory(prefix="v07-replay-") as directory:
        target = Path(directory)
        for folder in ("inputs", "main_requests", "responses"):
            (target / folder).symlink_to(root / folder, target_is_directory=True)

        def replay(path: Path, key: str, body: dict, endpoint: str) -> dict:
            state.intact(path == target and key not in seen and endpoint == "offline://v07")
            request, receipt = (read(root / folder / (key + ".json")) for folder in ("requests", "responses"))
            state.intact(request == {"body": body, "request_sha256": sha256_json(body)})
            state.intact(receipt["request_sha256"] == request["request_sha256"])
            seen.add(key)
            return receipt

        with state.use_collector(replay):
            for row in rows:
                key = row["canonical_pair_id"]
                path = root / "results" / (key + ".json")
                state.intact(path.is_file())
                original = read(path)
                state.intact(original["version"] == JUDGE_CONTRACT_VERSION)
                before = set(seen)
                expected = {key + "-" + stage["stage"] for stage in original["stages"]}
                state.intact(len(expected) == len(original["stages"]))
                for stage in original["stages"]:
                    receipt_path = root / "responses" / (key + "-" + stage["stage"] + ".json")
                    state.intact(sha256_file(receipt_path) == stage["response_sha256"])
                result = execute(target, row, "offline://v07", render)
                state.intact(result == original and seen - before == expected)
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


def check_smoke(root: Path, manifest: dict, *, persist_missing: bool = True) -> dict:
    rows = read(root / "smoke_panel.json")
    report = replay_results(root, rows)
    state.intact(len(rows) == manifest["smoke_size"])
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
        state.intact(read(path) == value)
    elif persist_missing:
        write_json_atomic(path, value)
    else:
        state.intact(False)
    require(passed, "V07_SMOKE_FAILED", "stop before the remaining population; preserve all smoke results")
    return {k: v for k, v in value.items() if k not in {"artifacts", "replayed_keys"}}


def audit(root: Path) -> dict:
    manifest = verify(root)
    legacy = manifest.get("tool_version") is None
    rows = read(root / "panel_index.json")
    state.intact(len(rows) == manifest["population"])
    state.intact({p.stem for p in (root / "results").glob("*.json")} == {r["canonical_pair_id"] for r in rows})
    check_smoke(root, manifest, persist_missing=not legacy)
    report = replay_results(root, rows)
    for folder in ("requests", "responses"):
        state.intact({p.stem for p in (root / folder).glob("*.json")} == set(report["replayed_keys"]))
    report.pop("replayed_keys")
    transport = [e for e in state.events(root / "transport_events.jsonl") if e.get("external_request")]
    value = {
        **report,
        "version": TOOL_VERSION,
        "tool_version": TOOL_VERSION,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "contract_digest": manifest["contract_digest"],
        "fresh_full20k": len(rows) == POPULATION,
        "release_eligible": False,
        "external_attempts": len(transport),
        "http_statuses": dict(Counter(e.get("http_status") for e in transport)),
    }
    path = root / "complete.json"
    if path.exists():
        existing = read(path)
        if legacy:
            state.intact(
                existing.get("contract_digest") == manifest["contract_digest"]
                and existing.get("population") == value["population"]
                and existing.get("saved_calls_replayed") == value["saved_calls_replayed"]
            )
            value["legacy_complete_verified"] = True
        else:
            state.intact(existing == json.loads(json.dumps(value)))
    elif not legacy:
        write_json_atomic(path, value)
    else:
        state.intact(False)
    return {k: v for k, v in value.items() if k != "artifacts"}


def collect_rows(
    root: Path,
    session: Path,
    rows: list[dict],
    relay: transport.RateLimitRelay,
    collector: transport.RateLimitCollector,
    workers: int,
    phase: str,
) -> None:
    pending = iter(r for r in rows if not (root / "results" / (r["canonical_pair_id"] + ".json")).exists())
    render = contract.coverage_renderer()
    with state.use_collector(collector), ThreadPoolExecutor(max_workers=workers) as pool:
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
                    "at_utc": state.now(),
                    "phase": phase,
                    "review_id": result["review_id"],
                    "status": result["status"],
                    "completed": len(list((root / "results").glob("*.json"))),
                }
                write_json_atomic(session / "progress" / (result["canonical_pair_id"] + ".json"), progress)
                print(json.dumps(progress), flush=True)
            require(relay._circuit_reason is None, "V07_CIRCUIT", "stop without consuming remaining pairs")
            for _ in ready:
                submit()


def run(root: Path, env_file: Path, session: Path, *, smoke_only: bool = False) -> None:
    from eval.dedup.cli import _load_repository_env

    with state.exclusive(root):
        manifest = verify(root)
        require_execution_mode(manifest, smoke_only=smoke_only)
        require(not (root / "complete.json").exists(), "V07_COMPLETE", "completed run is immutable")
        require(not (session / "started.json").exists(), "V07_SESSION", "fresh execution session required")
        write_json_atomic(
            session / "started.json",
            {"at_utc": state.now(), "pid": os.getpid(), "process_start": state.process_start(os.getpid())},
        )
        collector = transport.RateLimitCollector(root, session)
        stopped = threading.Event()

        def heartbeat() -> None:
            sequence = 0
            while not stopped.is_set():
                write_json_atomic(
                    session / "heartbeats" / f"{sequence:06d}.json",
                    {
                        "at_utc": state.now(),
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
            require(bool(credential), "V07_CREDENTIAL", "existing NVIDIA key required")
            used = sum(bool(e.get("external_request")) for e in state.events(root / "transport_events.jsonl"))
            budget = manifest["max_external_attempts"] - used
            require(budget > 0, "V07_BUDGET", "bounded external request budget")
            profile = TransportProfile(
                min_interval_seconds=manifest["min_interval_seconds"],
                max_in_flight=manifest["workers"],
                max_attempts=2,
                retry_base_seconds=10,
                retry_cap_seconds=40,
                request_deadline_seconds=640,
                max_external_attempts=budget,
            )
            with transport.RateLimitRelay(
                profile=profile,
                logical_model=contract.LOGICAL_MODEL,
                upstream_base_url=manifest["endpoint"],
                upstream_model=manifest["model"],
                upstream_api_key=credential,
                timeout_seconds=600,
                expected_generation_parameters=contract.GENERATION,
            ) as relay:
                relay.set_context(RelayContext("v07-" + session.name, 0, root / "transport_events.jsonl"))
                smoke = read(root / "smoke_panel.json")
                smoke_ids = {r["canonical_pair_id"] for r in smoke}
                if not (root / "smoke_complete.json").exists():
                    state.intact({p.stem for p in (root / "results").glob("*.json")} <= smoke_ids)
                    collect_rows(root, session, smoke, relay, collector, manifest["workers"], "SMOKE")
                print(json.dumps({"event": "SMOKE_GATE_PASSED", **check_smoke(root, manifest)}), flush=True)
                if not smoke_only:
                    print(
                        json.dumps({"event": "FULL20K_CONTINUATION", "runtime_version": JUDGE_CONTRACT_VERSION}),
                        flush=True,
                    )
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
                {"at_utc": state.now(), "status": "SMOKE_PASSED" if smoke_only else "COMPLETE", "exit_code": 0},
            )
        except BaseException as exc:
            write_json_atomic(
                session / "exit.json",
                {
                    "at_utc": state.now(),
                    "status": "STOPPED",
                    "error_code": state.error_code(exc),
                    "exit_code": 1,
                },
            )
            raise
        finally:
            stopped.set()
            thread.join(timeout=5)


def status(root: Path) -> dict:
    info = state.status(root)
    info["population"] = read(root / "manifest.json")["population"]
    if "session" in info:
        session = Path(info["session"])
        if not (session / "started.json").exists() and (session / "launch.json").exists():
            launched = read(session / "launch.json")
            info["running"] = (
                not (session / "exit.json").exists()
                and launched["process_start"] is not None
                and state.process_start(launched["pid"]) == launched["process_start"]
            )
    if (root / "smoke_complete.json").exists():
        info["smoke_passed"] = read(root / "smoke_complete.json")["passed"]
    return info


def launch(root: Path, env_file: Path, *, smoke_only: bool = False) -> dict:
    with state.exclusive(root):
        manifest = verify(root)
        require_execution_mode(manifest, smoke_only=smoke_only)
        require(
            not (root / "complete.json").exists() and not status(root)["running"],
            "V07_LAUNCH",
            "no duplicate active or completed run",
        )
        if (root / "smoke_complete.json").exists():
            require(read(root / "smoke_complete.json")["passed"], "V07_SMOKE_FAILED", "failed gate is immutable")
        session = root / "recovery/sessions" / state.datetime.now(state.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        session.mkdir(parents=True)
        argv = [
            "nohup",
            sys.executable,
            "-u",
            "-m",
            "eval.dedup",
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
            "at_utc": state.now(),
            "pid": process.pid,
            "process_start": state.process_start(process.pid),
            "launcher": "nohup",
            "session": str(session),
            "log": str(session / "run.log"),
            "smoke_only": smoke_only,
            "credential_persisted": False,
        }
        write_json_atomic(session / "launch.json", value)
        return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "run", "status", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Frozen 20K source bundle; defaults to CURATOR_V07_SOURCE_RUN.",
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            source = _configured_path(args.source_run, "CURATOR_V07_SOURCE_RUN")
            require(source is not None, "V07_SOURCE_RUN", "--source-run or CURATOR_V07_SOURCE_RUN")
            value = prepare(root, source=source)
        elif args.command in {"launch", "run"}:
            require(args.env_file is not None, "V07_ARGS", "existing env file required")
            if args.command == "launch":
                value = launch(root, args.env_file.resolve(), smoke_only=args.smoke_only)
            else:
                require(args.session is not None, "V07_ARGS", "execution session required")
                run(root, args.env_file.resolve(), args.session.resolve(), smoke_only=args.smoke_only)
                return 0
        elif args.command == "audit":
            with state.exclusive(root):
                value = audit(root)
        else:
            value = status(root)
        print(json.dumps(value), flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - do not print potentially credential-bearing exception messages
        print(json.dumps({"error_code": state.error_code(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
