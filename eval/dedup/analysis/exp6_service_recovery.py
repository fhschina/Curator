# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Append-only service-wait continuation and isolated infrastructure retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from eval.dedup.analysis import exp6_auth_db_recovery as previous
from eval.dedup.analysis import exp6_full20k as full
from eval.dedup.judging import service_wait_relay as transport
from eval.dedup.judging.service_wait_collector import ServiceWaitCollector
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

HERE = Path(__file__).resolve()
PATCH_DIR = "recovery/service-wait-v1"
CASE_IDS = ("P19310", "P19312")
read, recovery = full.read, full.recovery


def verify(root: Path) -> dict:
    previous.verify(root)
    base = read(root / "manifest.json")
    patch = read(root / PATCH_DIR / "manifest.json")
    recovery.intact(
        patch["contract_digest"] == sha256_json({k: v for k, v in patch.items() if k != "contract_digest"})
    )
    recovery.intact(base["contract_digest"] == patch["original_contract_digest"])
    for group in (patch["sources"], patch["artifacts"]):
        for path, digest in group.items():
            recovery.intact(Path(path).is_file() and sha256_file(path) == digest)
    with (root / "transport_events.jsonl").open("rb") as stream:
        prefix = stream.read(patch["transport_prefix_bytes"])
    recovery.intact(hashlib.sha256(prefix).hexdigest() == patch["transport_prefix_sha256"])
    return patch


def prepare(root: Path) -> dict:
    with recovery.exclusive(root):
        previous.verify(root)
        base = read(root / "manifest.json")
        require(
            not full.status(root)["running"] and not (root / "complete.json").exists(),
            "SERVICE_STATE",
            "inactive incomplete run required",
        )
        require(not (root / PATCH_DIR / "manifest.json").exists(), "SERVICE_FROZEN", "already frozen")
        rows = read(root / "panel_index.json")
        indexed = {r["review_id"]: r for r in rows}
        cases = [indexed[name] for name in CASE_IDS]
        for row in cases:
            result = read(root / "results" / (row["canonical_pair_id"] + ".json"))
            require(
                result["status"] == "ENGINEERING_FAILURE" and result["error_code"] == "DataDesignerError",
                "SERVICE_CASE",
                "only the authorized infrastructure failures",
            )
        paths = [p for folder in ("requests", "responses", "results") for p in (root / folder).glob("*.json")]
        artifacts = {
            str(p): sha256_file(p)
            for p in [
                *paths,
                root / "manifest.json",
                root / "smoke_complete.json",
                root / previous.PATCH_DIR / "manifest.json",
            ]
        }
        sources = [
            HERE,
            HERE.with_suffix(".md"),
            Path(transport.__file__),
            Path(sys.modules[ServiceWaitCollector.__module__].__file__),
            HERE.parents[3] / "tests/eval/dedup/test_exp6_service_recovery.py",
            HERE.parents[3] / "tests/eval/dedup/test_service_wait_relay.py",
            HERE.parents[3] / "tests/eval/dedup/test_service_wait_collector.py",
        ]
        prefix = (root / "transport_events.jsonl").read_bytes()
        saved = {p.stem for p in (root / "results").glob("*.json")}
        manifest = {
            "transport_patch_contract": transport.CONTRACT,
            "original_contract_digest": base["contract_digest"],
            "semantic_runtime_version": full.VERSION,
            "at_utc": recovery.now(),
            "judge_changes": False,
            "sources": {str(p): sha256_file(p) for p in sources},
            "artifacts": artifacts,
            "transport_prefix_bytes": len(prefix),
            "transport_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "initial_saved_results": len(saved),
            "pending_pair_ids": sorted({r["canonical_pair_id"] for r in rows} - saved),
            "recovery_cases": cases,
            "old_results_overwritten": False,
            "release_eligible": False,
            "wait_profile": {
                "inner_attempts": 2,
                "max_service_submissions_per_stage": 6,
                "service_backoff_seconds": [60, 120, 240, 300, 300],
                "max_service_backoff_seconds": 1800,
                "rate_limit_submissions": 60,
                "max_rate_backoff_seconds": 21600,
                "global_attempt_budget": base["max_external_attempts"],
                "workers": base["workers"],
                "min_interval_seconds": base["min_interval_seconds"],
                "local_relay_ambiguity_stops_pending": True,
            },
        }
        manifest["contract_digest"] = sha256_json(manifest)
        write_json_atomic(root / PATCH_DIR / "manifest.json", manifest)
        return {
            "transport_patch_contract": transport.CONTRACT,
            "contract_digest": manifest["contract_digest"],
            "saved": len(saved),
            "pending": len(manifest["pending_pair_ids"]),
            "separate_cases": list(CASE_IDS),
        }


@contextmanager
def transport_override() -> Iterator[None]:
    old_relay, old_collector = full.limits.RateLimitRelay, full.limits.RateLimitCollector
    full.limits.RateLimitRelay, full.limits.RateLimitCollector = transport.ServiceWaitRelay, ServiceWaitCollector
    try:
        yield
    finally:
        full.limits.RateLimitRelay, full.limits.RateLimitCollector = old_relay, old_collector


def finish_report(root: Path) -> dict:
    patch = verify(root)
    events = recovery.events(root / "transport_events.jsonl")
    added = [
        e for e in events if e.get("transport_patch_contract") == transport.CONTRACT and e.get("external_request")
    ]
    value = {
        "transport_patch_contract": transport.CONTRACT,
        "patch_contract_digest": patch["contract_digest"],
        "original_saved_results_preserved": patch["initial_saved_results"],
        "full_completion": read(root / "complete.json")["population"],
        "additional_external_attempts": len(added),
        "http_statuses": dict(Counter(str(e["http_status"]) for e in added)),
        "database_busy_401_attempts": sum(bool(e.get("auth_database_busy")) for e in added),
        "original_failures_preserved": True,
        "separate_case_recovery_not_merged": True,
        "release_eligible": False,
    }
    write_json_atomic(root / PATCH_DIR / "complete.json", value)
    return value


def run(root: Path, env_file: Path, session: Path) -> None:
    verify(root)
    with transport_override():
        full.run(root, env_file, session)
    print(json.dumps(finish_report(root)), flush=True)


def launch(root: Path, env_file: Path) -> dict:
    with recovery.exclusive(root):
        patch = verify(root)
        require(
            not full.status(root)["running"] and not (root / "complete.json").exists(),
            "SERVICE_LAUNCH",
            "no active or completed run",
        )
        session = root / "recovery/sessions" / recovery.datetime.now(recovery.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        session.mkdir(parents=True)
        argv = [
            "nohup",
            sys.executable,
            "-u",
            "-m",
            "eval.dedup.analysis.exp6_service_recovery",
            "run",
            "--root",
            str(root),
            "--env-file",
            str(env_file),
            "--session",
            str(session),
        ]
        with (session / "run.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell or credentials
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
            "transport_patch_contract": transport.CONTRACT,
            "patch_contract_digest": patch["contract_digest"],
            "credential_persisted": False,
        }
        write_json_atomic(session / "launch.json", value)
        return value


def recover_case(root: Path, target: Path, env_file: Path, review_id: str) -> dict:
    from eval.dedup.cli import _load_repository_env

    patch = verify(root)
    base = read(root / "manifest.json")
    selected = [r for r in patch["recovery_cases"] if r["review_id"] == review_id]
    require(len(selected) == 1, "SERVICE_CASE", "authorized isolated recovery only")
    key = selected[0]["canonical_pair_id"]
    require(not target.exists(), "SERVICE_FRESH_CASE", "separate fresh recovery directory required")
    target.mkdir(parents=True, mode=0o700)
    for folder in ("inputs", "main_requests"):
        write_json_atomic(target / folder / (key + ".json"), read(root / folder / (key + ".json")))
    manifest = {
        "mode": "USER_AUTHORIZED_SINGLE_INFRASTRUCTURE_FAILURE_RECOVERY_NOT_FULL_BENCHMARK",
        "version": full.VERSION,
        "source_run": str(root),
        "source_contract_digest": base["contract_digest"],
        "transport_patch_contract": transport.CONTRACT,
        "patch_contract_digest": patch["contract_digest"],
        "case_key": key,
        "review_id": review_id,
        "max_external_attempts": 32,
        "sources": {**base["sources"], **patch["sources"]},
        "artifacts": {
            str(p): sha256_file(p)
            for p in [root / PATCH_DIR / "manifest.json", root / "results" / (key + ".json"), *target.glob("*/*.json")]
        },
        "model": base["model"],
        "endpoint": base["endpoint"],
        "generation": full.runtime.old.GENERATION,
        "old_answers_reused": False,
        "credential_persisted": False,
        "release_eligible": False,
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(target / "manifest.json", manifest)
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "SERVICE_CREDENTIAL", "existing key required")
    collector = ServiceWaitCollector(target, target / "session")
    row = read(target / "inputs" / (key + ".json"))
    profile = recovery.original.TransportProfile(
        min_interval_seconds=base["min_interval_seconds"],
        max_in_flight=1,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=640,
        max_external_attempts=32,
    )
    with (
        transport.ServiceWaitRelay(
            profile=profile,
            logical_model=full.runtime.old.LOGICAL_MODEL,
            upstream_base_url=base["endpoint"],
            upstream_model=base["model"],
            upstream_api_key=credential,
            timeout_seconds=600,
            expected_generation_parameters=full.runtime.old.GENERATION,
        ) as relay,
        recovery.use_collector(collector),
    ):
        relay.set_context(
            recovery.original.RelayContext("service-wait-single-case", 0, target / "transport_events.jsonl")
        )
        result = full.execute(target, row, relay.endpoint, full.runtime.old.coverage_renderer())
    report = full.replay_results(target, [row])
    report.pop("replayed_keys")
    value = {
        **report,
        "review_id": row["review_id"],
        "version": full.VERSION,
        "status": result["status"],
        "error_code": result.get("error_code"),
        "public": result["public"],
        "contract_digest": manifest["contract_digest"],
        "original_failure_preserved": True,
        "not_fresh_benchmark": True,
        "release_eligible": False,
    }
    verify(root)
    write_json_atomic(target / "complete.json", value)
    return {k: v for k, v in value.items() if k != "artifacts"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "run", "recover-case", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--review-id", choices=CASE_IDS)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            value = prepare(root)
        elif args.command == "status":
            value = full.status(root)
        else:
            require(args.env_file is not None, "SERVICE_ARGS", "existing env file required")
            if args.command == "launch":
                value = launch(root, args.env_file.resolve())
            elif args.command == "recover-case":
                require(
                    args.target is not None and args.review_id is not None, "SERVICE_ARGS", "target and case required"
                )
                value = recover_case(root, args.target.resolve(), args.env_file.resolve(), args.review_id)
            else:
                require(args.session is not None, "SERVICE_ARGS", "execution session required")
                run(root, args.env_file.resolve(), args.session.resolve())
                return 0
        print(json.dumps(value), flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - never print credential-bearing exception details
        print(json.dumps({"error_code": recovery.error_code(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
