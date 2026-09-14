# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Immutable checkpoint, same-receipt regression audit and two-case local recovery."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from eval.dedup.analysis import exp6_coverage_format_fix as candidate
from eval.dedup.analysis import exp6_service_recovery as original
from eval.dedup.judging.service_wait_collector import ServiceWaitCollector
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

HERE = Path(__file__).resolve()
CASES = ("P14546", "P16717")
read, recovery = original.read, original.recovery


class MissingStage(BaseException):
    def __init__(self, key: str, request: dict):
        super().__init__(key)
        self.key, self.request = key, request


def checkpoint(root: Path) -> dict:
    original.verify(root)
    complete = read(root / "complete.json")
    for path, digest in complete["artifacts"].items():
        recovery.intact(sha256_file(path) == digest)
    paths = [
        root / "manifest.json",
        root / "complete.json",
        root / "transport_events.jsonl",
        root / original.PATCH_DIR / "manifest.json",
        root / original.PATCH_DIR / "complete.json",
        root / "reports/final-summary-v1.md",
    ]
    value = {
        "status": "USER_ACCEPTED_COMPLETE_EXPERIMENT_CHECKPOINT_NOT_RELEASE",
        "version": complete["version"],
        "source_root": str(root),
        "original_contract_digest": complete["contract_digest"],
        "population": complete["population"],
        "statuses": complete["statuses"],
        "semantic_unresolved_valid_only": sum(
            r["status"] == "VALID" and r["public"]["same_duplicate_group"] == "UNRESOLVED"
            for p in (root / "results").glob("*.json")
            for r in [read(p)]
        ),
        "historical_unresolved_field_includes_failures": True,
        "original_results_overwritten": False,
        "release_eligible": False,
        "artifacts": {str(p): sha256_file(p) for p in paths},
    }
    value["contract_digest"] = sha256_json(value)
    write_json_atomic(root / "checkpoint.json", value)
    return value


def prepare(root: Path, target: Path) -> dict:
    frozen = checkpoint(root)
    require(not target.exists(), "POSTRUN_FRESH", "new isolated experiment root required")
    sources = [
        HERE,
        HERE.with_suffix(".md"),
        Path(candidate.__file__),
        Path(candidate.critic_anchor_ids.__file__),
        Path(candidate.coverage_format_recovery.__file__),
        *[
            HERE.parents[3] / "tests/eval/dedup" / name
            for name in (
                "test_exp6_postrun.py",
                "test_exp6_coverage_format_fix.py",
                "test_critic_anchor_ids_v2.py",
                "test_coverage_format_recovery.py",
            )
        ],
    ]
    value = {
        "version": candidate.VERSION,
        "source_root": str(root),
        "cases": list(CASES),
        "checkpoint_digest": frozen["contract_digest"],
        "sources": {str(p): sha256_file(p) for p in sources},
        "checkpoint_sha256": sha256_file(root / "checkpoint.json"),
        "normal_judge_prompts_changed": False,
        "reference_changed": False,
        "semantic_rules_changed": False,
        "max_external_attempts_per_case": 8,
        "fresh_benchmark": False,
        "release_eligible": False,
    }
    value["contract_digest"] = sha256_json(value)
    write_json_atomic(target / "manifest.json", value)
    return value


def verify(target: Path) -> dict:
    value = read(target / "manifest.json")
    recovery.intact(
        value["contract_digest"] == sha256_json({k: v for k, v in value.items() if k != "contract_digest"})
    )
    root = Path(value["source_root"])
    recovery.intact(sha256_file(root / "checkpoint.json") == value["checkpoint_sha256"])
    frozen = read(root / "checkpoint.json")
    recovery.intact(frozen["contract_digest"] == value["checkpoint_digest"])
    for paths in (value["sources"], frozen["artifacts"], read(root / "complete.json")["artifacts"]):
        for path, digest in paths.items():
            recovery.intact(sha256_file(path) == digest)
    original.verify(root)
    return value


def saved_receipt(source: Path, key: str, request: dict) -> dict:
    path = source / "responses" / (key + ".json")
    if not path.exists():
        raise MissingStage(key, request)
    saved, receipt = read(source / "requests" / (key + ".json")), read(path)
    recovery.intact(saved == {"body": request, "request_sha256": sha256_json(request)})
    recovery.intact(receipt["request_sha256"] == saved["request_sha256"])
    return receipt


def replay_case(source: Path, target: Path, row: dict) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    responses = target / "responses"
    if not responses.exists():
        responses.symlink_to(source / "responses", target_is_directory=True)
    key = row["canonical_pair_id"]

    def collect(path: Path, name: str, request: dict, endpoint: str) -> dict:
        recovery.intact(path == target and endpoint == "offline://coverage-format")
        return saved_receipt(source, name, request)

    with recovery.use_collector(collect):
        return candidate.execute_case(
            target,
            row,
            "offline://coverage-format",
            read(source / "main_requests" / (key + ".json")),
            candidate.old.coverage_renderer(),
        )


def offline(target: Path) -> dict:
    manifest = verify(target)
    root, work = Path(manifest["source_root"]), target / "offline"
    require(not work.exists(), "POSTRUN_OFFLINE_EXISTS", "retain previous offline audit")
    identical, changed, pending = 0, [], []
    for row in read(root / "panel_index.json"):
        key = row["canonical_pair_id"]
        source = read(root / "results" / (key + ".json"))
        try:
            result = replay_case(root, work, read(root / "inputs" / (key + ".json")))
        except MissingStage as exc:
            require(row["review_id"] in CASES, "POSTRUN_UNEXPECTED_CALL", "normal paths must reuse exact receipts")
            pending.append({**row, "stage_key": exc.key, "request_sha256": sha256_json(exc.request)})
            continue
        same = {k: v for k, v in result.items() if k != "version"} == {
            k: v for k, v in source.items() if k != "version"
        }
        if row["review_id"] not in CASES:
            recovery.intact(same)
            identical += 1
        else:
            changed.append({**row, "before": source["status"], "after": result["status"], "public": result["public"]})
    value = {
        "version": candidate.VERSION,
        "unaffected_identical": identical,
        "changed_authorized": changed,
        "pending": pending,
        "external_calls": 0,
        "same_receipt_not_fresh_benchmark": True,
    }
    write_json_atomic(target / "offline_complete.json", value)
    return value


def execute_hybrid(source: Path, target: Path, row: dict, endpoint: str, online: Callable) -> dict:
    key = row["canonical_pair_id"]
    reused, added = [], []
    allowed = {key + "-" + s for s in ("coverage-format-repair", "coverage-evidence-repair", "subject", "verifier")}

    def collect(path: Path, name: str, request: dict, relay_endpoint: str) -> dict:
        recovery.intact(path == target and relay_endpoint == endpoint)
        try:
            receipt = saved_receipt(source, name, request)
        except MissingStage:
            require(
                name in allowed and name not in added,
                "POSTRUN_CALL_SCOPE",
                "one call per authorized missing downstream stage",
            )
            added.append(name)
            return online(path, name, request, relay_endpoint)
        write_json_atomic(
            target / "requests" / (name + ".json"), {"body": request, "request_sha256": sha256_json(request)}
        )
        write_json_atomic(target / "responses" / (name + ".json"), receipt)
        reused.append(name)
        return receipt

    with recovery.use_collector(collect):
        result = candidate.execute_case(
            target, row, endpoint, read(source / "main_requests" / (key + ".json")), candidate.old.coverage_renderer()
        )
    with tempfile.TemporaryDirectory(prefix="coverage-format-replay-") as directory:
        # Only the immutable initial request is read from the original run.
        initial = target / "main_requests" / (key + ".json")
        write_json_atomic(initial, read(source / "main_requests" / (key + ".json")))
        replayed = replay_case(target, Path(directory), row)
        recovery.intact(replayed == result)
    return {
        "review_id": row["review_id"],
        "status": result["status"],
        "error_code": result.get("error_code"),
        "public": result["public"],
        "reused_stages": reused,
        "new_stages": added,
        "offline_replay_identical": True,
        "original_result_overwritten": False,
    }


def recover(target: Path, review_id: str, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    manifest = verify(target)
    require(review_id in CASES, "POSTRUN_CASE", "only the two authorized engineering failures")
    root, work = Path(manifest["source_root"]), target / "recovered" / review_id
    require(not work.exists(), "POSTRUN_RECOVERY_EXISTS", "no retry-until-pass or overwriting recoveries")
    gate = read(target / "offline_complete.json")
    require(
        gate["unaffected_identical"] == read(root / "complete.json")["population"] - len(CASES),
        "POSTRUN_GATE",
        "full offline protection gate required",
    )
    row = next(r for r in read(root / "panel_index.json") if r["review_id"] == review_id)
    row = read(root / "inputs" / (row["canonical_pair_id"] + ".json"))
    work.mkdir(parents=True, mode=0o700)
    write_json_atomic(work / "inputs" / (row["canonical_pair_id"] + ".json"), row)
    write_json_atomic(
        work / "provenance.json",
        {"patch_contract_digest": manifest["contract_digest"], "source_root": str(root), "case": review_id},
    )
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "POSTRUN_CREDENTIAL", "existing credential required")
    base = read(root / "manifest.json")
    collector = ServiceWaitCollector(work, work / "session")
    profile = recovery.original.TransportProfile(
        min_interval_seconds=base["min_interval_seconds"],
        max_in_flight=1,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=640,
        max_external_attempts=manifest["max_external_attempts_per_case"],
    )
    with original.transport.ServiceWaitRelay(
        profile=profile,
        logical_model=candidate.old.LOGICAL_MODEL,
        upstream_base_url=base["endpoint"],
        upstream_model=base["model"],
        upstream_api_key=credential,
        timeout_seconds=600,
        expected_generation_parameters=candidate.old.GENERATION,
    ) as relay:
        relay.set_context(
            recovery.original.RelayContext("coverage-format-local-recovery", 0, work / "transport_events.jsonl")
        )
        value = execute_hybrid(root, work, row, relay.endpoint, collector)
    verify(target)
    value["external_attempts"] = len(recovery.events(work / "transport_events.jsonl"))
    write_json_atomic(work / "complete.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("checkpoint", "prepare", "offline", "recover"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--review-id", choices=CASES)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "checkpoint":
            value = checkpoint(args.root.resolve())
        elif args.command == "prepare":
            value = prepare(args.root.resolve(), args.target.resolve())
        elif args.command == "offline":
            value = offline(args.target.resolve())
        else:
            value = recover(args.target.resolve(), args.review_id, args.env_file.resolve())
        print(json.dumps(value), flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - never print credential-bearing exception details
        print(json.dumps({"error_code": recovery.error_code(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
