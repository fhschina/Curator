# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Audited construction and validation of immutable benchmark recovery runs."""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

from .artifacts import complete_marker, load_run, read_json, read_jsonl
from .contracts import execution_contract_digest, request_contract_digest
from .relay import audit_paired_request_events, require_no_context_overflow, select_successful_initial_request_events

RECOVERY_FILENAME = "recovery_manifest.json"
_REQUEST_SOURCE_SYMBOLS = {
    "eval/dedup/hosting_benchmark/relay.py": (
        "canonical_request_hash",
        "RelayTarget",
        "RelayContext",
        "RequestRelay",
    ),
    "eval/dedup/hosting_benchmark/runner.py": (
        "_benchmark_evaluation_config",
        "_BorrowedBlockRuntime",
        "_prompt_counter",
    ),
    "eval/dedup/judging/local_ndd.py": ("run_local_ndd_pending",),
    "eval/llm_judge/run_llm_judge.py": (
        "JudgePipelineRuntime",
        "LocalJudgeRuntime",
    ),
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_output(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(  # noqa: S603 - arguments are internal commits and repository paths
        ["git", *args],
        cwd=_repository_root(),
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def _source_at(commit: str, path: str) -> str:
    return str(_git_output("show", f"{commit}:{path}"))


def _request_source_digest(commit: str) -> str:
    selected: dict[str, dict[str, str]] = {}
    for path, names in _REQUEST_SOURCE_SYMBOLS.items():
        nodes = {node.name: node for node in ast.parse(_source_at(commit, path)).body if hasattr(node, "name")}
        require(
            all(name in nodes for name in names),
            "HOSTING_RECOVERY_SOURCE_MISSING",
            "request execution source symbol is missing",
            path=path,
        )
        selected[path] = {
            name: ast.dump(nodes[name], annotate_fields=True, include_attributes=False) for name in names
        }
    return sha256_json(selected)


def _event_rows(run_root: Path, marker: dict[str, Any]) -> list[dict[str, Any]]:
    attempt = run_root / marker["attempt_root"]
    return [row for path in sorted((attempt / "events").glob("*.jsonl")) for row in read_jsonl(path)]


def _verify_importable_marker(run_root: Path, marker: dict[str, Any], *, expected_contract: str) -> None:
    require(marker["status"] == "complete", "HOSTING_RUN_INCOMPLETE", "recovery marker is not complete")
    require(
        marker["contract_digest"] == expected_contract,
        "HOSTING_CONTRACT_CHANGED",
        "recovery marker has an unexpected execution contract",
    )
    attempt = run_root / marker["attempt_root"]
    terminal = attempt / "terminal.jsonl"
    event_paths = sorted((attempt / "events").glob("*.jsonl"))
    require(
        sha256_file(terminal) == marker["terminal_sha256"]
        and sha256_json([sha256_file(path) for path in event_paths]) == marker["event_sha256"],
        "HOSTING_ARTIFACT_CHANGED",
        "recovery source block changed after completion",
    )
    events = _event_rows(run_root, marker)
    require_no_context_overflow(events, block_id=str(marker["block_id"]))
    require(
        not any(int(event["http_status"]) == 429 for event in events),
        "HOSTING_HUB_QUOTA_LIMITED",
        "recovery source block contains a 429 response",
    )


def _marker_entry(run_root: Path, marker_path: Path, marker: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": marker["block_id"],
        "endpoint": marker["endpoint"],
        "attempt_root": marker["attempt_root"],
        "complete_path": str(marker_path.relative_to(run_root)),
        "complete_sha256": sha256_file(marker_path),
        "terminal_sha256": marker["terminal_sha256"],
        "event_sha256": marker["event_sha256"],
        "contract_digest": marker["contract_digest"],
    }


def _validate_imported_pair(run_root: Path, markers: dict[str, dict[str, Any]]) -> None:
    selected = {
        endpoint: select_successful_initial_request_events(
            _event_rows(run_root, marker),
            expected_count=int(marker["pairs"]),
            block_id=str(marker["block_id"]),
        )
        for endpoint, marker in markers.items()
    }
    audit_paired_request_events("local", selected["local"], "hub", selected["hub"])


def create_recovery_run(parent_run_root: str | Path, *, through_block: str) -> Path:
    """Create a new run that imports only a contiguous prefix of clean paired blocks."""

    parent_root, parent_manifest, config = load_run(parent_run_root)
    current_commit = str(_git_output("rev-parse", "HEAD")).strip()
    require(
        not str(_git_output("status", "--porcelain")).strip(),
        "HOSTING_SOURCE_DIRTY",
        "recovery checkout has uncommitted changes",
    )
    blocks = parent_manifest["workload"]["blocks"]
    block_ids = [str(block["block_id"]) for block in blocks]
    require(
        through_block in block_ids,
        "HOSTING_RECOVERY_BOUNDARY_INVALID",
        "recovery boundary is not a frozen workload block",
    )
    boundary = block_ids.index(through_block) + 1
    imported_blocks = blocks[:boundary]
    first_replayed_block = block_ids[boundary] if boundary < len(block_ids) else None
    require(
        first_replayed_block is not None,
        "HOSTING_RECOVERY_BOUNDARY_INVALID",
        "recovery must leave at least one measured block to replay",
    )
    parent_contract = execution_contract_digest(parent_manifest)
    parent_source_digest = _request_source_digest(str(parent_manifest["git_commit"]))
    current_source_digest = _request_source_digest(current_commit)
    require(
        parent_source_digest == current_source_digest,
        "HOSTING_REQUEST_EXECUTION_CHANGED",
        "request execution source changed across the recovery revision",
    )
    identity_digest = sha256_json(
        {
            "config": parent_manifest["config_digest"],
            "git_commit": current_commit,
            "judge_resources": parent_manifest["judge_resources"],
            "source": parent_manifest["source_digests"],
            "tokenizer": parent_manifest["tokenizer"],
            "workload": parent_manifest["workload_digest"],
        }
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"qwen38-hosting-{timestamp}-{identity_digest[:12]}-recovery"
    run_root = config.runs_root / run_id
    require(not run_root.exists(), "HOSTING_RUN_EXISTS", "recovery run root already exists", path=str(run_root))
    run_root.mkdir(parents=True)
    run_root.chmod(0o700)
    shutil.copytree(parent_root / "workload", run_root / "workload")
    shutil.copy2(parent_root / "benchmark_config.json", run_root / "benchmark_config.json")
    manifest = {
        **parent_manifest,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": current_commit,
        "identity_digest": identity_digest,
        "config_path": str(config.source_path),
    }
    write_json_atomic(run_root / "benchmark_manifest.json", manifest)

    imported_entries = []
    for block in imported_blocks:
        paired: dict[str, dict[str, Any]] = {}
        for endpoint in ("local", "hub"):
            source_block = parent_root / "measured" / endpoint / block["block_id"]
            marker_path = complete_marker(source_block)
            require(
                marker_path is not None,
                "HOSTING_RECOVERY_BLOCK_INCOMPLETE",
                "recovery prefix contains an incomplete endpoint block",
                block_id=block["block_id"],
                endpoint=endpoint,
            )
            marker = read_json(marker_path)
            require(
                marker["block_id"] == block["block_id"]
                and marker["endpoint"] == endpoint
                and int(marker["pairs"]) == int(block["pairs"]),
                "HOSTING_RECOVERY_MARKER_MISMATCH",
                "recovery marker differs from the frozen block",
            )
            _verify_importable_marker(parent_root, marker, expected_contract=parent_contract)
            destination = run_root / "measured" / endpoint / block["block_id"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_block, destination)
            copied_path = complete_marker(destination)
            require(copied_path is not None, "HOSTING_RECOVERY_COPY_FAILED", "copied marker is missing")
            copied_marker = read_json(copied_path)
            paired[endpoint] = copied_marker
            imported_entries.append(_marker_entry(run_root, copied_path, copied_marker))
        _validate_imported_pair(run_root, paired)
    gpu_path = parent_root / "gpu_samples.csv"
    gpu_sha256 = None
    if gpu_path.is_file():
        shutil.copy2(gpu_path, run_root / gpu_path.name)
        gpu_sha256 = sha256_file(run_root / gpu_path.name)
    diff = bytes(_git_output("diff", "--binary", str(parent_manifest["git_commit"]), current_commit, binary=True))
    changed_paths = str(
        _git_output("diff", "--name-only", str(parent_manifest["git_commit"]), current_commit)
    ).splitlines()
    recovery = {
        "schema_version": "hosting-recovery-manifest-v1",
        "mode": "resume",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_run_root": str(parent_root),
        "parent_run_id": parent_manifest["run_id"],
        "parent_identity_digest": parent_manifest["identity_digest"],
        "recovery_run_id": run_id,
        "recovery_identity_digest": identity_digest,
        "parent_git_commit": parent_manifest["git_commit"],
        "recovery_git_commit": current_commit,
        "config_digest": manifest["config_digest"],
        "workload_digest": manifest["workload_digest"],
        "request_contract_digest": request_contract_digest(manifest),
        "request_execution_source_digest": current_source_digest,
        "source_revision_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_revision_changed_paths": changed_paths,
        "imported_through_block": through_block,
        "first_replayed_block": first_replayed_block,
        "imported_block_ids": [block["block_id"] for block in imported_blocks],
        "replayed_block_ids": block_ids[boundary:],
        "imported_markers": imported_entries,
        "imported_gpu_samples_sha256": gpu_sha256,
        "status": "validated",
    }
    write_json_atomic(run_root / RECOVERY_FILENAME, recovery)
    validate_recovery_run(run_root, manifest)
    return run_root


def create_headline_validation_run(parent_run_root: str | Path) -> Path:
    """Create a fresh exact-workload replay of only the three concurrency-8 blocks."""

    parent_root, parent_manifest, config = load_run(parent_run_root)
    completion = read_json(parent_root / "run_complete.json")
    require(completion["status"] == "complete", "HOSTING_RUN_INCOMPLETE", "formal parent run is incomplete")
    parent_recovery = validate_recovery_run(parent_root, parent_manifest)
    current_commit = str(_git_output("rev-parse", "HEAD")).strip()
    require(
        current_commit == parent_manifest["git_commit"] and not str(_git_output("status", "--porcelain")).strip(),
        "HOSTING_SOURCE_CHANGED",
        "headline validation must use the clean formal-run revision",
    )
    blocks = parent_manifest["workload"]["blocks"]
    imported_blocks = [block for block in blocks if int(block["concurrency"]) != 8]
    replayed_blocks = [block for block in blocks if int(block["concurrency"]) == 8]
    require(
        len(replayed_blocks) == 3,
        "HOSTING_ACCOUNTING_MISMATCH",
        "headline validation requires exactly three frozen concurrency-8 blocks",
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"qwen38-hosting-{timestamp}-{parent_manifest['identity_digest'][:12]}-c8-validation"
    run_root = config.runs_root / run_id
    require(not run_root.exists(), "HOSTING_RUN_EXISTS", "validation run root already exists", path=str(run_root))
    run_root.mkdir(parents=True)
    run_root.chmod(0o700)
    shutil.copytree(parent_root / "workload", run_root / "workload")
    shutil.copy2(parent_root / "benchmark_config.json", run_root / "benchmark_config.json")
    manifest = {
        **parent_manifest,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config.source_path),
    }
    write_json_atomic(run_root / "benchmark_manifest.json", manifest)
    completed_markers = {(marker["block_id"], marker["endpoint"]): marker for marker in completion["measured"]}
    imported_entries = []
    for block in imported_blocks:
        paired: dict[str, dict[str, Any]] = {}
        for endpoint in ("local", "hub"):
            marker = completed_markers[(block["block_id"], endpoint)]
            require(
                marker_contract_is_accepted(
                    marker,
                    active_contract=completion["contract_digest"],
                    recovery=parent_recovery,
                ),
                "HOSTING_CONTRACT_CHANGED",
                "formal parent marker has an unaccepted execution contract",
            )
            _verify_importable_marker(parent_root, marker, expected_contract=marker["contract_digest"])
            source_block = parent_root / "measured" / endpoint / block["block_id"]
            destination = run_root / "measured" / endpoint / block["block_id"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_block, destination)
            copied_path = complete_marker(destination)
            require(copied_path is not None, "HOSTING_RECOVERY_COPY_FAILED", "copied marker is missing")
            copied_marker = read_json(copied_path)
            paired[endpoint] = copied_marker
            imported_entries.append(_marker_entry(run_root, copied_path, copied_marker))
        _validate_imported_pair(run_root, paired)
    gpu_path = parent_root / "gpu_samples.csv"
    gpu_sha256 = None
    if gpu_path.is_file():
        shutil.copy2(gpu_path, run_root / gpu_path.name)
        gpu_sha256 = sha256_file(run_root / gpu_path.name)
    recovery = {
        "schema_version": "hosting-recovery-manifest-v1",
        "mode": "headline_validation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_run_root": str(parent_root),
        "parent_run_id": parent_manifest["run_id"],
        "parent_identity_digest": parent_manifest["identity_digest"],
        "recovery_run_id": run_id,
        "recovery_identity_digest": manifest["identity_digest"],
        "parent_git_commit": parent_manifest["git_commit"],
        "recovery_git_commit": current_commit,
        "config_digest": manifest["config_digest"],
        "workload_digest": manifest["workload_digest"],
        "request_contract_digest": request_contract_digest(manifest),
        "request_execution_source_digest": _request_source_digest(current_commit),
        "source_revision_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "source_revision_changed_paths": [],
        "imported_through_block": None,
        "first_replayed_block": replayed_blocks[0]["block_id"],
        "imported_block_ids": [block["block_id"] for block in imported_blocks],
        "replayed_block_ids": [block["block_id"] for block in replayed_blocks],
        "imported_markers": imported_entries,
        "imported_gpu_samples_sha256": gpu_sha256,
        "status": "validated",
    }
    write_json_atomic(run_root / RECOVERY_FILENAME, recovery)
    validate_recovery_run(run_root, manifest)
    return run_root


def validate_recovery_run(run_root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    path = run_root / RECOVERY_FILENAME
    if not path.is_file():
        return None
    recovery = read_json(path)
    require(
        recovery["status"] == "validated"
        and recovery["recovery_run_id"] == manifest["run_id"]
        and recovery["recovery_identity_digest"] == manifest["identity_digest"]
        and recovery["recovery_git_commit"] == manifest["git_commit"]
        and recovery["config_digest"] == manifest["config_digest"]
        and recovery["workload_digest"] == manifest["workload_digest"]
        and recovery["request_contract_digest"] == request_contract_digest(manifest),
        "HOSTING_RECOVERY_MANIFEST_CHANGED",
        "recovery provenance differs from the active run manifest",
    )
    blocks = [str(block["block_id"]) for block in manifest["workload"]["blocks"]]
    c8_blocks = [str(block["block_id"]) for block in manifest["workload"]["blocks"] if int(block["concurrency"]) == 8]
    imported = [str(block_id) for block_id in recovery["imported_block_ids"]]
    replayed = [str(block_id) for block_id in recovery["replayed_block_ids"]]
    if recovery["mode"] == "resume":
        valid_boundary = (
            imported == blocks[: len(imported)]
            and len(imported) < len(blocks)
            and replayed == blocks[len(imported) :]
            and recovery["first_replayed_block"] == blocks[len(imported)]
        )
    else:
        require(
            recovery["mode"] == "headline_validation",
            "HOSTING_RECOVERY_MODE_INVALID",
            "unsupported recovery mode",
        )
        valid_boundary = replayed == c8_blocks and imported == [block for block in blocks if block not in c8_blocks]
    require(valid_boundary, "HOSTING_RECOVERY_BOUNDARY_INVALID", "recovery block partition is invalid")
    expected_contract = execution_contract_digest(manifest, git_commit=str(recovery["parent_git_commit"]))
    expected_entries = {(entry["block_id"], entry["endpoint"]): entry for entry in recovery["imported_markers"]}
    require(
        set(expected_entries) == {(block_id, endpoint) for block_id in imported for endpoint in ("local", "hub")},
        "HOSTING_RECOVERY_ACCOUNTING_MISMATCH",
        "recovery marker inventory differs from the imported paired blocks",
    )
    for (block_id, endpoint), entry in expected_entries.items():
        marker_path = complete_marker(run_root / "measured" / endpoint / block_id)
        require(marker_path is not None, "HOSTING_RECOVERY_COPY_FAILED", "imported complete marker is missing")
        marker = read_json(marker_path)
        require(
            sha256_file(marker_path) == entry["complete_sha256"]
            and marker["terminal_sha256"] == entry["terminal_sha256"]
            and marker["event_sha256"] == entry["event_sha256"]
            and marker["attempt_root"] == entry["attempt_root"],
            "HOSTING_RECOVERY_ARTIFACT_CHANGED",
            "an imported recovery artifact changed",
        )
        marker_contract = expected_contract if recovery["mode"] == "resume" else str(entry["contract_digest"])
        _verify_importable_marker(run_root, marker, expected_contract=marker_contract)
    return recovery


def marker_contract_is_accepted(
    marker: dict[str, Any],
    *,
    active_contract: str,
    recovery: dict[str, Any] | None,
) -> bool:
    if marker["contract_digest"] == active_contract:
        return True
    if recovery is None:
        return False
    imported = {entry["attempt_root"]: entry for entry in recovery["imported_markers"]}
    entry = imported.get(marker["attempt_root"])
    return bool(entry and entry["contract_digest"] == marker["contract_digest"])
