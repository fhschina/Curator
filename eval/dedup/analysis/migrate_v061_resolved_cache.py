# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Recontract validated resolved V0.6.1 cache rows after an adapter-only fix."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dedup.contracts import stable_record_id
from eval.dedup.judging.schema import JUDGE_SCHEMA_V2, validate_judge_output
from eval.dedup.judging.schema_v2 import JUDGE_FIELDS_V2
from eval.dedup.validation import read_json, require, sha256_file, write_json_atomic, write_text_atomic


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    require(all(isinstance(row, dict) for row in rows), "CACHE_MIGRATION_INVALID", "cache row must be an object")
    return rows


def migrate(source_run_root: Path, target_run_root: Path) -> dict[str, Any]:
    source_manifest = read_json(source_run_root / "run_manifest.json")
    target_manifest = read_json(target_run_root / "run_manifest.json")
    invariant_fields = (
        "candidate_pairs_sha256",
        "judge_payloads_sha256",
        "payload_membership_sha256",
        "source_pair_count",
        "source_pair_ids_sha256",
    )
    for field in invariant_fields:
        require(
            source_manifest[field] == target_manifest[field],
            "CACHE_MIGRATION_WORKLOAD_CHANGED",
            "source and target workload differ",
            field=field,
        )
    for field in ("hub_model", "prompt_version"):
        require(
            source_manifest["settings"][field] == target_manifest["settings"][field],
            "CACHE_MIGRATION_JUDGE_CHANGED",
            "source and target Judge settings differ",
            field=field,
        )

    source_cache = source_run_root / "logs" / "judge_cache.jsonl"
    target_cache = target_run_root / "logs" / "judge_cache.jsonl"
    require(not target_cache.exists(), "CACHE_MIGRATION_TARGET_NOT_EMPTY", "target cache already exists")
    source_rows = _read_jsonl(source_cache)
    pair_ids: set[str] = set()
    target_digest = target_manifest["judge_contract_digest"]
    migrated_rows = []
    for row in source_rows:
        pair_id = str(row.get("canonical_pair_id"))
        require(
            pair_id not in pair_ids, "CACHE_MIGRATION_DUPLICATE", "source cache has duplicate pair", pair_id=pair_id
        )
        pair_ids.add(pair_id)
        require(row.get("record_type") == "result", "CACHE_MIGRATION_NON_RESULT", "only valid results can migrate")
        require(
            row.get("same_duplicate_group") != "UNRESOLVED",
            "CACHE_MIGRATION_UNRESOLVED",
            "unresolved rows must be re-adapted from their raw response",
            pair_id=pair_id,
        )
        validate_judge_output({field: row[field] for field in JUDGE_FIELDS_V2}, JUDGE_SCHEMA_V2)
        require(
            row.get("judge_model") == target_manifest["settings"]["hub_model"]
            and row.get("prompt_version") == target_manifest["settings"]["prompt_version"]
            and row.get("schema_version") == JUDGE_SCHEMA_V2,
            "CACHE_MIGRATION_JUDGE_CHANGED",
            "cache row does not match the target Judge",
            pair_id=pair_id,
        )
        migrated_rows.append(
            {
                **row,
                "judge_contract_digest": target_digest,
                "judge_result_id": stable_record_id(
                    "judge-result-v2",
                    target_digest,
                    pair_id,
                    row["judge_payload_hash"],
                    row["judge_model"],
                    row["prompt_version"],
                    row["schema_version"],
                ),
            }
        )

    write_text_atomic(
        target_cache,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for row in migrated_rows
        ),
    )
    migration = {
        "schema_version": "dedup-v061-resolved-cache-migration-v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source_run_root": str(source_run_root.resolve()),
        "target_run_root": str(target_run_root.resolve()),
        "source_contract_digest": source_manifest["judge_contract_digest"],
        "target_contract_digest": target_digest,
        "source_cache_sha256": sha256_file(source_cache),
        "target_cache_sha256": sha256_file(target_cache),
        "migrated_resolved_results": len(migrated_rows),
        "eligibility": "same frozen workload, model, prompt, and schema; every migrated row is resolved and V2-valid",
    }
    write_json_atomic(target_run_root / "cache_migration.json", migration)
    return migration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--target-run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source_run_root, args.target_run_root), sort_keys=True))


if __name__ == "__main__":
    main()
