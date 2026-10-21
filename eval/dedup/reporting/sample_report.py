# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic reports derived exclusively from frozen pairs and their documents."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from eval.dedup.analysis.comparison import build_pair_comparisons
from eval.dedup.analysis.metrics import compute_metrics
from eval.dedup.core.validation import require, sha256_file, write_json_atomic, write_text_atomic
from eval.dedup.reporting import explorer
from eval.dedup.reporting.pair_explorer import _comparison_rows
from eval.dedup.runtime import state

if TYPE_CHECKING:
    from collections.abc import Iterable


def _jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def build(root: Path, manifest: dict, audit_counts: dict) -> dict:
    """Adapt the immutable runtime ledger to the existing comparison and metric APIs."""
    import pyarrow.parquet as pq

    panel = state.read(root / "panel_index.json")
    keys = [row["canonical_pair_id"] for row in panel]
    preparation = root / "preparation/data"
    candidates = pq.read_table(preparation / "candidate_pairs.parquet", filters=[("canonical_pair_id", "in", keys)])
    require(candidates.num_rows == len(keys), "REPORT_PAIR_JOIN", "one candidate per frozen pair")
    by_key = {row["canonical_pair_id"]: row for row in candidates.to_pylist()}
    endpoints = {int(row[side]) for row in by_key.values() for side in ("doc_id_low", "doc_id_high")}
    documents_path = root / manifest["sample_documents"]
    documents = pq.read_table(documents_path).to_pylist()
    require(
        {int(row["doc_id"]) for row in documents} == endpoints, "REPORT_DOCUMENT_SCOPE", "only frozen pair endpoints"
    )
    reports = root / "reports"
    comparisons = root / "analysis/pair_comparisons.parquet"
    successful, errors = [], []
    for row in panel:
        key = row["canonical_pair_id"]
        result = state.read(root / "results" / (key + ".json"))
        candidate = by_key[key]
        attempts = sum(attempt["requests"] for attempt in result["main_attempts"])
        if result["status"] == "VALID":
            successful.append(
                {
                    **result["public"],
                    "canonical_pair_id": key,
                    "canonical_pair_id_version": candidate["canonical_pair_id_version"],
                    "judge_payload_hash": candidate["judge_payload_hash"],
                    "attempts": attempts,
                }
            )
        else:
            errors.append(
                {"canonical_pair_id": key, "attempts": attempts, "errors": [{"error_type": result["error_code"]}]}
            )
    with tempfile.TemporaryDirectory(prefix="dedup-sample-report-") as directory:
        stage = Path(directory)
        for folder in ("data", "logs", "manifests"):
            (stage / folder).mkdir()
        pq.write_table(candidates, stage / "data/candidate_pairs.parquet", compression="zstd")
        pq.write_table(
            pq.read_table(preparation / "pair_provenance.parquet", filters=[("canonical_pair_id", "in", keys)]),
            stage / "data/pair_provenance.parquet",
            compression="zstd",
        )
        (stage / "data/document_outcomes.parquet").symlink_to(documents_path.resolve())
        (stage / "manifests/sut_run_manifest.json").symlink_to((root / "preparation/sut_manifest.json").resolve())
        _jsonl(stage / "data/judge_results.jsonl", successful)
        _jsonl(stage / "logs/judge_errors.jsonl", errors)
        _jsonl(
            stage / "data/judge_payloads.jsonl",
            (state.read(root / "inputs" / (row["canonical_pair_id"] + ".json")) for row in panel),
        )
        build_pair_comparisons(
            candidate_pairs_path=stage / "data/candidate_pairs.parquet",
            pair_provenance_path=stage / "data/pair_provenance.parquet",
            outcomes_path=documents_path,
            judge_results_path=stage / "data/judge_results.jsonl",
            judge_errors_path=stage / "logs/judge_errors.jsonl",
            destination=comparisons,
        )
        metrics = compute_metrics(
            comparisons,
            requested_judge_pairs=len(panel),
            metrics_destination=reports / "metrics.json",
            slices_destination=reports / "slices.csv",
            accounting_destination=reports / "accounting.csv",
            stage_markers=[
                {
                    "step": 6,
                    "name": "Judge",
                    "status": "COMPLETE",
                    "counts": {key: value for key, value in audit_counts.items() if key != "artifacts"},
                }
            ],
        )
        records = explorer.build_pair_explorer_records(stage, _comparison_rows(comparisons))
        reviews = {row["canonical_pair_id"]: row["review_id"] for row in panel}
        for record in records:
            record["review_id"] = reviews[record["pair_id"]]
        groups = explorer.attach_group_context(stage, records)
        write_text_atomic(
            reports / "pair_explorer.html",
            explorer.pair_explorer_html(
                evaluation_run_id=manifest["dataset"]["dataset_version"],
                records=records,
                group_contexts=groups,
            ),
        )
    tokens = [int(row["token_count"]) for row in documents]
    summary = {
        "statistics_scope": "frozen_pairs",
        "pairs": len(panel),
        "documents": len(documents),
        "length_buckets": dict(sorted(Counter(row["length_bucket"] for row in documents).items())),
        "tokens": {"min": min(tokens), "median": median(tokens), "max": max(tokens), "sum": sum(tokens)},
        "schema_valid": metrics["judge"]["schema_valid"],
        "engineering_errors": metrics["judge"]["errors"],
        "resolved": metrics["judge"]["resolved"],
        "unresolved": metrics["judge"]["unresolved"],
    }
    write_json_atomic(reports / "sample_statistics.json", summary)
    removal, cross = metrics["track_5a_removal_frame"], metrics["track_5b_candidate_pool"]
    write_text_atomic(
        reports / "final_report.md",
        (
            "# Dedup evaluation sample\n\n"
            f"Frozen population: {len(panel):,} pairs and {len(documents):,} distinct documents.\n\n"
            "All document distributions describe frozen pair endpoints. Group sizes describe the validated SUT; "
            "group length and hostname summaries describe evaluated members only.\n\n"
            f"Schema-valid pairs: {summary['schema_valid']:,}; engineering errors: {summary['engineering_errors']:,}; "
            f"resolved: {summary['resolved']:,}; unresolved: {summary['unresolved']:,}.\n\n"
            f"5a: {removal['safe']:,} safe and {removal['wrong']:,} wrong removals among "
            f"{removal['resolved']:,} resolved pairs; removal precision: {removal['removal_precision']}.\n\n"
            f"5b: {cross['judged_duplicate_yes']:,} judged duplicates among {cross['resolved']:,} resolved "
            f"candidate pairs; positive yield: {cross['positive_yield']}. This is not corpus recall.\n\n"
            "See [metrics](metrics.json), [slices](slices.csv), [sample statistics](sample_statistics.json), "
            "and [Pair Explorer](pair_explorer.html).\n"
        ),
    )
    files = [
        comparisons,
        *(
            reports / name
            for name in (
                "metrics.json",
                "slices.csv",
                "accounting.csv",
                "sample_statistics.json",
                "final_report.md",
                "pair_explorer.html",
            )
        ),
    ]
    return {**summary, "artifacts": {str(path): sha256_file(path) for path in files}}
