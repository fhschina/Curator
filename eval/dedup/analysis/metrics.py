# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step 9: frame-valid V0 metrics, diagnostics, and slices."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from eval.dedup.contracts import DuplicateAnswer
from eval.dedup.validation import require, write_json_atomic


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(
    comparisons_path: Path,
    *,
    requested_judge_pairs: int,
    metrics_destination: Path,
    slices_destination: Path,
    accounting_destination: Path,
    stage_markers: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for metrics"
        raise RuntimeError(msg) from exc
    rows = pq.read_table(comparisons_path).to_pylist()
    for row in rows:
        ratio = float(row["token_length_ratio"])
        row["token_length_ratio_bucket"] = (
            "0-0.25" if ratio < 0.25 else "0.25-0.5" if ratio < 0.5 else "0.5-0.8" if ratio < 0.8 else "0.8-1.0"
        )
        size = int(row["predicted_group_size_low"])
        row["group_size_bucket_low"] = (
            "singleton"
            if size == 1
            else "size_2"
            if size == 2
            else "size_3_5"
            if size <= 5
            else "size_6_20"
            if size <= 20
            else "size_21_plus"
        )
    valid = [row for row in rows if row["judge_status"] == "valid"]
    resolved = [row for row in valid if row["same_duplicate_group"] in {DuplicateAnswer.YES, DuplicateAnswer.NO}]
    removal_rows = [row for row in rows if row["has_track_5a"]]
    removal_valid = [row for row in removal_rows if row["judge_status"] == "valid"]
    removal = [row for row in removal_valid if row["removal_outcome"] in {"safe_removal", "wrong_removal"}]
    safe = sum(row["removal_outcome"] == "safe_removal" for row in removal)
    removal_total = len(removal)
    ci_low, ci_high = wilson_interval(safe, removal_total)
    frame_sizes = {row["removal_sampling_frame_size"] for row in removal_rows}
    probabilities = {row["removal_selection_probability"] for row in removal_rows}
    require(
        len(frame_sizes) == 1 and len(probabilities) == 1,
        "REMOVAL_FRAME_CONFLICT",
        "Step 5a frame metadata is inconsistent",
    )

    cross_rows = [row for row in rows if row["has_track_5b"]]
    cross_valid = [row for row in cross_rows if row["judge_status"] == "valid"]
    cross = [row for row in cross_valid if row["cross_group_outcome"] in {"discovered_candidate_fn", "hard_negative"}]
    cross_positive = sum(row["cross_group_outcome"] == "discovered_candidate_fn" for row in cross)
    retriever_breakdown = {}
    for category in ("lexical_only", "semantic_only", "both_or_overlap"):
        selected = [row for row in cross_rows if row["retriever_category"] == category]
        selected_resolved = [
            row for row in selected if row["cross_group_outcome"] in {"discovered_candidate_fn", "hard_negative"}
        ]
        positives = sum(row["cross_group_outcome"] == "discovered_candidate_fn" for row in selected_resolved)
        retriever_breakdown[category] = {
            "selected": len(selected),
            "resolved": len(selected_resolved),
            "judged_duplicate_yes": positives,
            "positive_yield": _rate(positives, len(selected_resolved)),
            "positive_contribution": _rate(positives, cross_positive),
        }
    judge_stage = next(marker for marker in stage_markers if marker["step"] == 6)
    judge_stage_counts = judge_stage["counts"]
    metrics = {
        "schema_version": "dedup-v0-metrics-v1",
        "judge": {
            "requested": requested_judge_pairs,
            "schema_valid": len(valid),
            "errors": requested_judge_pairs - len(valid),
            "completion_rate": _rate(len(valid), requested_judge_pairs),
            "resolved": len(resolved),
            "unresolved": len(valid) - len(resolved),
            "resolution_rate": _rate(len(resolved), len(valid)),
            "retried": sum(int(row["judge_attempts"]) > 1 for row in rows),
            "deterministically_repaired_results": judge_stage_counts.get("deterministically_repaired_results", 0),
            "deterministic_repair_events": judge_stage_counts.get("deterministic_repair_events", 0),
        },
        "track_5a_removal_frame": {
            "sampling_frame_size": next(iter(frame_sizes)),
            "inclusion_probability": next(iter(probabilities)),
            "requested": len(removal_rows),
            "schema_valid": len(removal_valid),
            "resolved": removal_total,
            "unresolved": len(removal_valid) - removal_total,
            "judge_errors": len(removal_rows) - len(removal_valid),
            "safe": safe,
            "wrong": removal_total - safe,
            "removal_precision": _rate(safe, removal_total),
            "wrong_removal_rate": _rate(removal_total - safe, removal_total),
            "wilson_95_ci": [ci_low, ci_high],
        },
        "track_5b_candidate_pool": {
            "requested": len(cross_rows),
            "schema_valid": len(cross_valid),
            "resolved": len(cross),
            "unresolved": len(cross_valid) - len(cross),
            "judge_errors": len(cross_rows) - len(cross_valid),
            "judged_duplicate_yes": cross_positive,
            "positive_yield": _rate(cross_positive, len(cross)),
            "metric_is_corpus_recall": False,
            "retriever_overlap_and_contribution": retriever_breakdown,
        },
        "prohibited_claims": [
            "corpus-wide recall",
            "complete cluster precision/recall/F1",
            "pooled track 5a and track 5b confusion matrix",
        ],
    }
    write_json_atomic(metrics_destination, metrics)
    slice_rows = []
    slice_specs = [
        ("track_5a", "group_size_bucket_low"),
        ("track_5a", "length_bucket_low"),
        ("track_5a", "token_length_ratio_bucket"),
        ("track_5a", "language_low"),
        ("track_5a", "same_hostname"),
        ("track_5b", "retriever_category"),
        ("track_5b", "length_bucket_low"),
        ("track_5b", "token_length_ratio_bucket"),
        ("track_5b", "language_low"),
        ("track_5b", "same_hostname"),
        ("judge", "relation_type"),
        ("judge", "material_difference"),
        ("judge", "fuzzy_scope"),
    ]
    for frame, field in slice_specs:
        frame_rows = [
            row
            for row in rows
            if frame == "judge"
            or (frame == "track_5a" and row["has_track_5a"])
            or (frame == "track_5b" and row["has_track_5b"])
        ]
        for value in sorted({str(row.get(field)) for row in frame_rows}):
            selected = [row for row in frame_rows if str(row.get(field)) == value]
            resolved_selected = [row for row in selected if row["judge_status"] == "valid"]
            slice_rows.append(
                {
                    "frame": frame,
                    "slice_field": field,
                    "slice_value": value,
                    "pairs": len(selected),
                    "schema_valid": len(resolved_selected),
                    "positive_or_safe": sum(
                        row.get("removal_outcome") == "safe_removal"
                        or row.get("cross_group_outcome") == "discovered_candidate_fn"
                        for row in resolved_selected
                    ),
                }
            )
    _write_csv(
        slices_destination,
        slice_rows,
        ["frame", "slice_field", "slice_value", "pairs", "schema_valid", "positive_or_safe"],
    )
    accounting_rows = [
        {
            "step": marker["step"],
            "name": marker["name"],
            "status": marker["status"],
            "counts_json": json.dumps(marker.get("counts", {}), sort_keys=True, separators=(",", ":")),
        }
        for marker in stage_markers
    ]
    _write_csv(accounting_destination, accounting_rows, ["step", "name", "status", "counts_json"])
    return metrics
