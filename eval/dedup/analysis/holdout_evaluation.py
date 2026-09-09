# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate the frozen V0.6.2 holdout once and emit a full-run approval on success."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Final

from eval.dedup.analysis.judge_calibration import evaluate_holdout_gates, evaluate_predictions
from eval.dedup.config import V062_RELEASE_PROMPT_VERSIONS
from eval.dedup.validation import require, sha256_file, write_json_atomic

RELEASE_APPROVAL_SCHEMA: Final = "dedup-v062-release-approval-v1"
HOLDOUT_REPORT_SCHEMA: Final = "dedup-v062-holdout-evaluation-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "HOLDOUT_INPUT_INVALID", "expected a JSON object", path=str(path))
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _canonical_labels(
    labels: list[dict[str, str]], private_manifest: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    by_qa = {str(row["qa_pair_id"]): row for row in private_manifest}
    require(len(by_qa) == len(private_manifest) == 400, "HOLDOUT_MANIFEST_INVALID", "holdout must contain 400 rows")
    require(len(labels) == 400, "HOLDOUT_LABELS_INVALID", "holdout labels must contain exactly 400 rows")
    qa_ids = [str(row.get("qa_pair_id", "")) for row in labels]
    require(
        len(set(qa_ids)) == 400 and set(qa_ids) == set(by_qa),
        "HOLDOUT_LABELS_INVALID",
        "holdout labels must join one-to-one with the private manifest",
    )
    primary_fields = ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a")
    relation_values = {
        "EXACT",
        "CANONICAL_EXACT",
        "NEAR_SURFACE",
        "CONTAINMENT",
        "VERSION_RELATED",
        "RELATED_NON_DUPLICATE",
        "UNRELATED",
        "UNRESOLVED",
    }
    material_values = {"NONE", "MINOR", "MAJOR", "UNRESOLVED"}
    canonical = []
    for row in labels:
        qa_id = str(row["qa_pair_id"])
        require(
            all(row.get(field) in {"YES", "NO", "UNRESOLVED"} for field in primary_fields),
            "HOLDOUT_LABELS_INVALID",
            "every holdout primary decision must be adjudicated",
            qa_pair_id=qa_id,
        )
        require(
            row.get("relation_type") in relation_values and row.get("material_difference") in material_values,
            "HOLDOUT_LABELS_INVALID",
            "holdout relation and material labels are required for containment safety gates",
            qa_pair_id=qa_id,
        )
        private = by_qa[qa_id]
        canonical.append(
            {
                "canonical_pair_id": private["canonical_pair_id"],
                **{f"human_{field}": row[field] for field in primary_fields},
                "human_relation_type": row["relation_type"],
                "human_material_difference": row["material_difference"],
                "human_reason_code": "holdout",
                "holdout_split": private["split"],
            }
        )
    representative = {str(row["canonical_pair_id"]) for row in private_manifest if row["split"] == "representative"}
    difficult = {str(row["canonical_pair_id"]) for row in private_manifest if row["split"] == "difficult"}
    require(
        len(representative) == len(difficult) == 200 and not representative & difficult,
        "HOLDOUT_MANIFEST_INVALID",
        "holdout must contain disjoint 200-pair splits",
    )
    return canonical, representative, difficult


def evaluate_frozen_holdout(
    *,
    labels: list[dict[str, str]],
    private_manifest: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical, representative_ids, difficult_ids = _canonical_labels(labels, private_manifest)

    def label_subset(pair_ids: set[str]) -> list[dict[str, Any]]:
        return [row for row in canonical if str(row["canonical_pair_id"]) in pair_ids]

    candidate_all = evaluate_predictions(canonical, candidate_predictions)
    candidate_representative = evaluate_predictions(label_subset(representative_ids), candidate_predictions)
    baseline_representative = evaluate_predictions(label_subset(representative_ids), baseline_predictions)
    candidate_difficult = evaluate_predictions(label_subset(difficult_ids), candidate_predictions)
    baseline_difficult = evaluate_predictions(label_subset(difficult_ids), baseline_predictions)
    gates = evaluate_holdout_gates(
        candidate_all=candidate_all,
        candidate_representative=candidate_representative,
        baseline_representative=baseline_representative,
        candidate_difficult=candidate_difficult,
        baseline_difficult=baseline_difficult,
    )
    return {
        "schema_version": HOLDOUT_REPORT_SCHEMA,
        "candidate_all": candidate_all,
        "candidate_representative": candidate_representative,
        "baseline_representative": baseline_representative,
        "candidate_difficult": candidate_difficult,
        "baseline_difficult": baseline_difficult,
        "holdout_gates": gates,
        "version_selection_axis": "human_primary_decisions",
        "excluded_selection_axes": ["deepseek_agreement", "proxy_total_score", "sut_outcome", "minhash_diagnostics"],
    }


def publish_holdout_evaluation(
    *,
    labels_path: Path,
    private_manifest_path: Path,
    baseline_results_path: Path,
    candidate_run_root: Path,
    report_destination: Path,
    approval_destination: Path,
) -> dict[str, Any]:
    """Consume one frozen holdout artifact and write approval only when every gate passes."""

    require(
        not report_destination.exists() and not approval_destination.exists(),
        "HOLDOUT_ALREADY_CONSUMED",
        "holdout report or approval destination already exists",
    )
    candidate_manifest = _read_json(candidate_run_root / "run_manifest.json")
    require(
        candidate_manifest["settings"]["prompt_version"] in V062_RELEASE_PROMPT_VERSIONS,
        "HOLDOUT_CANDIDATE_INVALID",
        "only the final V0.6.2 prompt may consume the holdout",
    )
    private_manifest = _read_jsonl(private_manifest_path)
    selected_ids = {str(row["canonical_pair_id"]) for row in private_manifest}
    candidate_predictions = _read_jsonl(candidate_run_root / "data" / "judge_results.jsonl")
    require(
        {str(row["canonical_pair_id"]) for row in candidate_predictions} == selected_ids,
        "HOLDOUT_CANDIDATE_INVALID",
        "candidate results must match the frozen holdout exactly",
    )
    report = evaluate_frozen_holdout(
        labels=_read_labels(labels_path),
        private_manifest=private_manifest,
        baseline_predictions=_read_jsonl(baseline_results_path),
        candidate_predictions=candidate_predictions,
    )
    report.update(
        {
            "holdout_manifest_sha256": sha256_file(private_manifest_path),
            "labels_sha256": sha256_file(labels_path),
            "candidate_run_root": str(candidate_run_root),
            "candidate_judge_contract_digest": candidate_manifest["judge_contract_digest"],
        }
    )
    prompt_version = candidate_manifest["settings"]["prompt_version"]
    write_json_atomic(report_destination, report)
    if report["holdout_gates"]["passed"]:
        write_json_atomic(
            approval_destination,
            {
                "schema_version": RELEASE_APPROVAL_SCHEMA,
                "status": "PASSED",
                "prompt_version": prompt_version,
                "judge_contract_digest": candidate_manifest["judge_contract_digest"],
                "holdout_manifest_sha256": report["holdout_manifest_sha256"],
                "holdout_report_sha256": sha256_file(report_destination),
                "holdout_evaluated_once": True,
                "gates": report["holdout_gates"],
            },
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    report = publish_holdout_evaluation(
        labels_path=args.labels.resolve(),
        private_manifest_path=args.private_manifest.resolve(),
        baseline_results_path=args.baseline_results.resolve(),
        candidate_run_root=args.candidate_run_root.resolve(),
        report_destination=args.report.resolve(),
        approval_destination=args.approval.resolve(),
    )
    return 0 if report["holdout_gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
