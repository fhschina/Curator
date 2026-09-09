# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eval.dedup.analysis.holdout_evaluation import publish_holdout_evaluation
from eval.dedup.config import (
    HS_V062_PROMPT_VERSION,
    HS_V0621_PROMPT_VERSION,
    HS_V0622_PROMPT_VERSION,
    HS_V0623_PROMPT_VERSION,
    HS_V0624_PROMPT_VERSION,
    HS_V0625_PROMPT_VERSION,
)


def _prediction(label: dict[str, str], pair_id: str, *, same: str | None = None) -> dict[str, str]:
    predicted_same = same or label["same_duplicate_group"]
    direction = "YES" if predicted_same == "YES" else "NO"
    relation = (
        "CONTAINMENT" if label["same_duplicate_group"] == "NO" and predicted_same == "YES" else label["relation_type"]
    )
    return {
        "canonical_pair_id": pair_id,
        "same_duplicate_group": predicted_same,
        "a_can_replace_b": direction,
        "b_can_replace_a": direction,
        "relation_type": relation,
        "material_difference": label["material_difference"],
    }


@pytest.mark.parametrize(
    "prompt_version",
    [
        HS_V062_PROMPT_VERSION,
        HS_V0621_PROMPT_VERSION,
        HS_V0622_PROMPT_VERSION,
        HS_V0623_PROMPT_VERSION,
        HS_V0624_PROMPT_VERSION,
        HS_V0625_PROMPT_VERSION,
    ],
)
def test_passing_holdout_writes_release_approval_for_exact_candidate_contract(
    tmp_path: Path, prompt_version: str
) -> None:
    private_rows = []
    labels = []
    baseline = []
    candidate = []
    for index in range(400):
        pair_id = f"cp1_{index:04d}"
        qa_id = f"qa-{index:04d}"
        split = "representative" if index < 200 else "difficult"
        difficult_index = index - 200
        is_negative = split == "difficult" and difficult_index < 20
        is_containment = split == "difficult" and 20 <= difficult_index < 120
        label = {
            "qa_pair_id": qa_id,
            "same_duplicate_group": "NO" if is_negative else "YES",
            "a_can_replace_b": "NO" if is_negative else "YES",
            "b_can_replace_a": "NO" if is_negative else "YES",
            "relation_type": "UNRELATED" if is_negative else "CONTAINMENT" if is_containment else "NEAR_SURFACE",
            "material_difference": "MAJOR" if is_negative or is_containment else "NONE",
        }
        private_rows.append(
            {
                "qa_pair_id": qa_id,
                "canonical_pair_id": pair_id,
                "split": split,
                "review_mode": "SINGLE",
            }
        )
        labels.append(label)
        baseline_same = "YES" if split == "difficult" and difficult_index < 10 else None
        if split == "difficult" and 20 <= difficult_index < 30:
            baseline_same = "NO"
        candidate_same = "YES" if split == "difficult" and difficult_index < 6 else None
        if split == "difficult" and 20 <= difficult_index < 30:
            candidate_same = "NO"
        baseline.append(_prediction(label, pair_id, same=baseline_same))
        candidate.append(_prediction(label, pair_id, same=candidate_same))

    private_path = tmp_path / "private.jsonl"
    private_path.write_text("".join(json.dumps(row) + "\n" for row in private_rows))
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    baseline_path = tmp_path / "baseline.jsonl"
    baseline_path.write_text("".join(json.dumps(row) + "\n" for row in baseline))
    candidate_root = tmp_path / "candidate"
    candidate_root.joinpath("data").mkdir(parents=True)
    candidate_root.joinpath("run_manifest.json").write_text(
        json.dumps(
            {
                "settings": {"prompt_version": prompt_version},
                "judge_contract_digest": "contract-v062",
            }
        )
    )
    candidate_root.joinpath("data", "judge_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidate)
    )
    report_path = tmp_path / "holdout-report.json"
    approval_path = tmp_path / "release-approval.json"

    report = publish_holdout_evaluation(
        labels_path=labels_path,
        private_manifest_path=private_path,
        baseline_results_path=baseline_path,
        candidate_run_root=candidate_root,
        report_destination=report_path,
        approval_destination=approval_path,
    )

    assert report["holdout_gates"]["passed"] is True
    approval = json.loads(approval_path.read_text())
    assert approval["status"] == "PASSED"
    assert approval["prompt_version"] == prompt_version
    assert approval["judge_contract_digest"] == "contract-v062"
    assert approval["holdout_evaluated_once"] is True
