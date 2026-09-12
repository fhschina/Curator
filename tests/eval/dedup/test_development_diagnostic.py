# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.dedup.analysis.development_diagnostic import (
    build_diagnostic,
    classify_primary_error,
    evaluate_primary_guards,
    load_run,
)
from eval.dedup.judging.schema_v3 import unresolved_judge_output_v3
from eval.dedup.validation import DedupEvaluationError, sha256_file


def _inputs() -> tuple:
    labels, predictions, payloads, metadata = [], [], [], []
    for index, (truth, predicted, weight) in enumerate(
        (
            ("NO", "YES", 2),
            ("YES", "NO", 3),
            ("YES", "UNRESOLVED", 5),
            ("YES", "YES", 7),
            ("NO", "NO", 11),
            ("YES", "YES", 13),
        )
    ):
        pair_id = f"p{index}"
        labels.append(
            {
                "review_id": f"H{index}",
                "canonical_pair_id": pair_id,
                "sample_weight": weight,
                "human_same_duplicate_group": truth,
                "human_a_can_replace_b": truth,
                "human_b_can_replace_a": truth,
                "human_relation_type": "NEAR_SURFACE" if truth == "YES" else "UNRELATED",
                "human_material_difference": "MINOR" if index == 5 else "NONE",
                "human_reason_code": "translation" if index == 5 else "boilerplate",
            }
        )
        predictions.append(
            {
                "canonical_pair_id": pair_id,
                "same_duplicate_group": predicted,
                "a_can_replace_b": "NO" if index == 3 else predicted,
                "b_can_replace_a": predicted,
                "relation_type": labels[-1]["human_relation_type"],
                "material_difference": "NONE",
                "reason_codes": [],
            }
        )
        payloads.append(
            {
                "canonical_pair_id": pair_id,
                "payload": {
                    "document_a": {"text": "A"},
                    "document_b": {"text": "B"},
                    "long_document_evidence": {"truncated": index == 2, "windows": []},
                },
            }
        )
        metadata.append(
            {"canonical_pair_id": pair_id, "has_track_5a": True, "language_low": "ENGLISH", "language_high": "ENGLISH"}
        )
    return labels, predictions, payloads, {"p0", "p2"}, metadata


def test_disjoint_errors_preserve_unresolved_recall_misses_and_weights() -> None:
    summary, rows = build_diagnostic(*_inputs())
    distribution = summary["primary_error_distribution"]
    assert distribution["error_types"] == {
        "CORRECT": 2,
        "DIRECTION_ONLY": 1,
        "OVER_GROUP": 1,
        "UNDER_GROUP_RESOLVED": 1,
        "UNRESOLVED": 1,
    }
    assert distribution["primary_errors"] == 4
    assert distribution["weighted_primary_errors"] == 17
    assert distribution["weighted_rows"] == 41
    assert summary["metrics"]["under_group"] == 2
    assert summary["metrics"]["unweighted"]["duplicate_recall"] == 0.5
    assert summary["taxonomy_only_ids"] == ["H5"]
    assert summary["eligible_for_promotion"] is False
    assert [r["review_id"] for r in rows][:2] == ["H5", "H4"]


def test_truncation_and_partition_denominators_are_not_inferred_from_nonempty_metadata() -> None:
    summary, _ = build_diagnostic(*_inputs())
    assert summary["cross_tabs"]["truncated"]["True"]["rows"] == 1
    assert summary["cross_tabs"]["truncated"]["False"]["rows"] == 5
    assert summary["partitions"]["original_residual_127"]["unweighted"]["rows"] == 2
    assert summary["partitions"]["development_complement_873"]["unweighted"]["rows"] == 4


def test_record_scope_and_retained_conflict_report_actual_errors_without_inventing_old_fields() -> None:
    inputs = list(_inputs())
    inputs[1][0]["reason_codes"] = ["RECORD_SCOPE:SAME_SPECIFIC_RECORD", "RETAINED_CONFLICT:NONE"]
    summary, rows = build_diagnostic(*inputs)
    scope = summary["cross_tabs"]["record_scope"]
    assert scope["SAME_SPECIFIC_RECORD"]["error_types"] == {"OVER_GROUP": 1}
    assert scope["MISSING"]["rows"] == 5
    assert summary["cross_tabs"]["retained_conflict"]["NONE"]["weighted_primary_errors"] == 2
    assert next(r for r in rows if r["review_id"] == "H0")["record_scope"] == "SAME_SPECIFIC_RECORD"


@pytest.mark.parametrize("source", [0, 1, 2, 4])
def test_duplicate_input_ids_are_rejected(source: int) -> None:
    inputs = list(_inputs())
    inputs[source].append(inputs[source][0])
    with pytest.raises(DedupEvaluationError, match="DIAGNOSTIC_DUPLICATE_ID"):
        build_diagnostic(*inputs)


def test_missing_prediction_cannot_shrink_the_error_denominator() -> None:
    inputs = list(_inputs())
    inputs[1].pop()
    with pytest.raises(DedupEvaluationError, match="DIAGNOSTIC_MEMBERSHIP"):
        build_diagnostic(*inputs)


@pytest.mark.parametrize("prediction", [None, "UNRESOLVED", "YES"])
def test_protected_negatives_fail_for_missing_abstaining_or_positive_predictions(prediction: str | None) -> None:
    keys = ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a")
    guards = [{"canonical_pair_id": "p", "review_id": "H1", "final": dict.fromkeys(keys, "NO")}]
    predictions = [] if prediction is None else [{"canonical_pair_id": "p", **dict.fromkeys(keys, prediction)}]
    check = evaluate_primary_guards(guards, predictions)
    assert check == {"expected": 1, "preserved": 0, "violations": ["H1"], "passed": False}
    assert evaluate_primary_guards(guards, [{"canonical_pair_id": "p", **dict.fromkeys(keys, "NO")}])["passed"]


def test_unresolved_reference_does_not_become_a_proven_direction_error() -> None:
    assert (
        classify_primary_error(
            {
                "human_same_duplicate_group": "UNRESOLVED",
                "human_a_can_replace_b": "UNRESOLVED",
                "human_b_can_replace_a": "UNRESOLVED",
            },
            {"same_duplicate_group": "YES", "a_can_replace_b": "YES", "b_can_replace_a": "YES"},
        )
        == "REFERENCE_UNRESOLVED"
    )


def _write_run(root: Path) -> None:
    (root / "data").mkdir()
    predictions = [{"canonical_pair_id": "pair", "judge_contract_digest": "contract", **unresolved_judge_output_v3()}]
    payloads = [
        {
            "canonical_pair_id": "pair",
            "payload": {
                "document_a": {"text": "A"},
                "document_b": {"text": "B"},
                "long_document_evidence": {"truncated": False, "windows": []},
            },
        }
    ]
    for name, rows in (("judge_results", predictions), ("judge_payloads", payloads)):
        (root / "data" / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_pair_count": 1,
                "judge_contract_digest": "contract",
                "judge_payloads_sha256": sha256_file(root / "data/judge_payloads.jsonl"),
            }
        )
    )
    (root / "run_complete.json").write_text(
        json.dumps(
            {"requested": 1, "valid": 1, "errors": 0, "results_sha256": sha256_file(root / "data/judge_results.jsonl")}
        )
    )


def test_run_loader_validates_actual_schema_and_frozen_artifact_hashes(tmp_path: Path) -> None:
    _write_run(tmp_path)
    assert len(load_run(tmp_path)[2]) == 1
    with (tmp_path / "data/judge_results.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(DedupEvaluationError, match="DIAGNOSTIC_ARTIFACT_CHANGED"):
        load_run(tmp_path)


def test_run_loader_checks_evidence_even_when_manifest_hashes_match(tmp_path: Path) -> None:
    _write_run(tmp_path)
    path = tmp_path / "data/judge_results.jsonl"
    row = json.loads(path.read_text())
    row.update(
        {
            "same_duplicate_group": "YES",
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "NEAR_SURFACE",
            "material_difference": "NONE",
            "primary_material_difference": "NONE",
            "dominant_overlap_source": "MAIN_CONTENT",
            "primary_risk_factor": "NONE",
            "confidence_tier": "MEDIUM",
        }
    )
    row["evidence"] = [
        {"side": "A", "start_char": 0, "end_char": 1, "quote": "X"},
        {"side": "B", "start_char": 0, "end_char": 1, "quote": "B"},
    ]
    path.write_text(json.dumps(row) + "\n")
    complete_path = tmp_path / "run_complete.json"
    complete = json.loads(complete_path.read_text())
    complete["results_sha256"] = sha256_file(path)
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(DedupEvaluationError, match="JUDGE_EVIDENCE_OFFSET_INVALID"):
        load_run(tmp_path)
