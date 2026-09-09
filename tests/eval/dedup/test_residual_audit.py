# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from eval.dedup.analysis.residual_audit import (
    CONTAINMENT_IDS,
    PROTECTED_IDS,
    apply_v06211_gates,
    compare_residual_runs,
    render_residual_report,
    summarize_residual,
)
from eval.dedup.validation import DedupEvaluationError


def _labels() -> list[dict]:
    return [
        {
            "review_id": name,
            "canonical_pair_id": name,
            "human_same_duplicate_group": group,
            "human_a_can_replace_b": a,
            "human_b_can_replace_a": b,
            "human_relation_type": relation,
            "human_material_difference": material,
        }
        for name, group, a, b, relation, material in (
            ("protected", "YES", "YES", "YES", "NEAR_SURFACE", "NONE"),
            ("containment", "YES", "NO", "YES", "CONTAINMENT", "MAJOR"),
            ("negative", "NO", "NO", "NO", "RELATED_NON_DUPLICATE", "MAJOR"),
        )
    ]


def _predictions() -> list[dict]:
    return [
        {
            "canonical_pair_id": row["canonical_pair_id"],
            "confidence_tier": "MEDIUM",
            **{key.removeprefix("human_"): value for key, value in row.items() if key.startswith("human_")},
        }
        for row in _labels()
    ]


def _summary(predictions: list[dict], *, retried: int = 0, guard: str = "protected") -> dict:
    return summarize_residual(
        _labels(),
        predictions,
        {"requested": 3, "valid": len(predictions), "errors": 3 - len(predictions), "retried": retried},
        protected_ids=frozenset({guard}),
        containment_ids=frozenset({"containment"}),
        negative_ids=frozenset({"negative"}),
    )


def test_residual_gates_require_direction_not_merely_group_agreement() -> None:
    predictions = _predictions()
    predictions[0]["a_can_replace_b"] = "NO"
    result = _summary(predictions)
    assert result["checks"]["protected_duplicates"] is True
    assert result["checks"]["protected_primary_tuples"] is False
    assert result["error_ids"]["direction_only"] == ["protected"]
    assert result["passed"] is False


def test_missing_prediction_is_an_abstention_and_fails_completion_and_containment() -> None:
    predictions = [p for p in _predictions() if p["canonical_pair_id"] != "containment"]
    result = _summary(predictions)
    assert result["unweighted"]["duplicate_recall"] == 0.5
    assert result["checks"]["containment_direction_exact"] is False
    assert result["checks"]["schema_completion"] is False
    assert result["checks"]["terminal_errors"] is False
    assert result["error_ids"]["missing_results"] == ["containment"]


def test_unresolved_diagnostic_negative_cannot_pass_negative_guard() -> None:
    predictions = _predictions()
    predictions[-1]["same_duplicate_group"] = "UNRESOLVED"
    result = _summary(predictions)
    assert result["diagnostic_negative_violations"] == ["negative"]
    assert result["checks"]["diagnostic_negatives"] is False


def test_residual_guard_omission_fails_instead_of_shrinking_denominator() -> None:
    with pytest.raises(DedupEvaluationError):
        _summary(_predictions(), guard="missing")


def test_duplicate_prediction_cannot_inflate_completion() -> None:
    predictions = _predictions()
    with pytest.raises(DedupEvaluationError):
        _summary([*predictions, predictions[0]])


def test_residual_report_exposes_abstentions_and_does_not_claim_release() -> None:
    result = _summary(_predictions())
    assert result["passed"] is True
    summary = {
        "baseline": result,
        "candidate": result,
        "passed": True,
        "provenance": {
            "baseline": {"run_root": "old"},
            "candidate": {"run_root": "new", "prompt_version": "candidate"},
        },
        "selection_rule": "Passing is not release approval.",
        "interpretation": "Development diagnostics only.",
    }
    report = render_residual_report(summary)
    assert "UNRESOLVED reference duplicates count as recall misses" in report
    assert "not release approval" in report
    assert "MEDIUM | 3 | 100.00%" in report
    assert _summary(_predictions(), retried=1)["checks"]["retry_rate"] is False


def test_v06211_gates_reject_containment_regression_even_if_v06210_gate_passes() -> None:
    candidate = _summary(_predictions())
    candidate["containment_over_group"] = 2
    summary = {"baseline": _summary(_predictions()), "candidate": candidate}
    assert candidate["checks"]["containment_over_group"] is True
    apply_v06211_gates(summary)
    assert summary["gate_profile"] == "v06211"
    assert candidate["checks"]["containment_over_group"] is False
    assert summary["passed"] is False


def test_residual_file_comparison_verifies_results_and_equal_payloads(tmp_path: Path) -> None:
    negatives = {f"negative-{index}" for index in range(23)}
    labels = []
    for name in sorted(PROTECTED_IDS | CONTAINMENT_IDS | negatives):
        template = _labels()[0 if name in PROTECTED_IDS else 1 if name in CONTAINMENT_IDS else 2]
        labels.append({**template, "review_id": name, "canonical_pair_id": name})
    labels_path, subset_path, negatives_path = [tmp_path / f"{name}.csv" for name in ("labels", "subset", "negatives")]
    for path, rows in (
        (labels_path, labels),
        (subset_path, [{"canonical_pair_id": row["canonical_pair_id"]} for row in labels]),
        (negatives_path, [{"review_id": name, "same_duplicate_group": "NO"} for name in sorted(negatives)]),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    predictions = [
        {
            "canonical_pair_id": row["canonical_pair_id"],
            "confidence_tier": "MEDIUM",
            **{k.removeprefix("human_"): v for k, v in row.items() if k.startswith("human_")},
        }
        for row in labels
    ]
    roots = [tmp_path / "baseline", tmp_path / "candidate"]
    manifest = {
        "source_pair_ids_sha256": "same-pairs",
        "payload_membership_sha256": "same-payloads",
        "judge_contract_digest": "contract",
        "settings": {
            "prompt_version": "test",
            "hub_model": "test",
            "logical_model": "test",
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 4096,
            "max_parallel_requests": 64,
            "max_visible_tokens": 20_000,
            "window_tokens": 4096,
            "window_overlap_tokens": 512,
        },
    }
    for root in roots:
        (root / "data").mkdir(parents=True)
        results_path = root / "data" / "judge_results.jsonl"
        results_path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
        (root / "run_manifest.json").write_text(json.dumps(manifest))
        (root / "run_complete.json").write_text(
            json.dumps(
                {
                    "requested": len(labels),
                    "valid": len(labels),
                    "errors": 0,
                    "retried": 0,
                    "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
                }
            )
        )
    args = {
        "labels_path": labels_path,
        "subset_path": subset_path,
        "negative_review_paths": [negatives_path],
        "baseline_root": roots[0],
        "candidate_root": roots[1],
        "expected_pairs": len(labels),
    }
    assert compare_residual_runs(**args)["passed"] is True
    candidate_path = roots[1] / "run_manifest.json"
    manifest["payload_membership_sha256"] = "different-payloads"
    candidate_path.write_text(json.dumps(manifest))
    with pytest.raises(DedupEvaluationError, match="RESIDUAL_EXECUTION_MISMATCH"):
        compare_residual_runs(**args)
    manifest["payload_membership_sha256"] = "same-payloads"
    candidate_path.write_text(json.dumps(manifest))
    (roots[1] / "data" / "judge_results.jsonl").write_text("{}\n")
    with pytest.raises(DedupEvaluationError, match="RESIDUAL_RESULT_CHANGED"):
        compare_residual_runs(**args)
