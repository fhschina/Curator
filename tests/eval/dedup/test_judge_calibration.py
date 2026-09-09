# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

from eval.dedup.analysis.judge_calibration import (
    compare_development_runs,
    evaluate_development_gates,
    evaluate_holdout_gates,
    evaluate_predictions,
)


def _label(  # noqa: PLR0913
    pair_id: str,
    same: str,
    *,
    reason: str,
    population: int = 1,
    sample: int = 1,
    relation: str = "RELATED_NON_DUPLICATE",
) -> dict:
    direction = "YES" if same == "YES" else "NO"
    return {
        "canonical_pair_id": pair_id,
        "human_same_duplicate_group": same,
        "human_a_can_replace_b": direction,
        "human_b_can_replace_a": direction,
        "human_relation_type": relation,
        "human_material_difference": "NONE" if same == "YES" else "MAJOR",
        "human_reason_code": reason,
        "stratum_population_n": population,
        "stratum_sample_n": sample,
    }


def _prediction(label: dict, *, same: str | None = None, tier: str = "HIGH") -> dict:
    predicted_same = same or label["human_same_duplicate_group"]
    direction = "YES" if predicted_same == "YES" else "NO"
    return {
        "canonical_pair_id": label["canonical_pair_id"],
        "same_duplicate_group": predicted_same,
        "a_can_replace_b": direction,
        "b_can_replace_a": direction,
        "relation_type": label["human_relation_type"],
        "material_difference": label["human_material_difference"],
        "confidence_tier": tier,
    }


def test_calibration_reports_weighted_metrics_cohorts_and_tier_accuracy() -> None:
    labels = [
        _label("tp", "YES", reason="translation", population=9),
        _label("fn", "YES", reason="meaningful_addition"),
        _label("fp", "NO", reason="identity_slot"),
        _label("tn", "NO", reason="identity_slot"),
    ]
    predictions = [
        _prediction(labels[0], tier="HIGH"),
        _prediction(labels[1], same="NO", tier="MEDIUM"),
        _prediction(labels[2], same="YES", tier="LOW"),
        _prediction(labels[3], tier="HIGH"),
    ]

    metrics = evaluate_predictions(labels, predictions)

    assert metrics["unweighted"]["duplicate_precision"] == 0.5
    assert metrics["unweighted"]["duplicate_recall"] == 0.5
    assert metrics["weighted"]["duplicate_recall"] == 0.9
    assert metrics["over_group"] == 1
    assert metrics["under_group"] == 1
    assert metrics["confidence_tiers"]["HIGH"]["duplicate_group_accuracy"] == 1.0
    assert metrics["cohorts"]["identity_slot"]["rows"] == 2


def test_calibration_reports_semantic_ledger_accuracy_from_derived_reason_codes() -> None:
    labels = [
        _label("correct", "NO", reason="boilerplate_only", population=3),
        _label("over", "NO", reason="boilerplate_only"),
    ]
    predictions = [_prediction(labels[0]), _prediction(labels[1], same="YES")]
    predictions[0]["reason_codes"] = [
        "SHARED_CONTENT_BASIS:NONE",
        "SEMANTIC_LEDGER_RULE:NO_SHARED_SUBSTANTIVE_ANCHOR",
    ]
    predictions[1]["reason_codes"] = [
        "SHARED_CONTENT_BASIS:SUBSTANTIVE_ANCHOR",
        "SEMANTIC_LEDGER_RULE:NONEMPTY_SUBSTANTIVE_CONTAINMENT",
    ]

    metrics = evaluate_predictions(labels, predictions)

    assert metrics["semantic_ledger"]["SHARED_CONTENT_BASIS"]["NONE"] == {
        "rows": 1,
        "primary_decision_accuracy": 1.0,
        "weighted_primary_decision_accuracy": 1.0,
        "over_group": 0,
        "under_group": 0,
    }
    assert metrics["semantic_ledger"]["SEMANTIC_LEDGER_RULE"]["NONEMPTY_SUBSTANTIVE_CONTAINMENT"]["over_group"] == 1


def test_unresolved_duplicates_count_as_misses_in_recall_and_containment_gates() -> None:
    labels = [
        _label("resolved", "YES", reason="translation"),
        _label("abstained", "YES", reason="meaningful_addition", population=3, relation="CONTAINMENT"),
    ]
    predictions = [_prediction(labels[0]), _prediction(labels[1], same="UNRESOLVED", tier="LOW")]
    metrics = evaluate_predictions(labels, predictions)
    assert metrics["unweighted"]["duplicate_recall"] == 0.5
    assert metrics["weighted"]["duplicate_recall"] == 0.25
    assert metrics["under_group"] == metrics["containment_under_group"] == 1


def test_development_gate_checks_thresholds_and_protected_cohorts() -> None:
    cohort = {
        name: {"weighted_primary_decision_exact": 0.8}
        for name in ("identity_slot", "meaningful_addition", "translation")
    }
    baseline = {"cohorts": cohort}
    candidate = {
        "weighted": {
            "duplicate_precision": 0.76,
            "duplicate_recall": 0.75,
            "primary_decision_exact": 0.80,
        },
        "over_group": 60,
        "cohorts": cohort,
    }

    result = evaluate_development_gates(
        candidate=candidate,
        baseline=baseline,
        schema_completion_rate=1.0,
        retry_rate=0.009,
    )

    assert result["passed"] is True
    candidate["over_group"] = 67
    assert (
        evaluate_development_gates(
            candidate=candidate,
            baseline=baseline,
            schema_completion_rate=1.0,
            retry_rate=0.009,
        )["passed"]
        is False
    )


def test_holdout_gate_uses_representative_noninferiority_and_difficult_error_counts() -> None:
    candidate_all = {"unweighted": {"duplicate_precision": 0.76, "duplicate_recall": 0.77}}
    candidate_representative = {"unweighted": {"primary_decision_exact": 0.79}}
    baseline_representative = {"unweighted": {"primary_decision_exact": 0.81}}
    candidate_difficult = {"containment_over_group": 6, "containment_under_group": 10}
    baseline_difficult = {"containment_over_group": 10, "containment_under_group": 10}

    result = evaluate_holdout_gates(
        candidate_all=candidate_all,
        candidate_representative=candidate_representative,
        baseline_representative=baseline_representative,
        candidate_difficult=candidate_difficult,
        baseline_difficult=baseline_difficult,
    )

    assert result["passed"] is True
    assert result["difficult_over_group_reduction"] == 0.4


def _write_fake_run(path: Path, labels: list[dict], *, prompt_version: str) -> None:
    path.joinpath("data").mkdir(parents=True)
    stable_settings = {
        "hub_model": "model",
        "logical_model": "logical-model",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 4096,
        "max_visible_tokens": 20_000,
        "window_tokens": 4096,
        "window_overlap_tokens": 512,
        "prompt_version": prompt_version,
    }
    path.joinpath("run_manifest.json").write_text(
        json.dumps(
            {
                "settings": stable_settings,
                "source_pair_ids_sha256": "same-pairs",
                "payload_membership_sha256": "same-payloads",
            }
        )
    )
    path.joinpath("run_complete.json").write_text(
        json.dumps({"requested": len(labels), "valid": len(labels), "retried": 0})
    )
    path.joinpath("data", "judge_results.jsonl").write_text(
        "".join(json.dumps(_prediction(label)) + "\n" for label in labels)
    )


def test_development_comparison_requires_equal_execution_and_applies_gates(tmp_path: Path) -> None:
    labels = [
        _label("translation", "YES", reason="translation", relation="NEAR_SURFACE"),
        _label("addition", "YES", reason="meaningful_addition", relation="CONTAINMENT"),
        _label("identity", "NO", reason="identity_slot"),
    ]
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_fake_run(baseline, labels, prompt_version="baseline")
    _write_fake_run(candidate, labels, prompt_version="candidate")

    result = compare_development_runs(
        labels=labels,
        baseline_run_root=baseline,
        candidate_run_roots={"final": candidate},
    )

    assert result["variants"]["final"]["development_gates"]["passed"] is True
