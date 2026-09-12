# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.analysis import retention_context_full1000 as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_retention_context_v1 import payload, review


def fixture():
    p = payload("Policy.\nProduct information.", "Policy.")
    positive = subject.pilot.candidate.adapt_main(review(p, "MAIN_ADDITION"), p)
    labels, results = [], []
    for i, reason in enumerate(("identity_slot", "meaningful_addition", "translation", "boilerplate_only")):
        labels.append(
            {
                "canonical_pair_id": str(i),
                "review_id": f"R{i}",
                "stratum_population_n": (i + 1) * 10,
                "stratum_sample_n": 10,
                "human_reason_code": reason,
                "comparison_scoring_eligibility": "INCLUDED",
                **{f"human_{k}": v for k, v in subject.adjudication.primary(positive).items()},
            }
        )
        results.append(
            {
                "canonical_pair_id": str(i),
                "review_id": f"R{i}",
                "status": "VALID",
                "public": deepcopy(positive),
                "components": {"main": deepcopy(positive), "coverage": deepcopy(positive)},
                "stages": [],
            }
        )
    histories = {
        v: [
            {"canonical_pair_id": r["canonical_pair_id"], **subject.adjudication.primary(r["public"])} for r in results
        ]
        for v in subject.HISTORICAL
    }
    return labels, histories, results


def test_all_three_versions_use_identical_weights_reference_and_only_explicit_mask():
    labels, histories, results = fixture()
    labels[-1]["comparison_scoring_eligibility"] = "EXCLUDED"
    before = deepcopy((labels, histories, results))
    out = subject.compare(labels, histories, results, {"3"})
    for score in out["scores"].values():
        assert score["original_rows"] == 4
        assert score["scored_rows"] == 3
        assert score["weighted"]["weight_total"] == 6
        assert score["weighted"]["primary_decision_exact"] == 1
        assert not score["taxonomy_scored"]
    assert len(out["ledger"]) == 4
    assert set(out["ledger"][-1]["errors"].values()) == {"EXCLUDED_NOT_SCORED"}
    assert (labels, histories, results) == before
    assert not out["release_eligible"]
    with pytest.raises(DedupEvaluationError):
        subject.compare(labels, histories, results, set())


def test_failure_sentinel_never_gets_credit_even_when_gold_is_unresolved_and_stages_are_preserved():
    labels, histories, results = fixture()
    labels[0].update({f"human_{k}": "UNRESOLVED" for k in subject.adjudication.primary(results[0]["public"])})
    results[0].update(
        status="ENGINEERING_FAILURE", error_code="EXAMPLE", public=subject.pilot.unresolved_judge_output_v4()
    )
    out = subject.compare(labels, histories, results, set())
    score = out["scores"][subject.VERSION]
    assert score["weighted"]["primary_decision_exact"] == pytest.approx(0.9)
    assert score["error_counts"]["ENGINEERING_MISSING_OUTPUT"] == 1
    assert out["ledger"][0]["errors"][subject.VERSION] == "ENGINEERING_MISSING_OUTPUT"
    assert not subject.projection(results[0], "main").get("metric_only_missing_output")
    assert subject.projection(results[0]).get("metric_only_missing_output")
    results[0]["components"] = {}
    assert subject.projection(results[0], "main")["metric_only_missing_output"]


def test_unresolved_prediction_keeps_positive_reference_in_recall_and_primary_denominators():
    labels, histories, results = fixture()
    results[0]["public"] = subject.pilot.unresolved_judge_output_v4()
    out = subject.compare(labels, histories, results, set())
    score = out["scores"][subject.VERSION]
    assert score["weighted"]["duplicate_recall"] == pytest.approx(0.9)
    assert score["unweighted"]["primary_decision_exact"] == pytest.approx(0.75)
    assert score["full_population_group_unresolved"] == 1
    assert out["confidence_tiers"]["LOW"]["unweighted"]["primary_decision_exact"] == 0
    assert out["paired_comparisons"][subject.HISTORICAL[0]]["regressed_pairs"] == 1


@pytest.mark.parametrize("damage", ["missing", "duplicate", "extra", "missing_baseline"])
def test_no_silent_population_intersection_or_dropped_baseline(damage):
    labels, histories, results = fixture()
    if damage == "missing":
        results.pop()
    elif damage == "duplicate":
        results.append(deepcopy(results[0]))
    elif damage == "extra":
        extra = deepcopy(results[0])
        extra["canonical_pair_id"] = "extra"
        results.append(extra)
    else:
        histories.pop(subject.HISTORICAL[0])
    with pytest.raises(DedupEvaluationError):
        subject.compare(labels, histories, results, set())


def test_transition_ledger_distinguishes_corrections_regressions_and_weight():
    labels, histories, results = fixture()
    histories[subject.HISTORICAL[0]][0].update(same_duplicate_group="NO", a_can_replace_b="NO", b_can_replace_a="NO")
    results[1]["public"] = subject.pilot.unresolved_judge_output_v4()
    out = subject.compare(labels, histories, results, set())
    delta = out["paired_comparisons"][subject.HISTORICAL[0]]
    assert delta["corrected_pairs"] == delta["regressed_pairs"] == 1
    assert delta["corrected_weight"] == 1
    assert delta["regressed_weight"] == 2
    assert not out["development_numeric_checks"]["protected_cohorts_vs_historical"][subject.HISTORICAL[0]][
        "meaningful_addition"
    ]


def test_no_existing_root_or_unbound_run_can_trigger_calls(tmp_path):
    with pytest.raises(DedupEvaluationError, match="CONTEXT_FULL_ROOT"):
        subject.prepare(tmp_path)
    with pytest.raises((FileNotFoundError, DedupEvaluationError)):
        subject.run(tmp_path, tmp_path / "never_read.env")
    with pytest.raises(RuntimeError, match="must never be called"):
        subject.unused_baseline_renderer({})
