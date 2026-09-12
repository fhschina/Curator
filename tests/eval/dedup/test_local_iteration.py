# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.analysis.local_iteration import local_gates, validate_projection
from eval.dedup.validation import DedupEvaluationError


def _label():
    return {
        "canonical_pair_id": "pair",
        "review_id": "review",
        "human_same_duplicate_group": "YES",
        "human_a_can_replace_b": "YES",
        "human_b_can_replace_a": "NO",
        "human_relation_type": "CONTAINMENT",
        "human_material_difference": "MAJOR",
        "human_reason_code": "meaningful_addition",
        "stratum_population_n": "100",
        "stratum_sample_n": "10",
    }


def test_projection_keeps_stratum_weights_and_all_scored_fields():
    original = _label()
    original["human_fuzzy_scope"] = "historical_unused_axis"
    validate_projection([_label()], [original])
    for field in ("stratum_population_n", "human_b_can_replace_a", "human_reason_code"):
        broken = _label()
        broken[field] = "1" if field.startswith("stratum") else "YES"
        with pytest.raises(DedupEvaluationError, match="LOCAL_REFERENCE_CHANGED"):
            validate_projection([broken], [original])
    broken = _label()
    broken.pop("stratum_population_n")
    broken.pop("stratum_sample_n")
    broken["sample_weight"] = ""
    with pytest.raises(DedupEvaluationError, match="LOCAL_REFERENCE_CHANGED"):
        validate_projection([broken], [original])


@pytest.mark.parametrize("failure", ["negative", "benign", "containment", "exact", "translation", "retry", "terminal"])
def test_local_gate_never_substitutes_aggregate_gain_for_a_guard_regression(failure):
    metrics = {"cohorts": {"translation": {"weighted_primary_decision_exact": 0.90}}}
    candidate = deepcopy(metrics)
    guards = {
        key: {"passed": True}
        for key in ("identical", "true_containment", "baseline_negative_floor", "baseline_benign_floor")
    }
    operations = {"requested": 258, "valid": 258, "errors": 0, "retried": 2}
    assert local_gates(candidate, metrics, guards, operations, 258)["passed"]
    keys = {
        "negative": "baseline_negative_floor",
        "benign": "baseline_benign_floor",
        "containment": "true_containment",
        "exact": "identical",
    }
    if failure in keys:
        guards[keys[failure]]["passed"] = False
    elif failure == "translation":
        candidate["cohorts"]["translation"]["weighted_primary_decision_exact"] = 0.86
    elif failure == "retry":
        operations["retried"] = 3
    else:
        operations["errors"] = 1
        operations["valid"] = 257
    assert not local_gates(candidate, metrics, guards, operations, 258)["passed"]


def test_additional_critic_guards_cannot_weaken_existing_gate():
    metrics = {"cohorts": {"translation": {"weighted_primary_decision_exact": 1.0}}}
    guards = {
        key: {"passed": True}
        for key in ("identical", "true_containment", "baseline_negative_floor", "baseline_benign_floor")
    }
    operations = {"requested": 258, "valid": 258, "errors": 0, "retried": 0}
    guards["additional_critic_repair_guards"] = {"passed": False}
    assert not local_gates(metrics, metrics, guards, operations, 258)["passed"]
    guards["additional_critic_repair_guards"]["passed"] = True
    guards["baseline_negative_floor"]["passed"] = False
    assert not local_gates(metrics, metrics, guards, operations, 258)["passed"]
