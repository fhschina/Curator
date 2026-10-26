# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision-layer selection in 7_analyze_results.py."""

import pandas as pd
import pytest
from policy_cases import load_module


def raw_decision():
    return {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
    }


def test_all_layer_sources_read_saved_values_without_correction():
    value = raw_decision()
    row = {
        "pair_type": "keeper_removed",
        "expected_duplicate": True,
        "pair_semantic_judgment": {k: {"score": v} for k, v in value.items()},
        "coverage_decision": value,
        "final_decision": {
            **value,
            "a_can_replace_b": "no",
            "b_can_replace_a": "no",
            "relation_type": "related_non_duplicate",
            "material_difference": "major",
            "primary_material_difference": "other_material",
        },
    }
    module = load_module("7_analyze_results.py")
    assert module.annotate(pd.DataFrame([row]), decision_source="main").iloc[0].relation_type == "near_surface"
    assert module.annotate(pd.DataFrame([row]), decision_source="coverage").iloc[0].relation_type == "near_surface"
    assert (
        module.annotate(pd.DataFrame([row]), decision_source="final").iloc[0].relation_type == "related_non_duplicate"
    )
    with pytest.raises(ValueError, match="corrections"):
        module.annotate(pd.DataFrame([row]), decision_source="final", apply_main_corrections=True)
    with pytest.raises(KeyError, match="coverage_decision"):
        module.annotate(
            pd.DataFrame([{k: v for k, v in row.items() if k != "coverage_decision"}]), decision_source="coverage"
        )


def test_missing_and_unvalidated_outputs_are_not_normal_unresolved_or_rate_denominators():
    module = load_module("7_analyze_results.py")
    value = raw_decision()
    rows = [
        {"final_decision": value},
        {"final_decision": None},
        {"final_decision": {**value, "relation_type": "unknown"}},
        {"final_decision": value, "coverage_action": "UNVALIDATED_KEEP_MAIN"},
    ]
    frame = module.annotate(
        pd.DataFrame([{**row, "pair_type": "keeper_removed", "expected_duplicate": True} for row in rows]),
        decision_source="final",
    )
    assert list(frame.decision_status) == ["resolved", "missing", "invalid", "unvalidated"]
    summary = module.summarize(frame).iloc[0]
    assert summary.num_conclusive == 1
    assert summary.num_missing == summary.num_invalid == summary.num_unvalidated == 1
    assert summary.judge_says_not_duplicate_rate == 0
