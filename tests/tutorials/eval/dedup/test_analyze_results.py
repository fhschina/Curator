# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision-layer selection in 7_analyze_results.py."""

import pandas as pd
import pytest
from policy_cases import load_module


def test_all_layer_sources_read_saved_values_without_correction():
    row = {
        "pair_type": "keeper_removed",
        "expected_duplicate": True,
        "pair_semantic_judgment": {"relation_type": {"score": "near_surface"}},
        "coverage_decision": {"relation_type": "near_surface"},
        "final_decision": {"relation_type": "related_non_duplicate"},
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
