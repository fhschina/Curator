# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.analysis.coverage_routing_audit import audit_inputs
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_witness import _case


def test_all_pairs_remain_in_population_and_positive_rows_are_not_fake_predictions():
    no, _, no_payload = _case()
    no["span_shared_basis"]["score"] = "none"
    exact, _, exact_payload = _case("Same text.", "Same text.")
    positive, _, positive_payload = _case()
    mains = {"no": no, "exact": exact, "positive": positive}
    inputs = [
        {"canonical_pair_id": pid, "payload": payload}
        for pid, payload in (("no", no_payload), ("exact", exact_payload), ("positive", positive_payload))
    ]
    saved = deepcopy((inputs, mains))
    result = audit_inputs(inputs, mains)
    assert result["input_pairs"] == result["retained_output_population"] == 3
    assert result["owned_branches_checked"] == 2
    assert result["requires_fresh_coverage"] == 1
    assert "public_output" not in next(r for r in result["rows"] if r["canonical_pair_id"] == "positive")
    assert (inputs, mains) == saved


def test_missing_main_is_not_silently_removed_from_population():
    with pytest.raises(DedupEvaluationError, match="each original input"):
        audit_inputs([{"canonical_pair_id": "p", "payload": {}}], {})
