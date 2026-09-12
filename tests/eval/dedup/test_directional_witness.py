# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging.context_coverage import adapt_context
from eval.dedup.judging.directional_witness import binding_issues, explain_invalid_binding
from eval.dedup.judging.local_ndd import _safe_retry_feedback
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_context_coverage import context_certificate
from tests.eval.dedup.test_context_experiment import fixture


def invalid_mirror():
    value, payload = context_certificate(reverse=True)
    value["a_meaning_in_b"].update(source_span_id="S001", counterpart_span_id="A001")
    value["hard_conflict"].update(a_span_id="S001", b_span_id="S001")
    return value, payload


def test_mirrored_failure_gets_specific_complete_safe_feedback_without_rewriting_it():
    value, payload = invalid_mirror()
    snapshot = deepcopy((value, payload))
    with pytest.raises(DedupEvaluationError, match="CONTEXT_DIFFERENCE_REQUIRED"):
        adapt_context(value, payload)
    with pytest.raises(DedupEvaluationError, match="DIRECTIONAL_WITNESS_BINDING") as error:
        explain_invalid_binding(value, payload)
    feedback = _safe_retry_feedback(error.value)
    issues = feedback["details"]["expected_fields"]
    opposite = next(i for i in issues if i["field"] == "a_meaning_in_b.counterpart_span_id")
    assert opposite["allowed_single_ids"] == ["S001", "S002"]
    assert "document B" in opposite["rule"]
    assert any(
        i["field"] == "hard_conflict" and i["unique_choices_by_document"] == {"A": ["A001"], "B": []} for i in issues
    )
    assert all(
        "A001" not in i.get("allowed_single_ids", []) for i in issues if i["field"] == "hard_conflict.b_span_id"
    )
    assert "a_meaning_in_b.counterpart_span_id" in encode_repair_feedback(feedback)
    assert (value, payload) == snapshot


@pytest.mark.parametrize("reverse", [False, True])
def test_valid_bilateral_context_and_whole_inventory_keep_original_public_output(reverse):
    value, payload = context_certificate(reverse)
    snapshot = deepcopy((value, payload, adapt_context(value, payload)))
    assert binding_issues(value, payload) == []
    assert explain_invalid_binding(value, payload) is None
    assert (value, payload, adapt_context(value, payload)) == snapshot


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_valid_v5_branches_are_not_made_stricter(stage):
    packets, values, _ = fixture(stage)
    for packet in packets:
        value, payload = values[packet["canonical_pair_id"]], packet["payload"]
        snapshot = deepcopy((value, payload))
        before = adapt_context(value, payload)
        explain_invalid_binding(value, payload)
        assert adapt_context(value, payload) == before
        assert (value, payload) == snapshot


def test_schema_and_translation_errors_remain_errors_without_fake_binding_advice():
    packets, values, _ = fixture("main")
    payload = next(p["payload"] for p in packets if p["canonical_pair_id"] == "uncovered")
    value = values["uncovered"]
    value["translation_status"] = "COMPLETE_FAITHFUL"
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_TRANSLATION"):
        explain_invalid_binding(value, payload)
    value["b_meaning_in_a"]["harmless_unique_ids"] = None
    with pytest.raises(DedupEvaluationError, match="CONTEXT_CONTRACT_INVALID"):
        explain_invalid_binding(value, payload)
