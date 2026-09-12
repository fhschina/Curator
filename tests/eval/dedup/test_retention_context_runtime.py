# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import retention_context_runtime as subject
from eval.dedup.judging import retention_v4_runtime as previous
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_retention_v4 import payload


def test_new_registry_contract_rubric_and_source_digest_are_isolated_from_previous_version():
    old = previous.specification()
    spec = subject.specification()
    assert spec["config"]["adapter"] == subject.adapter.__name__
    assert spec["config"]["model_response_contract"] == subject.adapter.CONTRACT
    assert spec["contract_digest"] != old["contract_digest"]
    assert not spec["release_eligible"]
    assert previous.specification() == old
    with pytest.raises(DedupEvaluationError):
        subject.specification(previous.VERSION)


def test_both_stages_render_new_field_semantics_without_changing_span_presentation_or_payload():
    p = payload("This claim is false: The service is free.", "The service is free.")
    before = deepcopy(p)
    main, critic = subject.messages(p), subject.messages(p, stage="critic")
    assert main[1] == critic[1]
    assert "context_conflict" in main[0]["content"]
    assert "a_addition_kind" in main[0]["content"]
    assert "only remove" in critic[0]["content"]
    assert (
        main[1]["content"].split("<semantic_diff")[1] == previous.messages(p)[1]["content"].split("<semantic_diff")[1]
    )
    assert p == before


@pytest.mark.parametrize("hidden", ["human_label", "gate_expected", "main_prediction", "exclusion", "review_id"])
def test_reference_and_selection_metadata_cannot_enter_candidate_prompt(hidden):
    p = payload("Text A.", "Text B.")
    p[hidden] = "NO"
    with pytest.raises(DedupEvaluationError):
        subject.messages(p)
