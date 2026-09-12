# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import critic_subject_binding as subject
from eval.dedup.judging.payload import _semantic_diff_packet
from eval.dedup.validation import DedupEvaluationError


def fixture(a="Agency Cedar is not liable for comments.", b="Agency Birch is not liable for comments."):
    payload = {
        "document_a": {"text": a},
        "document_b": {"text": b},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": _semantic_diff_packet(a, b, truncated=False),
    }
    main = {
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
        "relation_type": "NEAR_SURFACE",
        "material_difference": "NONE",
        "primary_material_difference": "NONE",
        "dominant_overlap_source": "MAIN_CONTENT",
        "primary_risk_factor": "TEMPLATE_SLOT_COLLISION",
        "confidence_tier": "MEDIUM",
        "reason_codes": [],
        "evidence": [
            {"side": side, "start_char": 0, "end_char": len(text), "quote": text}
            for side, text in (("A", a), ("B", b))
        ],
    }
    value = {
        "a_subject_span_id": "A001",
        "b_subject_span_id": "B001",
        "a_predicate_span_id": "S002",
        "b_predicate_span_id": "S002",
        "binding_type": "LIABILITY_PARTY",
        "target_relation": "DIFFERENT",
        "explanation": "Cedar versus Birch is the actual exempted party in the same comment liability assertion.",
    }
    return payload, main, value


def test_bound_party_veto_is_bilateral_exact_and_never_mutates_inputs():
    payload, main, value = fixture()
    before = deepcopy((payload, main, value))
    result, _ = subject.apply_review(main, payload, value)
    assert result["same_duplicate_group"] == "NO"
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"
    assert len(result["evidence"]) == 4
    subject.veto.validate_evidence_offsets(result, payload)
    assert before == (payload, main, value)


@pytest.mark.parametrize("relation", ["SAME", "UNCERTAIN", "NOT_APPLICABLE"])
def test_nonconflicts_preserve_fixed_coverage_without_inventing_directions(relation):
    payload, main, value = fixture()
    main.update(
        a_can_replace_b="NO",
        relation_type="CONTAINMENT",
        material_difference="MAJOR",
        primary_material_difference="MAIN_CONTENT_ADDITION_DELETION",
    )
    value.update(target_relation=relation, binding_type="NONE")
    for key in value:
        if key.endswith("span_id"):
            value[key] = ""
    assert subject.apply_review(main, payload, value)[0] == main


@pytest.mark.parametrize(
    "damage", ["wrong_side", "shared_subject", "missing_predicate", "invalid_uncited", "unknown_relation", "extra_key"]
)
def test_invalid_proof_cannot_silently_veto_or_be_repaired(damage):
    payload, main, value = fixture()
    if damage == "wrong_side":
        value["a_subject_span_id"] = "B001"
    elif damage == "shared_subject":
        value["a_subject_span_id"] = "S001"
    elif damage == "missing_predicate":
        value["b_predicate_span_id"] = ""
    elif damage == "invalid_uncited":
        payload["semantic_diff_evidence"]["spans"][0]["a_start_char"] = 99
    elif damage == "unknown_relation":
        value["target_relation"] = "YES"
    else:
        value["action"] = "REJECT_BOTH"
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(main, payload, value)


def test_single_sided_extension_exact_negative_and_truncation_bypass_specialist():
    for a, b in (("Full policy.", "Full policy. Product K9."), ("Same text.", "Same text.")):
        payload, main, _ = fixture(a, b)
        assert subject.apply_review(main, payload, None)[0] == main
    payload, main, _ = fixture()
    main = subject.veto.unresolved_judge_output_v3()
    assert subject.apply_review(main, payload, None)[0] == main
    payload, main, _ = fixture()
    payload["long_document_evidence"]["truncated"] = True
    assert subject.apply_review(main, payload, None)[0] == main


def test_schema_enforces_side_inventory_and_message_retains_original_order():
    payload, _, _ = fixture()
    schema = subject.response_schema(payload)
    assert schema["properties"]["a_subject_span_id"]["enum"] == ["", "A001"]
    assert "B001" not in schema["properties"]["a_predicate_span_id"]["enum"]
    message = subject.messages(payload, "Instructions")[1]["content"]
    assert message.index("[S001") < message.index("[A001") < message.index("[S002")
    assert message.count("Agency") == 2
