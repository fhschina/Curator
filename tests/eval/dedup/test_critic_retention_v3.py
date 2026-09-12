# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import critic_retention_v3 as subject
from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V2, _semantic_diff_packet
from eval.dedup.validation import DedupEvaluationError


def fixture(a="Shared policy.", b="Shared policy.\nIndependent product K7."):
    p = {
        "payload_schema_version": VISIBLE_PAYLOAD_V2,
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
        "primary_risk_factor": "CONTAINMENT_ASYMMETRY",
        "confidence_tier": "MEDIUM",
        "reason_codes": [],
        "evidence": [
            {"side": side, "start_char": 0, "end_char": min(80, len(text)), "quote": text[:80]}
            for side, text in (("A", a), ("B", b))
        ],
    }
    spans = p["semantic_diff_evidence"]["spans"]
    value = {
        "a_loss_span_id": "",
        "b_loss_span_id": "",
        "a_context_span_id": next(s["span_id"] for s in spans if s["kind"] in ("A_ONLY", "SHARED")),
        "b_context_span_id": next(s["span_id"] for s in spans if s["kind"] in ("B_ONLY", "SHARED")),
        "conflict": "NONE",
        "overlap_basis": "RETAINED_CONTENT",
        "shared_anchor_ids": [s["span_id"] for s in spans if s["kind"] == "SHARED"],
        "explanation": "Checked source additions and the actual retained context.",
    }
    return p, main, value


@pytest.mark.parametrize("mirror", [False, True])
def test_selected_loss_compiles_full_exact_evidence_and_correct_inverse_direction(mirror):
    text = ["完整政策。", "完整政策。\n产品 K7。"]
    p, main, value = fixture(*(text[::-1] if mirror else text))
    side = "a" if mirror else "b"
    value[f"{side}_loss_span_id"] = next(
        s["span_id"] for s in p["semantic_diff_evidence"]["spans"] if s["kind"] == f"{side.upper()}_ONLY"
    )
    saved = deepcopy((p, main, value))
    result, _ = subject.apply_review(main, p, value)
    assert result["a_can_replace_b"] == ("YES" if mirror else "NO")
    assert result["b_can_replace_a"] == ("NO" if mirror else "YES")
    assert {e["side"] for e in result["evidence"]} == {"A", "B"}
    subject.previous.veto.validate_evidence_offsets(result, p)
    assert saved == (p, main, value)


def test_real_identity_conflict_does_not_require_redundant_generated_loss_booleans():
    p, main, value = fixture("Using Forum Alpha accepts these rules.", "Using Forum Beta accepts these rules.")
    value.update(conflict="IDENTITY_CONFLICT", a_context_span_id="A001", b_context_span_id="B001")
    result, _ = subject.apply_review(main, p, value)
    assert result["same_duplicate_group"] == "NO"
    assert result["primary_material_difference"] == "DOCUMENT_IDENTITY_CHANGE"
    assert all(e["quote"] in p[f"document_{e['side'].lower()}"]["text"] for e in result["evidence"])


@pytest.mark.parametrize(
    "damage",
    [
        "missing_context",
        "foreign_context",
        "shared_loss",
        "unknown_loss",
        "id_array",
        "extra_flag",
        "schema_echo",
        "unaligned_uncited_span",
        "forged_kind",
        "uncertain_with_invalid_id",
    ],
)
def test_selected_proofs_never_repair_invalid_or_wrong_side_ids(damage):
    p, main, value = fixture()
    value["b_loss_span_id"] = "B001"
    if damage == "missing_context":
        value["a_context_span_id"] = ""
    elif damage == "foreign_context":
        value["a_context_span_id"] = "B001"
    elif damage == "shared_loss":
        value["a_loss_span_id"] = "S001"
    elif damage == "unknown_loss":
        value["b_loss_span_id"] = "B999"
    elif damage == "id_array":
        value["b_loss_span_id"] = ["B001"]
    elif damage == "extra_flag":
        value["a_has_uncovered_content"] = False
    elif damage == "schema_echo":
        value = subject.response_schema()
    elif damage == "unaligned_uncited_span":
        p["semantic_diff_evidence"]["spans"][0]["a_start_char"] = 1
    elif damage == "forged_kind":
        p["semantic_diff_evidence"]["spans"][-1]["side"] = "A"
    else:
        value.update(overlap_basis="UNCERTAIN", b_context_span_id="B999")
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(main, p, value)


def test_uncertainty_remains_low_and_does_not_invent_a_positive():
    p, main, value = fixture()
    value["overlap_basis"] = "UNCERTAIN"
    result, _ = subject.apply_review(main, p, value)
    assert result["same_duplicate_group"] == "UNRESOLVED"
    assert result["confidence_tier"] == "LOW"
    negative = subject.previous.veto.unresolved_judge_output_v3()
    assert subject.apply_review(negative, p, None)[0] == negative


def test_equal_input_bypass_does_not_require_model_proof():
    p, main, _ = fixture("Same complete text.", "Same complete text.")
    assert subject.apply_review(main, p, None)[0] == main


def test_semantic_empty_anchor_scope_and_translation_policy_are_unchanged():
    p, main, value = fixture("Cookie information. Accept.", "Cookie information. Aceptar.")
    assert subject.apply_review(main, p, value)[0] == main
    p, main, value = fixture("I read the privacy notice.", "I read the privacy notice.\nEconomic report.")
    main.update(
        a_can_replace_b="NO",
        relation_type="CONTAINMENT",
        material_difference="MAJOR",
        primary_material_difference="MAIN_CONTENT_ADDITION_DELETION",
    )
    value.update(overlap_basis="INTERFACE_ONLY", b_loss_span_id="B001")
    assert subject.apply_review(main, p, value)[0]["same_duplicate_group"] == "NO"
