# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import critic_retention_v2 as subject
from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V2, _semantic_diff_packet
from eval.dedup.validation import DedupEvaluationError


def fixture(a, b):
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
            {"side": side, "start_char": 0, "end_char": min(len(text), 80), "quote": text[:80]}
            for side, text in (("A", a), ("B", b))
        ],
    }
    spans = p["semantic_diff_evidence"]["spans"]
    shared = [s for s in spans if s["kind"] == "SHARED"]
    value = {
        "a_has_uncovered_content": False,
        "b_has_uncovered_content": False,
        "conflict": "NONE",
        "overlap_basis": "RETAINED_CONTENT",
        "a_evidence": [],
        "b_evidence": [],
        "shared_anchor_ids": [s["span_id"] for s in shared],
        "explanation": "Compare retained content in the cited sources.",
    }
    for side in ("a", "b"):
        chosen = next((s for s in spans if s["kind"] == f"{side.upper()}_ONLY"), shared[0] if shared else None)
        if chosen:
            value[f"{side}_evidence"] = [
                {"span_id": chosen["span_id"], "quote": chosen.get(f"{side}_text", chosen.get("text"))[:100]}
            ]
    return p, main, value


@pytest.mark.parametrize("mirror", [False, True])
def test_source_loss_derives_inverse_direction_and_byte_aligned_evidence(mirror):
    texts = ["完整保修政策: 14 天退货。", "完整保修政策: 14 天退货。\n独立产品: 红色 K7。"]
    p, main, value = fixture(*(texts[::-1] if mirror else texts))
    source = "a" if mirror else "b"
    value[f"{source}_has_uncovered_content"] = True
    before = deepcopy((p, main, value))
    result, _ = subject.apply_review(main, p, value)
    assert result["a_can_replace_b"] == ("YES" if mirror else "NO")
    assert result["b_can_replace_a"] == ("NO" if mirror else "YES")
    assert result["relation_type"] == "CONTAINMENT"
    assert result["primary_material_difference"] == "MAIN_CONTENT_ADDITION_DELETION"
    subject.veto.validate_evidence_offsets(result, p)
    assert before == (p, main, value)


@pytest.mark.parametrize("mirror", [False, True])
def test_shared_policy_cannot_be_claimed_as_source_only_loss(mirror):
    texts = ["Complete policy. Refund in 14 days.", "Complete policy. Refund in 14 days.\nIndependent product K7."]
    p, main, value = fixture(*(texts[::-1] if mirror else texts))
    value[f"{'b' if mirror else 'a'}_has_uncovered_content"] = True
    with pytest.raises(DedupEvaluationError, match="RETENTION_SOURCE_WITNESS"):
        subject.apply_review(main, p, value)


def test_conflicting_forum_subject_is_not_an_independent_addition():
    p, main, value = fixture("Using Forum Alpha accepts these rules.", "Using Forum Beta accepts these rules.")
    value.update(a_has_uncovered_content=True, b_has_uncovered_content=True, conflict="IDENTITY_CONFLICT")
    result, _ = subject.apply_review(main, p, value)
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"
    assert result["primary_risk_factor"] == "TEMPLATE_SLOT_COLLISION"


def test_empty_interface_anchor_only_vetoes_apparent_containment():
    p, main, value = fixture(
        "I read the privacy notice.", "I read the privacy notice.\nA substantive economic report."
    )
    value.update(b_has_uncovered_content=True, overlap_basis="INTERFACE_ONLY")
    equivalent, _ = subject.apply_review(main, p, value)
    assert equivalent == main
    main.update(
        a_can_replace_b="NO",
        relation_type="CONTAINMENT",
        material_difference="MAJOR",
        primary_material_difference="MAIN_CONTENT_ADDITION_DELETION",
    )
    result, _ = subject.apply_review(main, p, value)
    assert result["same_duplicate_group"] == "NO"


def test_translation_and_harmless_ui_do_not_create_loss_from_unique_tokens():
    p, main, value = fixture("Cookie information. Accept settings.", "Cookie information. Configuración de cookies.")
    result, _ = subject.apply_review(main, p, value)
    assert result == main


def test_uncertainty_remains_explicit_and_low_confidence():
    p, main, value = fixture("Credit notice.", "Credit notice.\nProduct information.")
    value["overlap_basis"] = "UNCERTAIN"
    result, _ = subject.apply_review(main, p, value)
    assert result["same_duplicate_group"] == "UNRESOLVED"
    assert result["confidence_tier"] == "LOW"


@pytest.mark.parametrize(
    "damage", ["version", "action", "string_bool", "extra", "quote", "cross_side", "conflict_flags", "truncated"]
)
def test_malformed_or_unaligned_new_proof_is_not_accepted(damage):
    p, main, value = fixture("Shared policy.", "Shared policy.\nProduct K7.")
    value["b_has_uncovered_content"] = True
    if damage == "version":
        value["contract_version"] = subject.CONTRACT
    elif damage == "action":
        value["action"] = "REJECT_A_REPLACES_B"
    elif damage == "string_bool":
        value["b_has_uncovered_content"] = "true"
    elif damage == "extra":
        value["new_field"] = None
    elif damage == "quote":
        value["b_evidence"][0]["quote"] = "Invented content."
    elif damage == "cross_side":
        value["a_evidence"] = deepcopy(value["b_evidence"])
    elif damage == "conflict_flags":
        value["conflict"] = "POLICY_CONFLICT"
    else:
        p["long_document_evidence"]["truncated"] = True
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(main, p, value)


def test_format_only_contract_is_new_and_never_accepts_a_model_version_field():
    p, main, value = fixture("Shared policy.", "Shared policy.\nProduct K7.")
    value["b_has_uncovered_content"] = True
    old = subject.proof(value)
    stripped = {k: v for k, v in old.items() if k != "contract_version"}
    result, _ = subject.apply_review(main, p, stripped, format_only=True)
    assert result["b_can_replace_a"] == "YES"
    assert result["a_can_replace_b"] == "NO"
    with pytest.raises(DedupEvaluationError, match="RETENTION_SCHEMA"):
        subject.apply_review(main, p, old, format_only=True)


def test_main_negative_and_exact_inputs_do_not_require_or_use_a_critic():
    p, main, _ = fixture("Exactly same.", "Exactly same.")
    assert subject.apply_review(main, p, None)[0] == main
    negative = subject.veto.unresolved_judge_output_v3()
    assert subject.apply_review(negative, p, None)[0] == negative


def test_source_loss_cannot_create_the_missing_positive_direction():
    p, main, value = fixture("Shared policy.", "Shared policy.\nProduct K7.")
    main.update(
        b_can_replace_a="NO",
        relation_type="CONTAINMENT",
        material_difference="MAJOR",
        primary_material_difference="MAIN_CONTENT_ADDITION_DELETION",
    )
    value["b_has_uncovered_content"] = True
    result, _ = subject.apply_review(main, p, value)
    assert result["same_duplicate_group"] == "NO"
