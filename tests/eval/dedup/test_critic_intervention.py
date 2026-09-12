# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import critic_intervention as subject
from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V2, _semantic_diff_packet
from eval.dedup.validation import DedupEvaluationError


def payload(
    a="Shared complete policy.\nRefund within 14 days.",
    b="Shared complete policy.\nRefund within 14 days.\nProduct: red K7.",
):
    result = {
        "payload_schema_version": VISIBLE_PAYLOAD_V2,
        "document_a": {"text": a},
        "document_b": {"text": b},
        "long_document_evidence": {"truncated": False, "windows": []},
    }
    result["semantic_diff_evidence"] = _semantic_diff_packet(a, b, truncated=False)
    return result


def main(p, a="NO", b="YES"):
    same = "YES" in {a, b}
    relation = "CONTAINMENT" if a != b else "NEAR_SURFACE" if same else "RELATED_NON_DUPLICATE"
    witnesses = []
    for side in ("A", "B"):
        text = p[f"document_{side.lower()}"]["text"]
        witnesses.append({"side": side, "start_char": 0, "end_char": min(len(text), 100), "quote": text[:100]})
    return {
        "same_duplicate_group": "YES" if same else "NO",
        "a_can_replace_b": a,
        "b_can_replace_a": b,
        "relation_type": relation,
        "material_difference": "NONE" if relation == "NEAR_SURFACE" else "MAJOR",
        "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION"
        if relation == "CONTAINMENT"
        else "NONE"
        if relation == "NEAR_SURFACE"
        else "OTHER_MATERIAL",
        "dominant_overlap_source": "MAIN_CONTENT",
        "primary_risk_factor": "CONTAINMENT_ASYMMETRY",
        "confidence_tier": "MEDIUM",
        "reason_codes": [],
        "evidence": witnesses,
    }


def response(p, action="KEEP_MAIN", basis="NO_MATERIAL_OBJECTION"):
    spans = p["semantic_diff_evidence"]["spans"]
    shared = [s for s in spans if s["kind"] == "SHARED"]
    evidence = {}
    for side in ("A", "B"):
        chosen = next((s for s in spans if s["kind"] == f"{side}_ONLY"), shared[0])
        quote = chosen.get(f"{side.lower()}_text", chosen.get("text"))[:100]
        evidence[f"{side.lower()}_evidence"] = [{"span_id": chosen["span_id"], "quote": quote}]
    return {
        "contract_version": subject.CONTRACT,
        "action": action,
        "basis": basis,
        **evidence,
        "shared_anchor_ids": [s["span_id"] for s in shared],
        "explanation": "Cited retained meanings and their consequence.",
    }


def test_full_policy_plus_independent_product_preserves_only_safe_main_direction():
    p = payload()
    before = main(p)
    r = response(p, "REJECT_A_REPLACES_B", "UNCOVERED_CONTENT")
    saved = deepcopy((before, p, r))
    after, rule = subject.apply_review(before, p, r)
    assert after == before
    assert rule == "OBJECTION_ONLY_TO_ALREADY_UNSAFE_DIRECTION"
    assert (before, p, r) == saved


def test_wrong_direction_can_only_lose_yes_not_gain_opposite_yes():
    p = payload()
    after, _ = subject.apply_review(main(p, "YES", "NO"), p, response(p, "REJECT_A_REPLACES_B", "UNCOVERED_CONTENT"))
    assert after["a_can_replace_b"] == after["b_can_replace_a"] == "NO"
    subject.validate_judge_output_v3(after)


def test_false_equivalence_can_become_one_way_without_erasing_real_addition():
    p = payload()
    after, _ = subject.apply_review(main(p, "YES", "YES"), p, response(p, "REJECT_A_REPLACES_B", "UNCOVERED_CONTENT"))
    assert after["a_can_replace_b"] == "NO"
    assert after["b_can_replace_a"] == "YES"
    assert after["relation_type"] == "CONTAINMENT"
    assert after["primary_material_difference"] == "MAIN_CONTENT_ADDITION_DELETION"


def test_rejecting_richer_side_requires_content_lost_from_poorer_side():
    p = payload()
    with pytest.raises(DedupEvaluationError, match="CRITIC_SCOPE_DIRECTION"):
        subject.apply_review(main(p), p, response(p, "REJECT_B_REPLACES_A", "UNCOVERED_CONTENT"))


def test_independent_addition_does_not_supply_two_sided_loss():
    p = payload()
    with pytest.raises(DedupEvaluationError, match="CRITIC_SCOPE_DIRECTION"):
        subject.validate_review(response(p, "REJECT_BOTH", "UNCOVERED_CONTENT"), p)


@pytest.mark.parametrize("basis", list(subject.CONFLICTS))
def test_actual_bilateral_value_conflict_still_vetoes(basis):
    p = payload("Shared policy.\nSeller Alpha handles refunds.", "Shared policy.\nSeller Beta handles refunds.")
    after, _ = subject.apply_review(main(p, "YES", "YES"), p, response(p, "REJECT_BOTH", basis))
    assert after["same_duplicate_group"] == "NO"
    assert after["confidence_tier"] == "MEDIUM"
    subject.validate_evidence_offsets(after, p)


def test_empty_anchor_refutation_requires_all_shared_spans_and_unique_delta():
    p = payload(
        "Cookie settings.\nNecessary cookies.\nAccept button.",
        "Cookie settings.\nNecessary cookies.\nAccept button.\nArticle about clocks.",
    )
    r = response(p, "REJECT_EMPTY_ANCHOR", "EMPTY_SHARED_ANCHOR")
    after, _ = subject.apply_review(main(p), p, r)
    assert after["same_duplicate_group"] == "NO"
    r["shared_anchor_ids"] = []
    with pytest.raises(DedupEvaluationError, match="CRITIC_SCOPE_EMPTY_ANCHOR"):
        subject.validate_review(r, p)


def test_empty_anchor_cannot_veto_equivalent_nonmain_messages():
    p = payload("Cookie settings.", "Cookie settings.\nAccept button.")
    before = main(p, "YES", "YES")
    after, rule = subject.apply_review(before, p, response(p, "REJECT_EMPTY_ANCHOR", "EMPTY_SHARED_ANCHOR"))
    assert after == before
    assert rule == "EMPTY_ANCHOR_VETO_OUTSIDE_CONTAINMENT_SCOPE"


def test_explicit_uncertainty_is_not_a_guessed_negative():
    p = payload()
    after, _ = subject.apply_review(main(p), p, response(p, "ABSTAIN", "INSUFFICIENT_EVIDENCE"))
    assert after["same_duplicate_group"] == "UNRESOLVED"
    assert after["confidence_tier"] == "LOW"
    assert after["evidence"] == []


@pytest.mark.parametrize(
    "damage", ["extra_key", "legacy", "wrong_side", "wrong_quote", "wrong_span", "missing_side", "truncated"]
)
def test_invalid_proof_is_not_silently_accepted(damage):
    p = payload()
    r = response(p, "REJECT_A_REPLACES_B", "UNCOVERED_CONTENT")
    if damage == "extra_key":
        r["extra"] = True
    elif damage == "legacy":
        r = {"record_binding_verdict": {"score": "separate_record_or_template_attachment", "reasoning": "S001 B001"}}
    elif damage == "wrong_side":
        r["a_evidence"] = deepcopy(r["b_evidence"])
    elif damage == "wrong_quote":
        r["b_evidence"][0]["quote"] = "Invented text"
    elif damage == "wrong_span":
        r["b_evidence"][0]["span_id"] = "B999"
    elif damage == "missing_side":
        r["a_evidence"] = []
    else:
        p["long_document_evidence"]["truncated"] = True
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(main(p), p, r)


def test_exact_complete_identity_bypasses_even_missing_critic():
    p = payload("Identical document.", "Identical document.")
    before = main(p, "YES", "YES")
    after, rule = subject.apply_review(before, p, None)
    assert after == before
    assert rule == "PRESERVE_COMPLETE_EXACT_INPUT"


def test_main_negative_and_unresolved_are_never_reopened():
    p = payload()
    for before in (main(p, "NO", "NO"), subject.unresolved_judge_output_v3()):
        after, rule = subject.apply_review(before, p, None)
        assert after == before
        assert rule == "PRESERVE_MAIN_NEGATIVE_OR_UNRESOLVED"


def test_utf8_quotes_and_mirrored_directions_remain_aligned():
    p = payload("完整政策。\n新增商品: 红色。", "完整政策。")
    r = response(p, "REJECT_B_REPLACES_A", "UNCOVERED_CONTENT")
    after, _ = subject.apply_review(main(p, "YES", "NO"), p, r)
    assert after["a_can_replace_b"] == "YES"
    assert after["b_can_replace_a"] == "NO"
    for item in subject.validate_review(r, p):
        text = p[f"document_{item['side'].lower()}"]["text"]
        assert text[item["start_char"] : item["end_char"]] == item["quote"]
