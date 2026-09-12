# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging.coverage_witness import CONTRACT, adapt_coverage_witness, parse_coverage
from eval.dedup.judging.payload import _semantic_diff_packet, validate_evidence_offsets
from eval.dedup.judging.schema_v3 import validate_judge_output_v3
from eval.dedup.validation import DedupEvaluationError


def _case(text_a="FAQ Model Z.", text_b="FAQ Model Z. Restart the device.", *, non_main=False):
    packet = _semantic_diff_packet(text_a, text_b, truncated=False)
    payload = {
        "document_a": {"text": text_a},
        "document_b": {"text": text_b},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": packet,
    }
    refs = " ".join(span["span_id"] for span in packet["spans"])
    scores = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "cookie_consent" if non_main else "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
        "span_content_profile_a": "non_main_only" if non_main else "substantive_main",
        "span_content_profile_b": "non_main_only" if non_main else "substantive_main",
        "span_shared_basis": "verified_equivalent_non_main_message" if non_main else "verified_substantive_record",
        "span_a_delta": "semantically_covered" if packet["span_counts"]["A_ONLY"] else "none",
        "span_b_delta": "semantically_covered" if packet["span_counts"]["B_ONLY"] else "none",
        "span_hard_conflict": "none",
        "span_translation_status": "not_translation",
    }
    main = {key: {"score": value, "reasoning": refs} for key, value in scores.items()}
    anchors = {}
    for side in ("A", "B"):
        anchors[side] = next(
            span["span_id"] for span in packet["spans"] if span["kind"] == "SHARED" or span.get("side") == side
        )
    value = {
        "contract_version": CONTRACT,
        "input_status": "COMPLETE",
        "record_scope": "NON_MAIN_MESSAGES" if non_main else "SAME_SUBSTANTIVE_RECORD",
        "anchor_a_ids": anchors["A"],
        "anchor_b_ids": anchors["B"],
        "scope_explanation": "Both anchors identify the same FAQ or reusable message.",
    }
    for side, field in (("A", "a_meaning_in_b"), ("B", "b_meaning_in_a")):
        unique = [s["span_id"] for s in packet["spans"] if s.get("side") == side]
        value[field] = {
            "status": "COVERED",
            "reviewed_unique_ids": " ".join(unique),
            "coverage_mode": "SEMANTIC_COUNTERPARTS" if unique else "NO_UNIQUE_SPANS",
            "coverage_counterpart_ids": anchors["B" if side == "A" else "A"] if unique else "",
            "coverage_explanation": "Opposite context expresses the retained meaning.",
            "uncovered_type": "NONE",
            "source_ids": "",
            "source_quote": "",
            "counterpart_ids": "",
            "counterpart_quote": "",
            "counterpart_relation": "NOT_APPLICABLE",
            "retention_consequence": "",
        }
    return main, value, payload


def _uncover(value, payload, side="B", kind="MAIN_CONTENT"):
    field = "a_meaning_in_b" if side == "A" else "b_meaning_in_a"
    span = next(s for s in payload["semantic_diff_evidence"]["spans"] if s.get("side") == side)
    value[field].update(
        status="UNCOVERED",
        coverage_mode="NOT_COVERED",
        coverage_counterpart_ids="",
        uncovered_type=kind,
        source_ids=span["span_id"],
        source_quote=span["text"],
        counterpart_relation="NO_EQUIVALENT_FOUND",
        retention_consequence="The opposite side omits this specific procedure or retained condition.",
    )


@pytest.mark.parametrize("side", ["A", "B"])
def test_source_side_coverage_maps_to_the_correct_replacement_direction(side):
    a, b = "FAQ Model Z.", "FAQ Model Z. Restart the device."
    main, value, payload = _case(*((b, a) if side == "A" else (a, b)))
    _uncover(value, payload, side)
    original = deepcopy((main, value, payload))
    result = adapt_coverage_witness(main, value, payload)
    assert result["relation_type"] == "CONTAINMENT"
    assert result["a_can_replace_b"] == ("YES" if side == "A" else "NO")
    assert result["b_can_replace_a"] == ("YES" if side == "B" else "NO")
    assert result["primary_material_difference"] == "MAIN_CONTENT_ADDITION_DELETION"
    validate_judge_output_v3(result)
    validate_evidence_offsets(result, payload)
    assert (main, value, payload) == original


@pytest.mark.parametrize("kind", ["POLICY_MEANING", "RECORD_IDENTITY", "PAGE_ROLE", "STATE_VERSION", "MEMBERSHIP"])
def test_actual_uncovered_nonmain_meaning_is_not_bidirectional_or_containment(kind):
    main, value, payload = _case(
        "Cookies need consent.", "Cookies need consent. Ads require prior permission.", non_main=True
    )
    _uncover(value, payload, kind=kind)
    result = adapt_coverage_witness(main, value, payload)
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"
    assert result["material_difference"] == "MAJOR"
    assert result["confidence_tier"] == "MEDIUM"


def test_same_message_with_only_optional_controls_remains_equivalent():
    main, value, payload = _case("Cookies need consent.", "Cookies need consent. Accept Settings", non_main=True)
    value["b_meaning_in_a"].update(
        coverage_mode="HARMLESS_ONLY",
        coverage_counterpart_ids="",
        coverage_explanation="Optional controls assert no changed consent state or permission.",
    )
    result = adapt_coverage_witness(main, value, payload)
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "YES"


def test_complete_translation_keeps_none_materiality_and_unicode_evidence():
    main, value, payload = _case("Cookies need consent.", "Cookie 需要获得同意。", non_main=True)
    main["span_translation_status"]["score"] = "complete_faithful"
    result = adapt_coverage_witness(main, value, payload)
    assert result["same_duplicate_group"] == "YES"
    assert result["material_difference"] == "NONE"
    validate_evidence_offsets(result, payload)


def test_new_coverage_can_correct_a_false_main_extension_with_explicit_counterparts():
    main, value, payload = _case("FAQ Model Z. Restart.", "FAQ Model Z. Reboot.")
    main["span_a_delta"]["score"] = "same_record_content_extension"
    result = adapt_coverage_witness(main, value, payload)
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "YES"
    assert result["relation_type"] == "NEAR_SURFACE"


def test_distinct_attachment_and_two_sided_losses_never_become_containment():
    main, value, payload = _case("FAQ Model Z. Reboot.", "FAQ Model Z. Replace the battery.")
    for side in ("A", "B"):
        _uncover(value, payload, side)
    assert adapt_coverage_witness(main, value, payload)["same_duplicate_group"] == "NO"
    main, value, payload = _case()
    _uncover(value, payload)
    value["record_scope"] = "DISTINCT_OR_UNBOUND_RECORDS"
    assert adapt_coverage_witness(main, value, payload)["same_duplicate_group"] == "NO"


@pytest.mark.parametrize("basis", ["none", "unresolved"])
def test_new_critic_does_not_reopen_owned_negative_or_unresolved_main(basis):
    main, value, payload = _case()
    main["span_shared_basis"]["score"] = basis
    result = adapt_coverage_witness(main, value, payload)
    assert result["same_duplicate_group"] == ("NO" if basis == "none" else "UNRESOLVED")


def test_exact_identity_is_protected_without_invented_unique_spans():
    main, value, payload = _case("Identical legal text.", "Identical legal text.", non_main=True)
    value["record_scope"] = "IDENTICAL_TEXT"
    result = adapt_coverage_witness(main, value, payload)
    assert result["relation_type"] == "EXACT"
    assert result["evidence"] == []


@pytest.mark.parametrize(
    "bad",
    [
        "missing_field",
        "extra_field",
        "old_contract",
        "unknown_id",
        "wrong_side",
        "source_shared_only",
        "omitted_unique",
        "wrong_quote",
        "no_consequence",
        "false_contradiction",
        "false_exact",
        "covered_witness",
        "invalid_range",
    ],
)
def test_incomplete_or_fabricated_proof_is_rejected_not_promoted(bad):
    main, value, payload = _case()
    _uncover(value, payload)
    witness = value["b_meaning_in_a"]
    if bad == "missing_field":
        value.pop("a_meaning_in_b")
    elif bad == "old_contract":
        value = {"record_binding_verdict": {"score": "atomic_same_record_extension", "reasoning": "S001 B001"}}
    else:
        target, key, replacement = {
            "extra_field": (value, "label", "YES"),
            "wrong_side": (value, "anchor_a_ids", "B001"),
            "false_exact": (value, "record_scope", "IDENTICAL_TEXT"),
            "unknown_id": (witness, "source_ids", "B999"),
            "source_shared_only": (witness, "source_ids", "S001"),
            "omitted_unique": (witness, "reviewed_unique_ids", ""),
            "wrong_quote": (witness, "source_quote", "Invented permission"),
            "no_consequence": (witness, "retention_consequence", ""),
            "false_contradiction": (witness, "counterpart_relation", "CONTRADICTS"),
            "covered_witness": (witness, "status", "COVERED"),
            "invalid_range": (witness, "source_ids", "B002-B001"),
        }[bad]
        target[key] = replacement
    with pytest.raises(DedupEvaluationError):
        adapt_coverage_witness(main, value, payload)


def test_truncation_requires_explicit_unresolved_contract():
    main, value, payload = _case()
    payload["long_document_evidence"]["truncated"] = True
    with pytest.raises(DedupEvaluationError, match="incomplete input"):
        parse_coverage(value, payload)
    value.update(input_status="UNRESOLVED", record_scope="UNRESOLVED", anchor_a_ids="", anchor_b_ids="")
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        value[field].update(status="UNRESOLVED", coverage_mode="NOT_COVERED", coverage_counterpart_ids="")
    result = adapt_coverage_witness(main, value, payload)
    assert result["same_duplicate_group"] == "UNRESOLVED"
    assert result["confidence_tier"] == "LOW"


def test_alignment_is_checked_even_for_uncited_spans():
    _, value, payload = _case()
    payload["semantic_diff_evidence"]["spans"][-1]["text"] = "Invented text"
    with pytest.raises(DedupEvaluationError, match="align exactly"):
        parse_coverage(value, payload)
