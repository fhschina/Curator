# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import retention_v4 as subject
from eval.dedup.judging.payload import _semantic_diff_packet, validate_evidence_offsets
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_critic_subject_binding import fixture as subject_fixture


def payload(a, b):
    return {
        "document_a": {"text": a},
        "document_b": {"text": b},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": _semantic_diff_packet(a, b, truncated=False),
    }


def review(packet, a="NONE", b="NONE", *, conflict="NONE", severity=None):
    spans = subject.selection.span_inventory(packet)
    value = {
        "shared_basis": "PRESENT",
        "conflict": conflict,
        "material_difference": severity
        or (
            "MAJOR"
            if "MAIN_ADDITION" in (a, b) or conflict != "NONE"
            else "MINOR"
            if "NON_MAIN_ADDITION" in (a, b)
            else "NONE"
        ),
        "dominant_overlap_source": "LEGAL_POLICY_TEMPLATE",
        "confidence_tier": "HIGH",
        "explanation": "Synthetic source-retention proof, not a model output.",
    }
    for side, kind in (("a", a), ("b", b)):
        own = [sid for sid, span in spans.items() if span["kind"] == side.upper() + "_ONLY"]
        context = own or [sid for sid, span in spans.items() if span["kind"] == "SHARED"]
        value.update(
            {
                f"{side}_loss_kind": kind,
                f"{side}_loss_span_id": own[0] if kind in {"MAIN_ADDITION", "NON_MAIN_ADDITION"} else "",
                f"{side}_context_span_id": context[0],
            }
        )
    return value


@pytest.mark.parametrize(
    ("addition", "kind"),
    [
        ("Product details: keeps drinks cold for six hours.", "MAIN_ADDITION"),
        ("An independent article about migrating birds.", "MAIN_ADDITION"),
        ("Field cup", "NON_MAIN_ADDITION"),
        ("Next page", "NON_MAIN_ADDITION"),
        ("Accept Settings", "NON_MAIN_ADDITION"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_nonempty_containment_has_only_the_complete_direction_without_profile_or_record_gate(addition, kind, reverse):
    common = "Returns accepted within thirty days."
    a, b = common + "\n\n" + addition, common
    if reverse:
        a, b = b, a
    p = payload(a, b)
    v = review(p, *(("NONE", kind) if reverse else (kind, "NONE")))
    before = deepcopy((p, v))
    output = subject.adapt_main(v, p)
    assert (output["a_can_replace_b"], output["b_can_replace_a"]) == (("NO", "YES") if reverse else ("YES", "NO"))
    assert output["material_difference"] == ("MINOR" if kind == "NON_MAIN_ADDITION" else "MAJOR")
    assert output["primary_material_difference"] == (
        "NON_MAIN_CONTENT_ADDITION_DELETION" if kind == "NON_MAIN_ADDITION" else "MAIN_CONTENT_ADDITION_DELETION"
    )
    validate_evidence_offsets(output, p)
    assert (p, v) == before


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Use cookies only with your consent.", "Cookie使用需征得您的同意。"),
        ("A short statement.", "A   short statement.\n"),
        ("The service is open. The service is open.", "The service is open."),
    ],
)
def test_full_translation_formatting_and_redundant_repetition_remain_bidirectional(a, b):
    p = payload(a, b)
    output = subject.adapt_main(review(p), p)
    assert output["a_can_replace_b"] == output["b_can_replace_a"] == "YES"
    assert output["material_difference"] == "NONE"


@pytest.mark.parametrize(
    ("conflict", "a", "b"),
    [
        ("IDENTITY_CONFLICT", "Product SKU A11 is available.", "Product SKU B22 is available."),
        ("STATE_CONFLICT", "Service is open.", "Service is closed."),
        ("POLICY_CONFLICT", "Consent covers analytics only.", "Consent covers advertising only."),
        ("ROLE_CONFLICT", "Permission is for viewing profiles.", "Permission is for editing profiles."),
        ("NEGATION_CONFLICT", "This claim is false: Earth is flat.", "Earth is flat."),
        ("IDENTITY_CONFLICT", "This service belongs to Ann.", "This service belongs to Anna."),
    ],
)
def test_actual_conflict_not_length_or_substring_decides_negative(conflict, a, b):
    p = payload(a, b)
    out = subject.adapt_main(review(p, conflict=conflict), p)
    assert out["a_can_replace_b"] == out["b_can_replace_a"] == "NO"
    assert out["material_difference"] == "MAJOR"
    assert out["relation_type"] == ("VERSION_RELATED" if conflict == "STATE_CONFLICT" else "RELATED_NON_DUPLICATE")


def test_both_uncovered_nonmain_additions_are_negative_without_fabricating_major_severity():
    p = payload("Shared article.\nContact", "Shared article.\nEvents")
    out = subject.adapt_main(review(p, "NON_MAIN_ADDITION", "NON_MAIN_ADDITION"), p)
    assert out["same_duplicate_group"] == "NO"
    assert out["material_difference"] == "MINOR"


@pytest.mark.parametrize("condition", ["empty", "whitespace", "truncated", "capped"])
def test_missing_evidence_remains_unresolved_and_does_not_need_a_model(condition):
    p = payload("Content A.", "Content B.")
    if condition == "truncated":
        p["long_document_evidence"]["truncated"] = True
    elif condition == "capped":
        p["semantic_diff_evidence"]["status"] = "INCOMPLETE_LIMIT"
    else:
        p["document_b"]["text"] = "" if condition == "empty" else "  \n"
    out = subject.adapt_main(None, p)
    assert out["same_duplicate_group"] == "UNRESOLVED"
    assert out["confidence_tier"] == "LOW"


def test_complete_exact_input_is_not_an_empty_set_and_bypasses_critics():
    p = payload("Identical policy.", "Identical policy.")
    out = subject.adapt_main(None, p)
    assert out["relation_type"] == "EXACT"
    assert subject.apply_critic(out, p, None)[0] == out
    assert subject.subject_route(out, p) != "REVIEW_BILATERAL_SUBJECTS"


@pytest.mark.parametrize(
    "damage",
    [
        "unknown",
        "shared_loss",
        "wrong_side",
        "extra_field",
        "missing_loss",
        "missing_context",
        "false_basis",
        "severity",
        "unaligned_uncited",
    ],
)
def test_bad_or_missing_proofs_cannot_be_silently_repaired_into_a_positive(damage):
    p = payload("Policy applies.\nProduct K9", "Policy applies.")
    value = review(p, "NON_MAIN_ADDITION")
    if damage == "unknown":
        value["a_loss_span_id"] = "A999"
    elif damage == "shared_loss":
        value["a_loss_span_id"] = "S001"
    elif damage == "wrong_side":
        value["b_context_span_id"] = "A001"
    elif damage == "extra_field":
        value["old_profile"] = "SUBSTANTIVE_MAIN"
    elif damage == "missing_loss":
        value["a_loss_span_id"] = ""
    elif damage == "missing_context":
        value["b_context_span_id"] = ""
    elif damage == "false_basis":
        value["shared_basis"] = "NONE"
    elif damage == "severity":
        value["material_difference"] = "NONE"
    else:
        p["semantic_diff_evidence"]["spans"][0]["a_text"] = "Corrupted"
    with pytest.raises(DedupEvaluationError):
        subject.adapt_main(value, p)


def test_critic_only_removes_directions_and_preserves_minor_nonmain_taxonomy():
    p = payload("Policy applies.\nNext page", "Policy applies.")
    mistaken_main = subject.adapt_main(review(p), p)
    value = review(p, "NON_MAIN_ADDITION")
    corrected, _ = subject.apply_critic(mistaken_main, p, value)
    assert corrected["a_can_replace_b"] == "YES"
    assert corrected["b_can_replace_a"] == "NO"
    assert corrected["material_difference"] == "MINOR"
    assert subject.apply_critic(corrected, p, review(p))[0] == corrected


def test_opposite_objection_combines_two_unsafe_directions_not_a_keeper_reversal():
    p = payload("Shared policy.\nAlpha", "Shared policy.\nBeta")
    main = subject.adapt_main(review(p, "NON_MAIN_ADDITION"), p)
    opposite = review(p, "NONE", "NON_MAIN_ADDITION")
    out, _ = subject.apply_critic(main, p, opposite)
    assert out["a_can_replace_b"] == out["b_can_replace_a"] == "NO"
    assert {e["side"] for e in out["evidence"]} == {"A", "B"}


def test_uncertainty_is_not_a_negative_or_an_upgrade_and_old_outputs_are_not_new_proofs():
    p = payload("Statement A.", "Statement B.")
    uncertain = review(p, "UNRESOLVED")
    out = subject.adapt_main(uncertain, p)
    assert out["same_duplicate_group"] == "UNRESOLVED"
    assert out["confidence_tier"] == "LOW"
    assert subject.apply_critic(out, p, None)[0] == out
    with pytest.raises(DedupEvaluationError, match="RETENTION_V4_FIELDS"):
        subject.adapt_main({"a_can_replace_b": {"score": "yes", "reasoning": "S001"}}, p)


def test_fixed_subject_verification_retains_narrow_scope_and_never_reselects_a_veto():
    p, main, proposal = subject_fixture()
    review_value = {
        "a_subject_kind": "NAMED_ACTUAL_TARGET",
        "b_subject_kind": "NAMED_ACTUAL_TARGET",
        "comparison": "SUPPORTED_DIFFERENT_NAMED_TARGETS",
        "explanation": "The exact selected parties differ.",
    }
    before = deepcopy((p, main, proposal, review_value))
    out, rule = subject.apply_subject_verification(main, p, proposal, review_value)
    assert out["same_duplicate_group"] == "NO"
    assert rule == "VERIFIED_FIXED_SUBJECT_VETO"
    assert (p, main, proposal, review_value) == before
    review_value["comparison"] = "UNSUPPORTED_COMPARISON"
    assert subject.apply_subject_verification(main, p, proposal, review_value)[0] == main
    proposal["binding_type"] = "RECORD_SUBJECT"
    assert subject.apply_subject_verification(main, p, proposal, None)[0] == main


def test_invalid_subject_proof_is_not_hidden_by_an_unsupported_verification():
    p, main, proposal = subject_fixture()
    proposal["a_subject_span_id"] = "B001"
    with pytest.raises(DedupEvaluationError):
        subject.apply_subject_verification(main, p, proposal, None)
