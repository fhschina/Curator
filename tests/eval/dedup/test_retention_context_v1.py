# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import retention_context_v1 as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_retention_v4 import payload
from tests.eval.dedup.test_retention_v4 import review as legacy_review


def review(p, *args, **kwargs):
    reverse = {v: k for k, v in subject.RENAMES.items()}
    return {reverse.get(k, k): v for k, v in legacy_review(p, *args, **kwargs).items()}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shared", ["The service is free.", "La porte est ouverte.", "服务免费。"])
def test_context_can_refute_shared_claim_without_any_addition_or_unique_span_on_other_side(reverse, shared):
    a, b = "The following claim is false: " + shared, shared
    p = payload(b, a) if reverse else payload(a, b)
    value = review(p, conflict="NEGATION_CONFLICT")
    value["shared_basis"] = "NONE"
    before = deepcopy((p, value))
    out = subject.adapt_main(value, p)
    assert out["a_can_replace_b"] == out["b_can_replace_a"] == "NO"
    assert out["primary_material_difference"] == "NEGATION_CHANGE"
    assert out["material_difference"] == "MAJOR"
    assert value["a_addition_kind"] == value["b_addition_kind"] == "NONE"
    assert {e["side"] for e in out["evidence"]} == {"A", "B"}
    assert (p, value) == before
    subject.validate_evidence_offsets(out, p)


def test_critic_uses_typed_context_conflict_and_never_parses_explanation_for_a_veto():
    p = payload("The following claim is false: The service is free.", "The service is free.")
    value = review(p)
    value["explanation"] = "This sentence is contradicted; should have selected NEGATION_CONFLICT."
    main = subject.adapt_main(value, p)
    assert main["same_duplicate_group"] == "YES"
    assert subject.apply_critic(main, p, value)[0]["same_duplicate_group"] == "YES"
    out, rule = subject.apply_critic(main, p, review(p, conflict="NEGATION_CONFLICT"))
    assert out["same_duplicate_group"] == "NO"
    assert rule == "SUPPORTED_DIRECTIONAL_VETO"
    assert subject.apply_critic(out, p, value)[0] == out


@pytest.mark.parametrize(
    ("a", "b", "kind"),
    [
        ("Policy.\nAn independent article about birds.", "Policy.", "MAIN_ADDITION"),
        ("Policy.\nNext page", "Policy.", "NON_MAIN_ADDITION"),
        ("Service free.\nThe bus is not free.", "Service free.", "MAIN_ADDITION"),
        ('Policy.\nA visitor wrote: "lovely weather".', "Policy.", "MAIN_ADDITION"),
        ("Cup K9.\nCapacity: 200 ml.", "Cup K9.", "MAIN_ADDITION"),
    ],
)
def test_ordinary_quotation_unrelated_negation_navigation_and_independent_additions_keep_containment(a, b, kind):
    p = payload(a, b)
    out = subject.adapt_main(review(p, kind), p)
    assert (out["a_can_replace_b"], out["b_can_replace_a"]) == ("YES", "NO")
    assert out["relation_type"] == "CONTAINMENT"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Use cookies only with consent.", "Cookie使用需征得同意。"),
        ("The following claim is true: Service is free.", "Service is free."),
        ("Simple text.", "Simple   text.\n"),
    ],
)
def test_equivalent_meaning_does_not_get_keyword_or_length_veto(a, b):
    p = payload(a, b)
    out = subject.adapt_main(review(p), p)
    assert out["a_can_replace_b"] == out["b_can_replace_a"] == "YES"
    assert out["material_difference"] == "NONE"


def test_independent_bilateral_facts_are_negative_without_claiming_state_conflict():
    p = payload("Park open.\nNorth entrance closes at six.", "Park open.\nSouth entrance closes at eight.")
    out = subject.adapt_main(review(p, "MAIN_ADDITION", "MAIN_ADDITION"), p)
    assert out["same_duplicate_group"] == "NO"
    assert out["relation_type"] == "RELATED_NON_DUPLICATE"


@pytest.mark.parametrize("conflict", ["IDENTITY_CONFLICT", "STATE_CONFLICT", "POLICY_CONFLICT", "ROLE_CONFLICT"])
def test_actual_typed_conflicts_keep_their_proof_and_taxonomy(conflict):
    p = payload("Selected value A11.", "Selected value B22.")
    value = review(p, "MAIN_ADDITION", "MAIN_ADDITION", conflict=conflict)
    value["a_context_span_id"] = value["b_context_span_id"] = "S001"
    out = subject.adapt_main(value, p)
    assert out["same_duplicate_group"] == "NO"
    assert {e["quote"] for e in out["evidence"]} >= {"A11", "B22"}
    assert (out["relation_type"] == "VERSION_RELATED") == (conflict == "STATE_CONFLICT")


@pytest.mark.parametrize("damage", ["no_unique", "invented", "wrong_side", "minor", "legacy", "uncertain"])
def test_context_fix_never_waives_proof_types_severity_or_explicit_uncertainty(damage):
    p = payload("This claim is false: Service free.", "Service free.")
    value = review(p, conflict="NEGATION_CONFLICT")
    if damage == "no_unique":
        value["a_context_span_id"] = "S001"
    elif damage == "invented":
        value["b_context_span_id"] = "B999"
    elif damage == "wrong_side":
        value["b_context_span_id"] = "A001"
    elif damage == "minor":
        value["material_difference"] = "MINOR"
    elif damage == "legacy":
        value = legacy_review(p, conflict="NEGATION_CONFLICT")
    else:
        value["b_addition_kind"] = "UNRESOLVED"
        out = subject.adapt_main(value, p)
        assert out["same_duplicate_group"] == "UNRESOLVED"
        assert out["confidence_tier"] == "LOW"
        return
    with pytest.raises(DedupEvaluationError):
        subject.adapt_main(value, p)


@pytest.mark.parametrize("route", ["truncated", "capped", "exact"])
def test_deterministic_routes_do_not_reinterpret_legacy_responses(route):
    p = payload("Policy.", "Policy.")
    if route == "truncated":
        p["long_document_evidence"]["truncated"] = True
    elif route == "capped":
        p["semantic_diff_evidence"]["status"] = "INCOMPLETE_LIMIT"
    out = subject.adapt_main(None, p)
    assert out["same_duplicate_group"] == ("YES" if route == "exact" else "UNRESOLVED")
