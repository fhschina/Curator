# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from copy import deepcopy

import pytest

from eval.dedup.judging.record_scope import (
    arbitrate_record_scope,
    arbitrate_record_scope_v8,
    arbitrate_record_scope_v9,
    complete_visible_equality,
    parse_record_scope,
)
from eval.dedup.judging.retained_conflict import RetainedConflict
from eval.dedup.validation import DedupEvaluationError


def _payload() -> dict:
    return {
        "document_a": {"text": "FAQ for Model Z. Step one. Open settings."},
        "document_b": {"text": "FAQ for Model Z."},
        "long_document_evidence": {"truncated": False},
        "semantic_diff_evidence": {
            "status": "COMPLETE",
            "spans": [
                {"span_id": "S001", "kind": "SHARED"},
                {"span_id": "A001", "kind": "A_ONLY"},
                {"span_id": "A002", "kind": "A_ONLY"},
            ],
        },
    }


def _critic(scope: str = "same_specific_record", coverage: str = "S001 A001-A002 are covered.") -> dict:
    return {
        "record_scope": {"score": scope, "reasoning": "S001 independently identifies the FAQ; A001 A002 add steps."},
        "record_binding_verdict": {"score": "benign_non_record_delta", "reasoning": coverage},
    }


def _main() -> dict:
    return {
        "a_can_replace_b": "YES",
        "b_can_replace_a": "NO",
        "relation_type": "CONTAINMENT",
        "material_difference": "MAJOR",
        "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
    }


def _arbitrate(value: dict, *, main: dict | None = None, verdict: str = "BENIGN_NON_RECORD_DELTA", retained="NONE"):
    main = main or _main()
    return arbitrate_record_scope(
        main,
        parse_record_scope(value, _payload()),
        RetainedConflict(retained, [], ()),
        (verdict, ["S001", "A001", "A002"], None),
        legacy_target=main,
        legacy_rule="legacy",
        unresolved={"a_can_replace_b": "UNRESOLVED", "b_can_replace_a": "UNRESOLVED"},
    )


def test_exact_identity_requires_nonempty_complete_original_text_not_alignment_counts() -> None:
    payload = _payload()
    payload["document_b"] = deepcopy(payload["document_a"])
    assert complete_visible_equality(payload)
    payload["document_b"]["text"] = payload["document_b"]["text"].upper()
    payload["semantic_diff_evidence"]["span_counts"] = {"SHARED": 1, "A_ONLY": 0, "B_ONLY": 0}
    assert not complete_visible_equality(payload)
    for text in ("", " \n", None):
        payload["document_a"]["text"] = payload["document_b"]["text"] = text
        assert not complete_visible_equality(payload)


@pytest.mark.parametrize("failure", ["truncated", "packet", "missing_full_text", "missing_truncation_contract"])
def test_exact_identity_does_not_rescue_incomplete_evidence(failure: str) -> None:
    payload = _payload()
    payload["document_b"] = deepcopy(payload["document_a"])
    if failure == "truncated":
        payload["long_document_evidence"]["truncated"] = True
    elif failure == "packet":
        payload["semantic_diff_evidence"]["status"] = "INCOMPLETE_LIMIT"
    elif failure == "missing_full_text":
        payload["document_b"]["text"] = None
    else:
        payload.pop("long_document_evidence")
    assert not complete_visible_equality(payload)


def test_scope_requires_new_real_contract_not_fabricated_historical_proof() -> None:
    with pytest.raises(DedupEvaluationError, match="record-scope proof"):
        parse_record_scope({"record_binding_verdict": _critic()["record_binding_verdict"]}, _payload())


@pytest.mark.parametrize("scope", ["generic_context_only", "distinct_record", "list_membership_change"])
def test_record_or_membership_separation_needs_independent_agreeing_evidence(scope: str) -> None:
    value = _critic(scope)
    result, _ = _arbitrate(value, verdict="SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT")
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"
    assert result["primary_material_difference"] == (
        "RESULT_SET_CHANGE" if scope == "list_membership_change" else "DOCUMENT_IDENTITY_CHANGE"
    )
    result, _ = _arbitrate(value, verdict="ATOMIC_SAME_RECORD_EXTENSION")
    assert result["a_can_replace_b"] == "UNRESOLVED"
    value["record_scope"]["reasoning"] = "S001 is generic."
    result, _ = _arbitrate(value, verdict="SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT")
    assert result["a_can_replace_b"] == "UNRESOLVED"


@pytest.mark.parametrize("scope", ["same_specific_record", "same_organization_description"])
def test_true_bound_extension_keeps_its_direction(scope: str) -> None:
    result, rule = _arbitrate(_critic(scope), verdict="ATOMIC_SAME_RECORD_EXTENSION")
    assert result == _main()
    assert rule == "legacy"


@pytest.mark.parametrize("scope", ["equivalent_complete_message", "not_applicable", "unresolved"])
def test_non_record_or_missing_identity_cannot_confirm_containment(scope: str) -> None:
    result, _ = _arbitrate(_critic(scope), verdict="ATOMIC_SAME_RECORD_EXTENSION")
    assert result["a_can_replace_b"] == "UNRESOLVED"


def test_only_fully_cited_benign_coverage_can_reclassify_main_extension() -> None:
    result, rule = _arbitrate(_critic())
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "YES"
    assert result["relation_type"] == "NEAR_SURFACE"
    assert rule == "RECORD_SCOPE_VERIFIED_BENIGN_EQUIVALENCE"
    for coverage in ("S001 A001 are covered.", "S001 A001-A003 are covered.", "S001 A002-A001 are covered."):
        result, _ = _arbitrate(_critic(coverage=coverage))
        assert result["a_can_replace_b"] == "UNRESOLVED"
    result, _ = _arbitrate(_critic(), retained="IDENTITY_OR_SERVICE_TARGET_CHANGE")
    assert result["a_can_replace_b"] == "UNRESOLVED"


@pytest.mark.parametrize("direction", ["NO", "UNRESOLVED"])
def test_scope_cannot_rescue_a_conflicting_or_unresolved_main(direction: str) -> None:
    main = {"a_can_replace_b": direction, "b_can_replace_a": direction, "relation_type": "UNRESOLVED"}
    assert _arbitrate(_critic(), main=main)[0] == main


def _v8(*, main=None, target=None, scope=None, verdict="BENIGN_NON_RECORD_DELTA", issue=None, ledger=None):  # noqa: PLR0913
    return arbitrate_record_scope_v8(
        main or _main(),
        scope or parse_record_scope(_critic(), _payload()),
        RetainedConflict("NONE", [], ()),
        (verdict, ["S001", "A001", "A002"], issue),
        ledger=ledger or {},
        legacy_target=target or main or _main(),
        legacy_rule="owned",
        unresolved={"a_can_replace_b": "UNRESOLVED", "b_can_replace_a": "UNRESOLVED"},
    )


def test_v8_preserves_valid_negative_when_advisory_scope_omits_delta() -> None:
    value = _critic("generic_context_only")
    value["record_scope"]["reasoning"] = "S001 is generic."
    target = {"a_can_replace_b": "NO", "b_can_replace_a": "NO", "relation_type": "RELATED_NON_DUPLICATE"}
    assert _v8(target=target, scope=parse_record_scope(value, _payload())) == (target, "owned")


@pytest.mark.parametrize("direction", ["YES", "NO", "UNRESOLVED"])
def test_v8_non_main_scope_cannot_bypass_retained_conflict_owner(direction: str) -> None:
    target = {"a_can_replace_b": direction, "b_can_replace_a": direction}
    result, rule = _v8(
        target=target,
        scope=parse_record_scope(_critic("generic_context_only"), _payload()),
        verdict="SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
        ledger={"span_content_profile_a": "NON_MAIN_ONLY", "span_content_profile_b": "NON_MAIN_ONLY"},
    )
    assert (result, rule) == (target, "owned")


def test_v8_citation_coverage_does_not_erase_a_known_instruction_or_faq_addition() -> None:
    result, rule = _v8()
    assert result == {**_main(), "confidence_tier": "LOW"}
    assert rule == "RECORD_SCOPE_EXTENSION_BENIGN_DISAGREEMENT_PRESERVED"
    # Historical v7 keeps its original arbitration for immutable replay.
    assert _arbitrate(_critic())[0]["b_can_replace_a"] == "YES"


def test_v8_partial_translation_preserves_owned_extension_without_inventing_proof() -> None:
    main = {**_main(), "confidence_tier": "LOW"}
    result, rule = _v8(
        main=main,
        verdict="ATOMIC_SAME_RECORD_EXTENSION",
        issue="MISSING_BILATERAL_BASIS",
        ledger={"span_translation_status": "PARTIAL_OR_ADDITIVE"},
    )
    assert (result, rule) == (main, "owned")
    unresolved = {"a_can_replace_b": "UNRESOLVED", "b_can_replace_a": "UNRESOLVED"}
    assert (
        _v8(target=unresolved, verdict="ATOMIC_SAME_RECORD_EXTENSION", issue="MISSING_BILATERAL_BASIS")[0]
        == unresolved
    )


def test_v8_still_rejects_proven_separate_attachment_and_abstains_on_unbound_extension() -> None:
    scope = parse_record_scope(_critic("generic_context_only"), _payload())
    assert _v8(scope=scope, verdict="SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT")[0]["a_can_replace_b"] == "NO"
    assert _v8(scope=scope, verdict="ATOMIC_SAME_RECORD_EXTENSION")[0]["a_can_replace_b"] == "UNRESOLVED"


def _v9(critic, review, *, target=None, main=None):
    main = main or {**_main(), "b_can_replace_a": "YES", "relation_type": "NEAR_SURFACE"}
    return arbitrate_record_scope_v9(
        main,
        parse_record_scope(_critic(), _payload()),
        review,
        critic,
        ledger={"span_content_profile_a": "SUBSTANTIVE_MAIN", "span_content_profile_b": "SUBSTANTIVE_MAIN"},
        legacy_target=target or main,
        legacy_rule="legacy",
        unresolved={"a_can_replace_b": "UNRESOLVED", "b_can_replace_a": "UNRESOLVED"},
    )


def test_v9_cited_permission_change_is_not_ignored_inside_substantive_text():
    result, rule = _v9(
        ("NON_MAIN_POLICY_OR_STATE_CHANGE", ["S001", "A001"], None),
        RetainedConflict("POLICY_PERMISSION_CHANGE", ["S001", "A001"], ()),
    )
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"
    assert result["primary_material_difference"] == "LEGAL_CONTEXT_CHANGE"
    assert rule == "RETENTION_MATERIAL_VETO_POLICY_PERMISSION_CHANGE"


@pytest.mark.parametrize("failure", ["missing_own_citation", "binding_disagrees"])
def test_v9_does_not_invent_a_veto_from_an_inconsistent_permission_claim(failure):
    verdict = "ATOMIC_SAME_RECORD_EXTENSION" if failure == "binding_disagrees" else "NON_MAIN_POLICY_OR_STATE_CHANGE"
    issues = ("MISSING_BILATERAL_BASIS",) if failure == "missing_own_citation" else ()
    result, _ = _v9((verdict, ["S001", "A001"], None), RetainedConflict("POLICY_PERMISSION_CHANGE", ["A001"], issues))
    assert result["a_can_replace_b"] == "UNRESOLVED"


def test_v9_equivalence_and_proved_extension_disagreement_cannot_guess_direction():
    result, rule = _v9(("ATOMIC_SAME_RECORD_EXTENSION", ["S001", "A001"], None), RetainedConflict("NONE", [], ()))
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "UNRESOLVED"
    assert rule == "RETENTION_EQUIVALENCE_EXTENSION_DISAGREEMENT"
    result, _ = _v9(
        ("ATOMIC_SAME_RECORD_EXTENSION", ["S001", "A001"], None), RetainedConflict("NONE", [], ()), main=_main()
    )
    assert result == _main()


@pytest.mark.parametrize("direction", ["NO", "UNRESOLVED"])
def test_v9_never_reopens_owned_negative_or_unknown_decisions(direction):
    target = {"a_can_replace_b": direction, "b_can_replace_a": direction}
    result, _ = _v9(
        ("ATOMIC_SAME_RECORD_EXTENSION", ["S001", "A001"], None),
        RetainedConflict("POLICY_PERMISSION_CHANGE", [], ()),
        target=target,
    )
    assert result == target
