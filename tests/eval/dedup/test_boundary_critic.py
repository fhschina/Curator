# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import pytest

from eval.dedup.judging.boundary_critic import arbitrate_boundary_review, parse_boundary_review
from eval.dedup.validation import DedupEvaluationError


def _packet() -> dict:
    return {
        "semantic_diff_evidence": {
            "status": "COMPLETE",
            "spans": [
                {"span_id": "S001", "kind": "SHARED"},
                {"span_id": "A001", "kind": "A_ONLY"},
                {"span_id": "B001", "kind": "B_ONLY"},
            ],
        }
    }


def _review(**overrides: tuple[str, str]) -> dict:
    fields = {
        "non_main_delta_subtype": ("NOT_APPLICABLE", "No policy boundary."),
        "translation_delta_direction": ("NOT_TRANSLATION", "Same language."),
        "record_binding_verdict": (
            "ATOMIC_SAME_RECORD_EXTENSION",
            "S001 binds A001's extra instruction; B001 rewords it.",
        ),
        **overrides,
    }
    return {key: {"score": score, "reasoning": reason} for key, (score, reason) in fields.items()}


def _ledger(**overrides: str) -> dict:
    return {
        "span_content_profile_a": "SUBSTANTIVE_MAIN",
        "span_content_profile_b": "SUBSTANTIVE_MAIN",
        "span_shared_basis": "VERIFIED_SUBSTANTIVE_RECORD",
        "span_hard_conflict": "NONE",
        "span_translation_status": "NOT_TRANSLATION",
        "span_a_delta": "SAME_RECORD_CONTENT_EXTENSION",
        "span_b_delta": "SEMANTICALLY_COVERED",
        **overrides,
    }


def _main() -> dict:
    return {"a_can_replace_b": "YES", "b_can_replace_a": "NO", "relation_type": "CONTAINMENT"}


def _arbitrate(value: dict, *, main: dict | None = None, ledger: dict | None = None) -> tuple[dict, str]:
    return arbitrate_boundary_review(
        main or _main(), ledger or _ledger(), parse_boundary_review(value, _packet()), {"relation_type": "UNRESOLVED"}
    )


@pytest.mark.parametrize(
    "subtype",
    [
        "POLICY_PROPOSITION_CHANGE",
        "CONSENT_OR_LEGAL_STATE_CHANGE",
        "COOKIE_INVENTORY_CHANGE",
        "PAGE_CONTEXT_OR_IDENTITY_CHANGE",
    ],
)
def test_specialized_material_boundary_vetoes_even_substantive_inventory(subtype: str) -> None:
    result, _ = _arbitrate(_review(non_main_delta_subtype=(subtype, "S001 is context; A001 changes it.")))
    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["material_difference"] == "MAJOR"


@pytest.mark.parametrize("reason", ["S001", "A001", "A001 B999", ""])
def test_translation_cannot_borrow_citations_from_record_verdict(reason: str) -> None:
    review = parse_boundary_review(_review(translation_delta_direction=("B_ADDS", reason)), _packet())
    assert review.issues


def test_chrome_requires_exhaustive_own_delta_coverage() -> None:
    incomplete = _review(record_binding_verdict=("CHROME_OR_REDUNDANCY_ONLY", "S001 A001 are UI."))
    complete = _review(record_binding_verdict=("CHROME_OR_REDUNDANCY_ONLY", "S001 A001 B001 cover only chrome."))
    assert _arbitrate(incomplete)[0]["relation_type"] == "UNRESOLVED"
    result, _ = _arbitrate(complete)
    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")


@pytest.mark.parametrize(("direction", "expected"), [("A_ADDS", ("YES", "NO")), ("B_ADDS", ("NO", "YES"))])
def test_bilingual_fact_delta_corrects_false_complete_translation(direction: str, expected: tuple) -> None:
    result, _ = _arbitrate(
        _review(translation_delta_direction=(direction, "A001 and B001 align the subject but add a service.")),
        main={"a_can_replace_b": "YES", "b_can_replace_a": "YES", "relation_type": "NEAR_SURFACE"},
        ledger=_ledger(span_translation_status="COMPLETE_FAITHFUL", span_a_delta="SEMANTICALLY_COVERED"),
    )
    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == expected
    assert result["primary_material_difference"] == "MAIN_CONTENT_ADDITION_DELETION"


def test_semantic_equivalence_does_not_erase_a_cited_instruction() -> None:
    result, _ = _arbitrate(_review(record_binding_verdict=("SEMANTIC_EQUIVALENCE", "A001 B001 mean the same.")))
    assert result["relation_type"] == "CONTAINMENT"
    assert result["confidence_tier"] == "LOW"


def test_disagreeing_translation_directions_fail_closed() -> None:
    result, _ = _arbitrate(
        _review(translation_delta_direction=("B_ADDS", "A001 B001 align but B adds a fact.")),
        ledger=_ledger(span_translation_status="PARTIAL_OR_ADDITIVE"),
    )
    assert result["relation_type"] == "UNRESOLVED"


def test_translation_cannot_rescue_a_negative_main_decision() -> None:
    negative = {"a_can_replace_b": "NO", "b_can_replace_a": "NO", "relation_type": "RELATED_NON_DUPLICATE"}
    result, _ = _arbitrate(
        _review(translation_delta_direction=("B_ADDS", "A001 B001 align but B adds.")),
        main=negative,
        ledger=_ledger(span_translation_status="PARTIAL_OR_ADDITIVE"),
    )
    assert result == negative


def test_missing_critic_axis_is_a_contract_error() -> None:
    value = _review()
    del value["translation_delta_direction"]
    with pytest.raises(DedupEvaluationError):
        parse_boundary_review(value, _packet())


def test_incomplete_packet_cannot_be_upgraded() -> None:
    packet = _packet()
    packet["semantic_diff_evidence"]["status"] = "TRUNCATED"
    review = parse_boundary_review(_review(), packet)
    result, _ = arbitrate_boundary_review(_main(), _ledger(), review, {"relation_type": "UNRESOLVED"})
    assert result["relation_type"] == "UNRESOLVED"
