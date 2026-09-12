# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import pytest

from eval.dedup.judging.retained_conflict import (
    MATERIAL_CONFLICTS,
    arbitrate_retained_conflict,
    parse_retained_conflict,
)
from eval.dedup.validation import DedupEvaluationError


def _review(score: str, reasoning: str = "S001 is the shared disclaimer; B001 adds an actual water-heater service."):
    return parse_retained_conflict(
        {"retained_conflict": {"score": score, "reasoning": reasoning}},
        {
            "semantic_diff_evidence": {
                "status": "COMPLETE",
                "spans": [
                    {"span_id": "S001", "kind": "SHARED"},
                    {"span_id": "B001", "kind": "B_ONLY"},
                ],
            }
        },
    )


def _arbitrate(
    review,
    *,
    profile: str = "NON_MAIN_ONLY",
    verdict: str = "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
    positive: bool = True,
):
    main = {
        "a_can_replace_b": "YES" if positive else "NO",
        "b_can_replace_a": "YES" if positive else "NO",
        "relation_type": "NEAR_SURFACE" if positive else "RELATED_NON_DUPLICATE",
    }
    legacy = (
        main
        if profile == "NON_MAIN_ONLY"
        else {"a_can_replace_b": "NO", "b_can_replace_a": "NO", "relation_type": "RELATED_NON_DUPLICATE"}
    )
    return arbitrate_retained_conflict(
        main,
        {"span_content_profile_a": profile, "span_content_profile_b": profile},
        review,
        critic_verdict=verdict,
        legacy_target=legacy,
        legacy_rule="LEGACY",
        unresolved={"a_can_replace_b": "UNRESOLVED", "b_can_replace_a": "UNRESOLVED", "confidence_tier": "LOW"},
    )


@pytest.mark.parametrize("score", list(MATERIAL_CONFLICTS))
def test_material_non_main_override_requires_own_bilateral_delta_evidence(score: str) -> None:
    result, rule = _arbitrate(_review(score))
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"
    assert result["material_difference"] == "MAJOR"
    assert result["primary_material_difference"] == MATERIAL_CONFLICTS[score][0]
    assert rule == f"RETAINED_CONFLICT_VETO_{score}"


@pytest.mark.parametrize(
    "reason", ["S001 only", "B001 only", "S001 B999", "S001 B001-B003", "S001 B003-B001", "S001 A001-B001"]
)
def test_unsupported_active_warning_abstains_instead_of_becoming_a_veto(reason: str) -> None:
    result, _ = _arbitrate(_review("PAGE_ROLE_CHANGE", reason))
    assert result["a_can_replace_b"] == "UNRESOLVED"
    assert result["confidence_tier"] == "LOW"


@pytest.mark.parametrize("score", ["NONE", "NOT_APPLICABLE"])
def test_generic_critic_warning_cannot_turn_benign_controls_into_policy_veto(score: str) -> None:
    result, _ = _arbitrate(
        _review(score, "S001 B001 same meaning with an Accept button."), verdict="NON_MAIN_POLICY_OR_STATE_CHANGE"
    )
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "YES"


def test_material_warning_and_positive_binding_disagreement_is_unresolved() -> None:
    result, _ = _arbitrate(_review("PAGE_ROLE_CHANGE"), verdict="BENIGN_NON_RECORD_DELTA")
    assert result["a_can_replace_b"] == "UNRESOLVED"


@pytest.mark.parametrize("score", ["NONE", "UNRESOLVED", "PAGE_ROLE_CHANGE"])
def test_non_main_supplement_never_reopens_substantive_record_veto(score: str) -> None:
    result, rule = _arbitrate(_review(score), profile="SUBSTANTIVE_MAIN")
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"
    assert rule == "LEGACY"


def test_inactive_invalid_proof_does_not_cancel_a_valid_main_negative() -> None:
    result, rule = _arbitrate(_review("PAGE_ROLE_CHANGE", "A999"), positive=False)
    assert result["a_can_replace_b"] == "NO"
    assert rule == "LEGACY"


def test_retained_conflict_is_required_and_categorical() -> None:
    for value in (None, {}, {"retained_conflict": {"score": "guess", "reasoning": "S001"}}):
        with pytest.raises(DedupEvaluationError, match="LOCAL_NDD_OUTPUT_INVALID"):
            parse_retained_conflict(value, {})
