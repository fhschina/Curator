"""Literal constraints cannot credit an internally contradictory positive judgment."""
# ruff: noqa: S101, PLR2004

import copy

import pytest
from ledger_audit_core import ledger_issues
from policy_tuning import agreement


def sample(a: str = "yes", b: str = "yes") -> dict:
    decision = {
        "a_can_replace_b": a,
        "b_can_replace_a": b,
        "relation_type": "near_surface" if a == b == "yes" else "related_non_duplicate",
        "material_difference": "none" if a == b == "yes" else "major",
        "primary_material_difference": "none" if a == b == "yes" else "other_material",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
    }
    values = {
        **decision,
        "span_a_delta": "universal_ui_or_repetition",
        "span_b_delta": "semantically_covered",
        "span_hard_conflict": "none",
        "span_shared_basis": "verified_substantive_record",
        "span_translation_status": "not_translation",
        "span_content_profile_a": "substantive_main",
        "span_content_profile_b": "substantive_main",
    }
    return {
        "final_decision": decision,
        "pair_semantic_judgment": {
            key: {"score": value, "reasoning": "Evidence checked."} for key, value in values.items()
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("span_b_delta", "same_record_content_extension"),
        ("span_a_delta", "material_non_main_message_change"),
        ("span_hard_conflict", "legal_context"),
        ("span_shared_basis", "none"),
    ],
)
def test_positive_contradiction_loses_credit_without_rewriting_the_answer(field: str, value: str) -> None:
    row = sample()
    row["pair_semantic_judgment"][field]["score"] = value
    before = copy.deepcopy(row)
    issues = ledger_issues(row)
    assert issues
    scored = agreement(
        {"item": row},
        [{"pair_id": "item", "a_can_replace_b": "yes", "b_can_replace_a": "yes"}],
        {"item": issues},
        "final",
    )
    assert scored["pairs"] == 1
    assert scored["directions"] == 2
    assert scored["correct_directions"] == 0
    assert scored["invalid_pairs"] == 1
    assert row == before


def test_protected_body_translation_negative_is_not_a_contradiction() -> None:
    row = sample("no", "no")
    row["pair_semantic_judgment"]["span_translation_status"]["score"] = "complete_faithful"
    assert not ledger_issues(row)


def test_harmless_localized_positive_keeps_its_original_credit() -> None:
    row = sample()
    row["pair_semantic_judgment"]["span_translation_status"]["score"] = "complete_faithful"
    assert not ledger_issues(row)
    assert (
        agreement(
            {"item": row}, [{"pair_id": "item", "a_can_replace_b": "yes", "b_can_replace_a": "yes"}], {}, "final"
        )["correct_directions"]
        == 2
    )


@pytest.mark.parametrize("field", ["a_can_replace_b", "b_can_replace_a"])
def test_negative_with_covered_removed_side_is_invalid_without_changing_raw_signals(field: str) -> None:
    row = sample()
    row["pair_semantic_judgment"][field]["score"] = "no"
    row["final_decision"][field] = "no"
    before = copy.deepcopy(row)
    issues = ledger_issues(row)
    assert issues == ["MAIN_LEDGER_CONTRADICTION:" + field + ":covered_removed_side"]
    labels = [{"pair_id": "item", "a_can_replace_b": "yes", "b_can_replace_a": "yes"}]
    native = agreement({"item": row}, labels, {}, "final")
    checked = agreement({"item": row}, labels, {"item": issues}, "final")
    assert checked["correct_directions"] == 0
    assert checked["directions"] == native["directions"] == 2
    assert checked["reference_wrong_blocks"] == native["reference_wrong_blocks"] == 1
    assert row == before


@pytest.mark.parametrize("translation", ["complete_faithful", "partial_or_additive", "unresolved"])
def test_language_retention_is_not_inferred_harmless_from_semantic_coverage(translation: str) -> None:
    row = sample("no", "no")
    row["pair_semantic_judgment"]["span_translation_status"]["score"] = translation
    assert not ledger_issues(row)


def test_actual_material_loss_and_missing_basis_remain_consistent_negatives() -> None:
    row = sample("no", "yes")
    row["pair_semantic_judgment"]["span_b_delta"]["score"] = "same_record_content_extension"
    assert not ledger_issues(row)
    row = sample("no", "no")
    row["pair_semantic_judgment"]["span_shared_basis"]["score"] = "none"
    assert not ledger_issues(row)
