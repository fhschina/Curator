# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import pytest

from eval.dedup.judging.scoped_critic import arbitrate_scoped_review, expand_citations, parse_scoped_review


def _packet() -> dict:
    return {
        "semantic_diff_evidence": {
            "status": "COMPLETE",
            "spans": [
                {"span_id": s, "kind": k}
                for s, k in [("S001", "SHARED"), ("A001", "A_ONLY"), ("A002", "A_ONLY"), ("B001", "B_ONLY")]
            ],
        }
    }


def _review(**overrides: tuple[str, str]) -> dict:
    return {
        k: {"score": score, "reasoning": reason}
        for k, (score, reason) in {
            "record_binding_verdict": ("atomic_same_record_extension", "S001 binds A001-A002; B001 is covered."),
            "non_main_delta_subtype": ("not_applicable", "The main instruction decides."),
            "translation_delta_direction": ("not_translation", "Same language."),
            **overrides,
        }.items()
    }


def _main() -> dict:
    return {"a_can_replace_b": "YES", "b_can_replace_a": "NO", "relation_type": "CONTAINMENT"}


def _arbitrate(
    value: dict,
    *,
    main: dict | None = None,
    translation: str = "NOT_TRANSLATION",
    profile: str = "SUBSTANTIVE_MAIN",
    fallback: dict | None = None,
) -> tuple[dict, str]:
    main = main or _main()
    return arbitrate_scoped_review(
        main,
        {
            "span_content_profile_a": profile,
            "span_content_profile_b": profile,
            "span_shared_basis": "VERIFIED_SUBSTANTIVE_RECORD",
            "span_translation_status": translation,
        },
        parse_scoped_review(value, _packet()),
        legacy_target=fallback or main,
        legacy_rule="legacy",
        unresolved={"relation_type": "UNRESOLVED"},
    )


@pytest.mark.parametrize("text", ["A001-A002", "A001\u2013A002", "A001 through A002", "A001 to 002"])
def test_explicit_existing_ranges_expand(text: str) -> None:
    assert expand_citations(text, {"A001", "A002"}) == (["A001", "A002"], ())


@pytest.mark.parametrize("text", ["A002-A001", "A001-B001", "A001-A003", "B999"])
def test_unknown_reversed_or_cross_side_ranges_fail(text: str) -> None:
    assert expand_citations(text, {"A001", "A002", "B001"})[1]


def test_negative_main_is_not_invalidated_by_irrelevant_translation_evidence() -> None:
    negative = {"a_can_replace_b": "NO", "b_can_replace_a": "NO", "relation_type": "RELATED_NON_DUPLICATE"}
    result, _ = _arbitrate(_review(translation_delta_direction=("b_adds", "B999")), main=negative)
    assert result == negative


def test_valid_policy_veto_ignores_unrelated_translation_missing_side() -> None:
    result, _ = _arbitrate(
        _review(
            non_main_delta_subtype=("cookie_inventory_change", "S001 A001 change inventory."),
            translation_delta_direction=("a_adds", "S001 A001"),
        )
    )
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"


def test_material_policy_claim_with_missing_evidence_is_not_ignored() -> None:
    result, _ = _arbitrate(_review(non_main_delta_subtype=("cookie_inventory_change", "A001 only.")))
    assert result["relation_type"] == "UNRESOLVED"


def test_same_language_addition_does_not_activate_translation_even_when_critic_says_adds() -> None:
    result, _ = _arbitrate(_review(translation_delta_direction=("a_adds", "A001 B001")))
    assert result == _main()


def test_not_applicable_non_main_axis_preserves_independently_verified_message() -> None:
    equivalent = {"a_can_replace_b": "YES", "b_can_replace_a": "YES", "relation_type": "NEAR_SURFACE"}
    result, _ = _arbitrate(_review(), main=equivalent, profile="NON_MAIN_ONLY")
    assert result == equivalent


def test_valid_bilingual_direction_does_not_depend_on_redundant_record_citation() -> None:
    result, _ = _arbitrate(
        _review(
            translation_delta_direction=("a_adds", "A001 B001 align; A002 adds."),
            record_binding_verdict=("atomic_same_record_extension", "A002 only."),
        ),
        translation="PARTIAL_OR_ADDITIVE",
        fallback={"relation_type": "UNRESOLVED"},
    )
    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "NO")


def test_opposite_bilingual_direction_still_fails_closed() -> None:
    result, _ = _arbitrate(
        _review(translation_delta_direction=("b_adds", "A001 B001 align.")), translation="PARTIAL_OR_ADDITIVE"
    )
    assert result["relation_type"] == "UNRESOLVED"


def test_chrome_can_flatten_only_when_supplement_accounts_for_all_unique_spans() -> None:
    value = _review(
        record_binding_verdict=("benign_non_record_delta", "S001 A001-A002 B001 are UI."),
        non_main_delta_subtype=("equivalent_message_or_wrapper", "S001 A001-A002 B001 are only chrome."),
    )
    result, _ = _arbitrate(value)
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "YES"
    value["non_main_delta_subtype"]["reasoning"] = "S001 A001 is chrome; the rest is not assessed."
    assert _arbitrate(value)[0] == _main()


def test_policy_only_ablation_does_not_require_a_translation_output() -> None:
    value = _review()
    del value["translation_delta_direction"]
    assert "translation_delta_direction" not in parse_scoped_review(value, _packet(), translation_enabled=False).scores


def test_incomplete_packet_and_main_unresolved_cannot_be_rescued() -> None:
    packet = _packet()
    packet["semantic_diff_evidence"]["status"] = "TRUNCATED"
    review = parse_scoped_review(_review(), packet)
    result, _ = arbitrate_scoped_review(
        _main(), {}, review, legacy_target=_main(), legacy_rule="legacy", unresolved={"relation_type": "UNRESOLVED"}
    )
    assert result["relation_type"] == "UNRESOLVED"
    assert _arbitrate(_review(), main={"relation_type": "UNRESOLVED"})[0]["relation_type"] == "UNRESOLVED"
