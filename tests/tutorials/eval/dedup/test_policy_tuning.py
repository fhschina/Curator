# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Development agreement preserves failed rows and coherent accepted alternatives."""

import copy
import re
from fractions import Fraction
from pathlib import Path

import pytest
import yaml
from policy_cases import aligned, blind_row
from policy_tuning import aggregate, agreement, nonregression_status, target_status


def personal_summary(directions: list[int], pairs: list[int]) -> dict:
    return {
        "repeats": [str(i) for i in range(len(directions))],
        "denominator_pairs": [120],
        "direction_agreement": {"correct_total": sum(directions), "denominator_total": 240 * len(directions)},
        "pair_agreement": {"correct_total": sum(pairs), "denominator_total": 120 * len(pairs)},
    }


def test_overnight_target_uses_exact_three_run_means_without_an_each_run_gate():
    summary = personal_summary([215, 216, 217], [101, 102, 103])
    assert target_status(summary, Fraction(9, 10), Fraction(17, 20)) == "PASS"
    assert (
        target_status(personal_summary([216, 216, 215], [102, 102, 102]), Fraction(9, 10), Fraction(17, 20)) == "FAIL"
    )
    assert (
        target_status(personal_summary([216, 216, 216], [102, 102, 101]), Fraction(9, 10), Fraction(17, 20)) == "FAIL"
    )


def test_old_target_and_incomplete_repeats_do_not_satisfy_overnight_goal():
    assert target_status(personal_summary([192] * 3, [90] * 3), Fraction(9, 10), Fraction(17, 20)) == "FAIL"
    assert target_status(personal_summary([240] * 2, [120] * 2), Fraction(9, 10), Fraction(17, 20)) == "NOT_COMPLETED"
    changed = personal_summary([240] * 3, [120] * 3)
    changed["denominator_pairs"] = [119]
    with pytest.raises(ValueError, match="denominator changed"):
        target_status(changed, Fraction(9, 10), Fraction(17, 20))


def test_nonregression_compares_exact_counts_and_requires_complete_repeats():
    baseline = personal_summary([224, 224, 224], [110] * 3)
    candidate = personal_summary([223, 224, 225], [109] * 3)
    assert nonregression_status(candidate, baseline) == "PASS"
    candidate["direction_agreement"]["correct_total"] -= 1
    assert nonregression_status(candidate, baseline) == "FAIL"
    assert nonregression_status(personal_summary([240], [120]), baseline) == "NOT_COMPLETED"


def test_missing_declared_repeats_earn_zero_credit_in_the_full_fixed_denominator():
    labels = [{"pair_id": str(index), "a_can_replace_b": "yes", "b_can_replace_a": "yes"} for index in range(120)]
    metrics = agreement({row["pair_id"]: output() for row in labels}, labels, {}, "final")
    runs = [{"run": "r1", "groups": {"personal120": dict.fromkeys(("main", "coverage", "final"), metrics)}}]
    summary = aggregate(runs, "personal120", ["r1", "r2", "r3"], 120)["final"]
    assert summary["direction_agreement"]["correct_total"] == 240
    assert summary["direction_agreement"]["denominator_total"] == 720
    assert summary["pair_agreement"]["correct_total"] == 120
    assert summary["pair_agreement"]["denominator_total"] == 360
    assert summary["missing_repeats"] == ["r2", "r3"]
    assert target_status(summary, Fraction(9, 10), Fraction(17, 20)) == "NOT_COMPLETED"
    with pytest.raises(ValueError, match="counted twice"):
        aggregate(runs + runs, "personal120", ["r1", "r2", "r3"], 120)
    with pytest.raises(ValueError, match="predeclared group"):
        aggregate(runs, "personal120", ["replacement", "r2", "r3"], 120)


def output(a: str = "yes", b: str = "yes"):
    value = {
        "a_can_replace_b": a,
        "b_can_replace_a": b,
        "relation_type": "near_surface" if a == b == "yes" else "related_non_duplicate",
        "material_difference": "none" if a == b == "yes" else "major",
        "primary_material_difference": "none" if a == b == "yes" else "other_material",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
    }
    return {"final_decision": value}


def test_failures_and_unresolved_remain_in_fixed_denominator():
    rows = {"valid": output(), "invalid": output(), "unresolved": output()}
    rows["unresolved"]["final_decision"] = {
        **dict.fromkeys(rows["unresolved"]["final_decision"], "unresolved"),
        "confidence_tier": "low",
        "primary_risk_factor": "extraction_or_payload_limit",
    }
    labels = [{"pair_id": key, "a_can_replace_b": "yes", "b_can_replace_a": "yes"} for key in rows]
    result = agreement(rows, labels, {"invalid": ["SUBJECT_UNVALIDATED"]}, "final")
    assert result["pairs"] == 3
    assert result["directions"] == 6
    assert result["correct_directions"] == 2
    assert result["correct_pairs"] == 1
    assert result["invalid_pairs"] == 1
    assert result["unresolved_pairs"] == 1
    assert result["valid_judgment_coverage"] == 1 / 3


def test_accepted_alternatives_cannot_mix_directions_from_incompatible_pairs():
    # yes/no is coherent containment, but it is neither allowed bilateral answer.
    row = output()
    row["final_decision"].update(
        b_can_replace_a="no",
        relation_type="containment",
        material_difference="major",
        primary_material_difference="main_content_addition_deletion",
    )
    labels = [{"pair_id": "ambiguous", "a_can_replace_b": "no", "b_can_replace_a": "no"}]
    accepted = {
        "ambiguous": [
            {"a_can_replace_b": "yes", "b_can_replace_a": "yes"},
            {"a_can_replace_b": "no", "b_can_replace_a": "no"},
        ]
    }
    result = agreement({"ambiguous": row}, labels, {}, "final", accepted)
    assert result["correct_directions"] == 1
    assert result["correct_pairs"] == 0
    unchanged = copy.deepcopy(labels)
    agreement({"ambiguous": output()}, labels, {}, "final", accepted)
    assert labels == unchanged


def test_raw_unsafe_allow_is_visible_even_when_judgment_is_invalid():
    labels = [{"pair_id": "failed", "a_can_replace_b": "no", "b_can_replace_a": "no"}]
    result = agreement({"failed": output()}, labels, {"failed": ["SUBJECT_UNVALIDATED"]}, "final")
    assert result["reference_wrong_allows"] == 2
    assert result["correct_directions"] == 0
    assert result["invalid_pairs"] == 1


def test_unresolved_reference_is_not_silently_treated_as_a_binary_label():
    labels = [{"pair_id": "undecided", "a_can_replace_b": "unresolved", "b_can_replace_a": "yes"}]
    with pytest.raises(ValueError, match="explicit binary references"):
        agreement({"undecided": output()}, labels, {}, "final")


def test_accepted_raw_answer_with_technical_failure_is_not_a_semantic_error():
    labels = [{"pair_id": "item", "a_can_replace_b": "yes", "b_can_replace_a": "yes"}]
    accepted = {
        "item": [
            {"a_can_replace_b": "yes", "b_can_replace_a": "yes"},
            {"a_can_replace_b": "no", "b_can_replace_a": "no"},
        ]
    }
    result = agreement({"item": output("no", "no")}, labels, {"item": ["SUBJECT_UNVALIDATED"]}, "final", accepted)
    assert result["correct_directions"] == 0
    assert result["invalid_pairs"] == 1
    assert result["reference_wrong_allows"] == 0
    assert result["reference_wrong_blocks"] == 0


def test_repeat_summary_preserves_worst_run_and_failed_row_coverage():
    labels = [{"pair_id": "item", "a_can_replace_b": "yes", "b_can_replace_a": "yes"}]
    passing = agreement({"item": output()}, labels, {}, "final")
    failed = agreement({"item": output()}, labels, {"item": ["SUBJECT_UNVALIDATED"]}, "final")
    reports = [
        {"run": str(index), "groups": {"regression": dict.fromkeys(("main", "coverage", "final"), value)}}
        for index, value in enumerate((passing, failed, passing))
    ]
    summary = aggregate(reports, "regression")["final"]
    assert summary["direction_agreement"]["mean"] == 2 / 3
    assert summary["direction_agreement"]["worst"] == 0
    assert summary["valid_judgment_coverage"]["each"] == [1, 0, 1]
    assert summary["denominator_pairs"] == [1]


def test_equal_integer_totals_cannot_appear_as_a_mean_regression():
    labels = [{"pair_id": str(index), "a_can_replace_b": "yes", "b_can_replace_a": "yes"} for index in range(97)]

    def trial(correct_pairs: int, repeat: int) -> dict:
        rows = {str(index): output() if index < correct_pairs else output("no", "no") for index in range(97)}
        metrics = agreement(rows, labels, {}, "final")
        return {"run": str(repeat), "groups": {"fixed": dict.fromkeys(("main", "coverage", "final"), metrics)}}

    baseline = aggregate([trial(93, repeat) for repeat in range(3)], "fixed")["final"]
    candidate = aggregate([trial(count, repeat) for repeat, count in enumerate((92, 92, 95))], "fixed")["final"]
    assert candidate["direction_agreement"]["mean"] == baseline["direction_agreement"]["mean"] == 558 / 582
    assert candidate["pair_agreement"]["mean"] == baseline["pair_agreement"]["mean"] == 279 / 291


def test_native_prompt_preserves_aligned_evidence_and_window():
    from critics.spans import SpanPacket
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
    )
    from data_designer.engine.models.recipes.response_recipes import TextResponseRecipe

    template = (Path(__file__).parents[4] / "tutorials/eval/dedup/judge_config/pair.jinja").read_text()
    renderer = RecordBasedPromptRenderer(TextResponseRecipe())
    sources = aligned(
        [
            {
                "pair_id": "localized_heading",
                "text_a": "Library information\nThe library opens at 10:00. Entry costs 8 euros.",
                "text_b": "Informations bibliothèque\nThe library opens at 10:00. Entry costs 8 euros.",
            },
            {"pair_id": "limited", "text_a": "a " * 3000 + "UNSEEN_CONTINUATION", "text_b": "a " * 3000},
        ]
    )
    for row in sources:
        rendered = renderer.render(prompt_template=template, record=blind_row(row), prompt_type=PromptType.USER_PROMPT)
        packet = SpanPacket.from_record(row)
        presented = dict(
            re.findall(
                r"\[([AB]\d{3}) [AB]_ONLY\] (.*?)(?=\n\[|\n</semantic_diff>)",
                rendered,
                flags=re.DOTALL,
            )
        )
        for key, span in packet.spans.items():
            if span.kind != "SHARED":
                assert span.excerpt(span.kind[0]).text in presented[key]
        assert "UNSEEN_CONTINUATION" not in rendered
    assert sources[1]["truncated"]


def test_native_compiler_accepts_all_judge_and_critic_templates():
    import data_designer.config as dd
    from critics.coverage import coverage_column
    from critics.subject import subject_column, verifier_column
    from data_designer.engine.validation import validate_prompt_templates

    root = Path(__file__).parents[4] / "tutorials/eval/dedup"
    config = yaml.safe_load((root / "judge_config/fuzzy_pair_judge.yaml").read_text())
    judge = config["execution"]["stages"][0]["judges"][0]
    main = dd.LLMJudgeColumnConfig(
        name=judge["name"],
        model_alias="judge",
        prompt=(root / "judge_config/pair.jinja").read_text(),
        system_prompt=(root / "judge_config/system.jinja").read_text(),
        scores=[dd.Score(**score) for score in judge["scores"]],
    )
    prompts = root / "critics/prompts"
    violations = validate_prompt_templates(
        columns=[
            main,
            coverage_column("judge", prompts),
            subject_column("judge", prompts),
            verifier_column("judge", prompts),
        ],
        allowed_references=[
            "pair_id",
            "text_a",
            "text_b",
            "semantic_diff",
            "coverage_packet",
            "subject_packet",
            "subject_candidate",
        ],
    )
    assert not violations
