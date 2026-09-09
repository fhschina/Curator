# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.dedup.judging.client import _json_mode_system_prompt
from eval.dedup.judging.schema import (
    JUDGE_SCHEMA_V1,
    JUDGE_SCHEMA_V2,
    JUDGE_SCHEMA_V3,
    flatten_reason_codes,
    judge_output_schema,
    unresolved_judge_output,
    validate_judge_output,
)
from eval.dedup.validation import DedupEvaluationError


def valid_output() -> dict:
    return {
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
        "relation_type": "EXACT",
        "material_difference": "NONE",
        "fuzzy_scope": "IN_SCOPE",
        "confidence": 0.99,
        "reason_codes": [],
        "evidence": [],
    }


def test_valid_judge_output() -> None:
    assert validate_judge_output(valid_output())["relation_type"] == "EXACT"


def test_judge_output_rejects_extra_field() -> None:
    value = {**valid_output(), "chain_of_thought": "hidden"}
    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value)
    assert error.value.issue.code == "JUDGE_SCHEMA_INVALID"
    assert error.value.issue.details["missing_fields"] == []
    assert error.value.issue.details["extra_fields"] == ["chain_of_thought"]


def test_judge_output_reports_missing_field() -> None:
    value = valid_output()
    del value["evidence"]
    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value)
    assert error.value.issue.code == "JUDGE_SCHEMA_INVALID"
    assert error.value.issue.details["missing_fields"] == ["evidence"]
    assert error.value.issue.details["extra_fields"] == []


def test_judge_output_rejects_consistency_conflict() -> None:
    value = {**valid_output(), "relation_type": "UNRELATED"}
    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value)
    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_json_mode_prompt_embeds_frozen_schema() -> None:
    marker = "Use each required key exactly once:\n"

    prompt = _json_mode_system_prompt("base prompt")

    assert json.loads(prompt.split(marker, 1)[1]) == judge_output_schema()


def valid_v1_output() -> dict:
    return {
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
        "relation_type": "EXACT",
        "material_difference": "NONE",
        "fuzzy_scope": "IN_SCOPE",
        "confidence": 0.99,
        "reason_codes": {
            "material_differences": [],
            "overlap_sources": ["MAIN_CONTENT"],
            "primary_risk_factor": "NONE",
            "secondary_risk_factors": [],
            "evidence_quality": {"status": "SUFFICIENT", "issues": []},
        },
        "evidence": [],
    }


def test_v1_static_schema_matches_generated_contract() -> None:
    path = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "judge_output_schema_v1.json"

    assert json.loads(path.read_text()) == judge_output_schema(JUDGE_SCHEMA_V1)


def test_v1_accepts_translation_equivalence() -> None:
    value = valid_v1_output()
    value["relation_type"] = "TRANSLATION_EQUIVALENT"
    value["fuzzy_scope"] = "OUT_OF_SCOPE"
    value["reason_codes"]["overlap_sources"] = ["TRANSLATED_MAIN_CONTENT"]
    value["reason_codes"]["primary_risk_factor"] = "TRANSLATION_EQUIVALENCE"

    assert validate_judge_output(value, JUDGE_SCHEMA_V1)["same_duplicate_group"] == "YES"


def test_v1_rejects_duplicate_group_without_safe_replacement() -> None:
    value = valid_v1_output()
    value["a_can_replace_b"] = "NO"
    value["b_can_replace_a"] = "NO"

    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value, JUDGE_SCHEMA_V1)

    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_v1_unresolved_contract_is_internally_consistent() -> None:
    value = unresolved_judge_output(schema_version=JUDGE_SCHEMA_V1)

    assert validate_judge_output(value, JUDGE_SCHEMA_V1) == value
    assert value["reason_codes"]["evidence_quality"]["status"] == "INSUFFICIENT"


def test_v1_json_mode_prompt_embeds_selected_schema() -> None:
    marker = "Use each required key exactly once:\n"

    prompt = _json_mode_system_prompt("base prompt", JUDGE_SCHEMA_V1)

    assert json.loads(prompt.split(marker, 1)[1]) == judge_output_schema(JUDGE_SCHEMA_V1)


def test_v1_reason_codes_flatten_to_namespaced_dashboard_labels() -> None:
    reasons = valid_v1_output()["reason_codes"]
    reasons["material_differences"] = ["NUMBER_CHANGE"]
    reasons["primary_risk_factor"] = "IDENTIFIER_UNDERWEIGHTING"

    assert flatten_reason_codes(reasons) == [
        "MATERIAL_DELTA:NUMBER_CHANGE",
        "OVERLAP_SOURCE:MAIN_CONTENT",
        "PRIMARY_RISK:IDENTIFIER_UNDERWEIGHTING",
        "EVIDENCE_STATUS:SUFFICIENT",
    ]


def valid_v2_output() -> dict:
    return {
        "same_duplicate_group": "NO",
        "a_can_replace_b": "NO",
        "b_can_replace_a": "NO",
        "relation_type": "UNRELATED",
        "material_difference": "MAJOR",
        "primary_material_difference": "DOCUMENT_IDENTITY_CHANGE",
        "expected_minhash_action": "KEEP_SEPARATE",
        "surface_evidence_sufficiency": "SUFFICIENT",
        "expected_minhash_outcome": "EXPECTED_TRUE_NEGATIVE",
        "dominant_overlap_source": "NONE",
        "primary_risk_factor": "NONE",
        "evidence_quality": "SUFFICIENT",
        "confidence": 0.9,
        "reason_codes": ["MATERIAL_DELTA:DOCUMENT_IDENTITY_CHANGE", "EVIDENCE_STATUS:SUFFICIENT"],
        "evidence": [
            {"side": "A", "start_char": 0, "end_char": 5, "quote": "alpha"},
            {"side": "B", "start_char": 0, "end_char": 4, "quote": "beta"},
        ],
    }


def test_v2_distinguishes_correct_negative_from_false_positive_risk() -> None:
    correct_negative = valid_v2_output()
    assert validate_judge_output(correct_negative, JUDGE_SCHEMA_V2)["expected_minhash_outcome"] == (
        "EXPECTED_TRUE_NEGATIVE"
    )

    false_positive_risk = {
        **correct_negative,
        "expected_minhash_action": "GROUP",
        "surface_evidence_sufficiency": "INSUFFICIENT",
        "expected_minhash_outcome": "FALSE_POSITIVE_RISK",
    }
    assert validate_judge_output(false_positive_risk, JUDGE_SCHEMA_V2)["expected_minhash_outcome"] == (
        "FALSE_POSITIVE_RISK"
    )


def test_v2_rejects_free_form_minhash_diagnostics() -> None:
    value = {**valid_v2_output(), "surface_evidence_sufficiency": "OUT_OF_SCOPE"}

    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value, JUDGE_SCHEMA_V2)

    assert error.value.issue.code == "JUDGE_SCHEMA_INVALID"


def test_v2_requires_aligned_evidence_from_both_sides_for_non_exact_results() -> None:
    value = valid_v2_output()
    value["evidence"] = value["evidence"][:1]

    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value, JUDGE_SCHEMA_V2)

    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_v2_unresolved_contract_is_internally_consistent() -> None:
    value = unresolved_judge_output(schema_version=JUDGE_SCHEMA_V2)

    assert validate_judge_output(value, JUDGE_SCHEMA_V2) == value


def valid_v3_output() -> dict:
    return {
        "same_duplicate_group": "NO",
        "a_can_replace_b": "NO",
        "b_can_replace_a": "NO",
        "relation_type": "RELATED_NON_DUPLICATE",
        "material_difference": "MAJOR",
        "primary_material_difference": "DOCUMENT_IDENTITY_CHANGE",
        "dominant_overlap_source": "SHARED_PAGE_TEMPLATE",
        "primary_risk_factor": "TEMPLATE_SLOT_COLLISION",
        "confidence_tier": "HIGH",
        "reason_codes": [
            "MATERIAL_DELTA:DOCUMENT_IDENTITY_CHANGE",
            "OVERLAP_SOURCE:SHARED_PAGE_TEMPLATE",
            "PRIMARY_RISK:TEMPLATE_SLOT_COLLISION",
        ],
        "evidence": [
            {"side": "A", "start_char": 0, "end_char": 5, "quote": "alpha"},
            {"side": "B", "start_char": 0, "end_char": 4, "quote": "beta"},
        ],
    }


def test_v3_contract_omits_llm_minhash_and_numeric_confidence() -> None:
    schema = judge_output_schema(JUDGE_SCHEMA_V3)

    assert validate_judge_output(valid_v3_output(), JUDGE_SCHEMA_V3)["confidence_tier"] == "HIGH"
    assert "expected_minhash_action" not in schema["properties"]
    assert "expected_minhash_outcome" not in schema["properties"]
    assert "evidence_quality" not in schema["properties"]
    assert "confidence" not in schema["properties"]


def test_v3_containment_requires_one_direction_and_main_content_addition() -> None:
    value = valid_v3_output()
    value.update(
        {
            "same_duplicate_group": "YES",
            "a_can_replace_b": "YES",
            "b_can_replace_a": "NO",
            "relation_type": "CONTAINMENT",
            "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
            "dominant_overlap_source": "MAIN_CONTENT",
            "primary_risk_factor": "CONTAINMENT_ASYMMETRY",
            "confidence_tier": "MEDIUM",
        }
    )

    assert validate_judge_output(value, JUDGE_SCHEMA_V3)["same_duplicate_group"] == "YES"

    value["b_can_replace_a"] = "YES"
    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value, JUDGE_SCHEMA_V3)
    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_v3_translation_is_bidirectional_with_no_material_difference() -> None:
    value = valid_v3_output()
    value.update(
        {
            "same_duplicate_group": "YES",
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "NEAR_SURFACE",
            "material_difference": "NONE",
            "primary_material_difference": "NONE",
            "dominant_overlap_source": "MAIN_CONTENT",
            "primary_risk_factor": "TRANSLATION_EQUIVALENCE",
            "confidence_tier": "MEDIUM",
        }
    )

    assert validate_judge_output(value, JUDGE_SCHEMA_V3)["material_difference"] == "NONE"


def test_v3_non_exact_requires_exact_evidence_from_both_sides() -> None:
    value = valid_v3_output()
    value["evidence"] = value["evidence"][:1]

    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value, JUDGE_SCHEMA_V3)
    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_v3_unresolved_contract_is_low_confidence_and_consistent() -> None:
    value = unresolved_judge_output(schema_version=JUDGE_SCHEMA_V3)

    assert validate_judge_output(value, JUDGE_SCHEMA_V3) == value
    assert value["confidence_tier"] == "LOW"


def test_v3_allows_low_confidence_for_a_resolved_boundary() -> None:
    value = valid_v3_output()
    value["confidence_tier"] = "LOW"

    assert validate_judge_output(value, JUDGE_SCHEMA_V3)["confidence_tier"] == "LOW"


@pytest.mark.parametrize(
    ("relation", "material", "primary", "overlap", "risk", "same", "a_to_b", "b_to_a"),
    [
        (
            "NEAR_SURFACE",
            "MINOR",
            "OTHER_MATERIAL",
            "COOKIE_CONSENT",
            "BOILERPLATE_DOMINATED_SIMILARITY",
            "YES",
            "YES",
            "YES",
        ),
        (
            "NEAR_SURFACE",
            "MINOR",
            "OTHER_MATERIAL",
            "SITE_CHROME",
            "NONE",
            "YES",
            "YES",
            "YES",
        ),
        (
            "CONTAINMENT",
            "MAJOR",
            "MAIN_CONTENT_ADDITION_DELETION",
            "MAIN_CONTENT",
            "CONTAINMENT_ASYMMETRY",
            "YES",
            "YES",
            "NO",
        ),
        (
            "RELATED_NON_DUPLICATE",
            "MAJOR",
            "ENTITY_SLOT_CHANGE",
            "SHARED_PAGE_TEMPLATE",
            "TEMPLATE_SLOT_COLLISION",
            "NO",
            "NO",
            "NO",
        ),
        (
            "RELATED_NON_DUPLICATE",
            "MAJOR",
            "PAGE_ROLE_CHANGE",
            "MAIN_CONTENT",
            "PAGE_ROLE_COLLISION",
            "NO",
            "NO",
            "NO",
        ),
        (
            "VERSION_RELATED",
            "MAJOR",
            "PRODUCT_VERSION_CHANGE",
            "MAIN_CONTENT",
            "IDENTIFIER_UNDERWEIGHTING",
            "NO",
            "NO",
            "NO",
        ),
    ],
)
def test_v3_synthetic_boundary_regressions_are_contract_valid(  # noqa: PLR0913
    relation: str,
    material: str,
    primary: str,
    overlap: str,
    risk: str,
    same: str,
    a_to_b: str,
    b_to_a: str,
) -> None:
    value = valid_v3_output()
    value.update(
        {
            "same_duplicate_group": same,
            "a_can_replace_b": a_to_b,
            "b_can_replace_a": b_to_a,
            "relation_type": relation,
            "material_difference": material,
            "primary_material_difference": primary,
            "dominant_overlap_source": overlap,
            "primary_risk_factor": risk,
            "confidence_tier": "MEDIUM",
        }
    )

    assert validate_judge_output(value, JUDGE_SCHEMA_V3) == value
