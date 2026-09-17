# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Selected semantic Judge v0.7 output schema."""

from __future__ import annotations

from typing import Any, Final

from eval.dedup.core.contracts import DuplicateAnswer, MaterialDifference, RelationType
from eval.dedup.core.validation import require

JUDGE_SCHEMA: Final = "dedup-judge-output-v3"
CONFIDENCE_TIERS: Final = ("LOW", "MEDIUM", "HIGH")
DOMINANT_OVERLAP_SOURCES: Final = (
    "MAIN_CONTENT",
    "SHARED_PAGE_TEMPLATE",
    "SITE_CHROME",
    "COOKIE_CONSENT",
    "LEGAL_POLICY_TEMPLATE",
    "ERROR_AUTH_PAYWALL",
    "LOCAL_PASSAGE",
    "PARSER_ARTIFACT",
    "NONE",
    "UNRESOLVED",
)
PRIMARY_RISK_FACTORS: Final = (
    "NONE",
    "BOILERPLATE_DOMINATED_SIMILARITY",
    "TEMPLATE_SLOT_COLLISION",
    "IDENTIFIER_UNDERWEIGHTING",
    "TOPIC_ONLY_SIMILARITY",
    "LIST_SNAPSHOT_COLLISION",
    "PAGE_ROLE_COLLISION",
    "LEGAL_CONTEXT_COLLISION",
    "LONG_DOCUMENT_LOCAL_OVERLAP",
    "TRANSLATION_EQUIVALENCE",
    "PARAPHRASE_EQUIVALENCE",
    "CONTAINMENT_ASYMMETRY",
    "EXTRACTION_OR_PAYLOAD_LIMIT",
    "PARSER_ARTIFACT_DOMINANCE",
    "OTHER",
)
PRIMARY_MATERIAL_DIFFERENCES: Final = (
    "NONE",
    "MAIN_CONTENT_ADDITION_DELETION",
    "ENTITY_SLOT_CHANGE",
    "DOCUMENT_IDENTITY_CHANGE",
    "NUMBER_CHANGE",
    "DATE_TIME_CHANGE",
    "PRODUCT_VERSION_CHANGE",
    "RESULT_SET_CHANGE",
    "PAGE_ROLE_CHANGE",
    "LEGAL_CONTEXT_CHANGE",
    "NEGATION_CHANGE",
    "CODE_LITERAL_CHANGE",
    "CODE_OUTPUT_CHANGE",
    "OTHER_MATERIAL",
    "UNRESOLVED",
)
JUDGE_FIELDS_V3: Final = {
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
    "primary_material_difference",
    "dominant_overlap_source",
    "primary_risk_factor",
    "confidence_tier",
    "reason_codes",
    "evidence",
}


def _evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["side", "start_char", "end_char", "quote"],
        "properties": {
            "side": {"type": "string", "enum": ["A", "B"]},
            "start_char": {"type": "integer", "minimum": 0},
            "end_char": {"type": "integer", "minimum": 0},
            "quote": {"type": "string", "minLength": 1, "maxLength": 240},
        },
    }


def judge_output_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(JUDGE_FIELDS_V3),
        "properties": {
            "same_duplicate_group": {"type": "string", "enum": list(DuplicateAnswer)},
            "a_can_replace_b": {"type": "string", "enum": list(DuplicateAnswer)},
            "b_can_replace_a": {"type": "string", "enum": list(DuplicateAnswer)},
            "relation_type": {"type": "string", "enum": list(RelationType)},
            "material_difference": {"type": "string", "enum": list(MaterialDifference)},
            "primary_material_difference": {
                "type": "string",
                "enum": list(PRIMARY_MATERIAL_DIFFERENCES),
            },
            "dominant_overlap_source": {"type": "string", "enum": list(DOMINANT_OVERLAP_SOURCES)},
            "primary_risk_factor": {"type": "string", "enum": list(PRIMARY_RISK_FACTORS)},
            "confidence_tier": {"type": "string", "enum": list(CONFIDENCE_TIERS)},
            "reason_codes": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string"},
            },
            "evidence": {"type": "array", "maxItems": 4, "items": _evidence_schema()},
        },
    }


def _is_enum_value(value: Any, enum_type: type) -> bool:
    return isinstance(value, str) and any(value == member.value for member in enum_type)


def _validate_evidence(value: Any) -> None:
    require(
        isinstance(value, list) and len(value) <= 4,
        "JUDGE_SCHEMA_INVALID",
        "evidence must contain at most four items",
    )
    for index, item in enumerate(value):
        require(isinstance(item, dict), "JUDGE_SCHEMA_INVALID", "evidence item must be an object", index=index)
        require(
            set(item) == {"side", "start_char", "end_char", "quote"},
            "JUDGE_SCHEMA_INVALID",
            "evidence fields differ",
            index=index,
        )
        require(item["side"] in {"A", "B"}, "JUDGE_SCHEMA_INVALID", "evidence side is invalid", index=index)
        require(
            isinstance(item["start_char"], int)
            and not isinstance(item["start_char"], bool)
            and isinstance(item["end_char"], int)
            and not isinstance(item["end_char"], bool)
            and 0 <= item["start_char"] <= item["end_char"],
            "JUDGE_SCHEMA_INVALID",
            "evidence offsets are invalid",
            index=index,
        )
        require(
            isinstance(item["quote"], str) and 0 < len(item["quote"]) <= 240,
            "JUDGE_SCHEMA_INVALID",
            "evidence quote is invalid",
            index=index,
        )


def _validate_consistency(value: dict[str, Any]) -> None:
    same = value["same_duplicate_group"]
    a_to_b = value["a_can_replace_b"]
    b_to_a = value["b_can_replace_a"]
    relation = value["relation_type"]
    material = value["material_difference"]
    primary_difference = value["primary_material_difference"]

    if same == DuplicateAnswer.UNRESOLVED:
        require(
            a_to_b == b_to_a == DuplicateAnswer.UNRESOLVED
            and relation == RelationType.UNRESOLVED
            and material == MaterialDifference.UNRESOLVED
            and primary_difference == "UNRESOLVED"
            and value["dominant_overlap_source"] == "UNRESOLVED",
            "JUDGE_CONSISTENCY_INVALID",
            "an unresolved group decision requires every semantic field to be UNRESOLVED",
        )
        require(
            value["primary_risk_factor"] == "EXTRACTION_OR_PAYLOAD_LIMIT"
            and value["confidence_tier"] == "LOW"
            and not value["evidence"],
            "JUDGE_CONSISTENCY_INVALID",
            "unresolved decisions require the extraction risk, LOW confidence, and no evidence",
        )
        return

    require(
        DuplicateAnswer.UNRESOLVED not in {a_to_b, b_to_a}
        and relation != RelationType.UNRESOLVED
        and material != MaterialDifference.UNRESOLVED
        and primary_difference != "UNRESOLVED"
        and value["dominant_overlap_source"] != "UNRESOLVED"
        and value["confidence_tier"] in CONFIDENCE_TIERS,
        "JUDGE_CONSISTENCY_INVALID",
        "a resolved decision cannot mix unresolved fields",
    )
    if same == DuplicateAnswer.YES:
        require(
            DuplicateAnswer.YES in {a_to_b, b_to_a},
            "JUDGE_CONSISTENCY_INVALID",
            "same_duplicate_group=YES requires at least one safe replacement direction",
        )
    else:
        require(
            a_to_b == b_to_a == DuplicateAnswer.NO,
            "JUDGE_CONSISTENCY_INVALID",
            "same_duplicate_group=NO requires both replacement directions to be NO",
        )

    if relation in {RelationType.EXACT, RelationType.CANONICAL_EXACT}:
        require(
            same == a_to_b == b_to_a == DuplicateAnswer.YES and material == MaterialDifference.NONE,
            "JUDGE_CONSISTENCY_INVALID",
            "exact relations require bidirectional replacement and no material difference",
        )
    if relation == RelationType.NEAR_SURFACE:
        require(
            same == a_to_b == b_to_a == DuplicateAnswer.YES
            and material in {MaterialDifference.NONE, MaterialDifference.MINOR},
            "JUDGE_CONSISTENCY_INVALID",
            "near-surface relations require bidirectional replacement without a major difference",
        )
    if relation == RelationType.CONTAINMENT:
        require(
            same == DuplicateAnswer.YES
            and {a_to_b, b_to_a} == {DuplicateAnswer.YES, DuplicateAnswer.NO}
            and material == MaterialDifference.MAJOR
            and primary_difference == "MAIN_CONTENT_ADDITION_DELETION",
            "JUDGE_CONSISTENCY_INVALID",
            "containment requires one safe direction and a major main-content addition",
        )
    if relation in {RelationType.VERSION_RELATED, RelationType.RELATED_NON_DUPLICATE, RelationType.UNRELATED}:
        require(
            same == DuplicateAnswer.NO
            and a_to_b == b_to_a == DuplicateAnswer.NO
            and material == MaterialDifference.MAJOR,
            "JUDGE_CONSISTENCY_INVALID",
            "non-interchangeable relations require no replacement and a major difference",
        )
    require(
        (material == MaterialDifference.NONE) == (primary_difference == "NONE"),
        "JUDGE_CONSISTENCY_INVALID",
        "material severity NONE must agree with the primary material-difference diagnosis",
    )
    if relation == RelationType.EXACT:
        require(not value["evidence"], "JUDGE_CONSISTENCY_INVALID", "exact decisions do not retain quote evidence")
    else:
        require(
            {item["side"] for item in value["evidence"]} == {"A", "B"},
            "JUDGE_CONSISTENCY_INVALID",
            "resolved non-exact decisions require aligned quote evidence from both documents",
        )


def validate_judge_output(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "JUDGE_SCHEMA_INVALID", "judge output must be a JSON object")
    actual_fields = set(value)
    require(
        actual_fields == JUDGE_FIELDS_V3,
        "JUDGE_SCHEMA_INVALID",
        "judge output fields differ",
        expected_fields=sorted(JUDGE_FIELDS_V3),
        actual_fields=sorted(actual_fields),
        missing_fields=sorted(JUDGE_FIELDS_V3 - actual_fields),
        extra_fields=sorted(actual_fields - JUDGE_FIELDS_V3),
    )
    for field in ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a"):
        require(_is_enum_value(value[field], DuplicateAnswer), "JUDGE_SCHEMA_INVALID", "invalid answer", field=field)
    require(_is_enum_value(value["relation_type"], RelationType), "JUDGE_SCHEMA_INVALID", "invalid relation_type")
    require(
        _is_enum_value(value["material_difference"], MaterialDifference),
        "JUDGE_SCHEMA_INVALID",
        "invalid material_difference",
    )
    enum_fields = {
        "primary_material_difference": PRIMARY_MATERIAL_DIFFERENCES,
        "dominant_overlap_source": DOMINANT_OVERLAP_SOURCES,
        "primary_risk_factor": PRIMARY_RISK_FACTORS,
        "confidence_tier": CONFIDENCE_TIERS,
    }
    for field, allowed in enum_fields.items():
        require(
            isinstance(value[field], str) and value[field] in allowed,
            "JUDGE_SCHEMA_INVALID",
            f"invalid {field}",
            field=field,
        )
    reasons = value["reason_codes"]
    require(
        isinstance(reasons, list)
        and all(isinstance(reason, str) for reason in reasons)
        and len(reasons) == len(set(reasons)),
        "JUDGE_SCHEMA_INVALID",
        "reason_codes must be a unique string list",
    )
    _validate_evidence(value["evidence"])
    _validate_consistency(value)
    return value


def unresolved_judge_output() -> dict[str, Any]:
    return {
        "same_duplicate_group": "UNRESOLVED",
        "a_can_replace_b": "UNRESOLVED",
        "b_can_replace_a": "UNRESOLVED",
        "relation_type": "UNRESOLVED",
        "material_difference": "UNRESOLVED",
        "primary_material_difference": "UNRESOLVED",
        "dominant_overlap_source": "UNRESOLVED",
        "primary_risk_factor": "EXTRACTION_OR_PAYLOAD_LIMIT",
        "confidence_tier": "LOW",
        "reason_codes": ["INSUFFICIENT_EVIDENCE"],
        "evidence": [],
    }


def flatten_reason_codes(value: Any) -> list[str]:
    """Normalize reason-code data for reports and Pair Explorer filters."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []
