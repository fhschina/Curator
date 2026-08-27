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

"""V1 judge output schema and strict local validation."""

from __future__ import annotations

from typing import Any, Final

from eval.dedup.contracts import DuplicateAnswer, FuzzyScope, MaterialDifference
from eval.dedup.validation import require

JUDGE_SCHEMA_V1: Final = "dedup-judge-output-v1"
JUDGE_FIELDS_V1: Final = {
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
    "fuzzy_scope",
    "confidence",
    "reason_codes",
    "evidence",
}


def _is_enum_value(value: Any, enum_type: type) -> bool:
    """Return whether *value* is a string value of an enum on Python 3.11+."""

    return isinstance(value, str) and any(value == member.value for member in enum_type)


V1_RELATION_TYPES: Final = (
    "EXACT",
    "CANONICAL_EXACT",
    "TRANSLATION_EQUIVALENT",
    "NEAR_SURFACE",
    "SEMANTIC_PARAPHRASE",
    "CONTAINMENT",
    "VERSION_RELATED",
    "TEMPLATE_SIBLINGS",
    "RELATED_NON_DUPLICATE",
    "UNRELATED",
    "UNRESOLVED",
)
V1_MATERIAL_DIFFERENCE_CODES: Final = frozenset(
    {
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
    }
)
V1_OVERLAP_SOURCE_CODES: Final = frozenset(
    {
        "MAIN_CONTENT",
        "TRANSLATED_MAIN_CONTENT",
        "SHARED_PAGE_TEMPLATE",
        "SITE_CHROME",
        "COOKIE_CONSENT",
        "LEGAL_POLICY_TEMPLATE",
        "ERROR_AUTH_PAYWALL",
        "LOCAL_PASSAGE",
        "PARSER_ARTIFACT",
    }
)
V1_PRIMARY_RISK_FACTORS: Final = frozenset(
    {
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
    }
)
V1_SECONDARY_RISK_FACTORS: Final = V1_PRIMARY_RISK_FACTORS - {"NONE"}
V1_EVIDENCE_STATUSES: Final = frozenset({"SUFFICIENT", "INSUFFICIENT"})
V1_EVIDENCE_ISSUES: Final = frozenset(
    {
        "MAIN_CONTENT_MISSING",
        "TRUNCATED_PAYLOAD",
        "PARSER_NOISE",
        "INSUFFICIENT_VISIBLE_EVIDENCE",
    }
)


def _evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["side", "start_char", "end_char", "quote"],
        "properties": {
            "side": {"type": "string", "enum": ["A", "B"]},
            "start_char": {"type": "integer", "minimum": 0},
            "end_char": {"type": "integer", "minimum": 0},
            "quote": {"type": "string", "maxLength": 240},
        },
    }


def _reason_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "material_differences",
            "overlap_sources",
            "primary_risk_factor",
            "secondary_risk_factors",
            "evidence_quality",
        ],
        "properties": {
            "material_differences": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(V1_MATERIAL_DIFFERENCE_CODES)},
            },
            "overlap_sources": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(V1_OVERLAP_SOURCE_CODES)},
            },
            "primary_risk_factor": {"type": "string", "enum": sorted(V1_PRIMARY_RISK_FACTORS)},
            "secondary_risk_factors": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(V1_SECONDARY_RISK_FACTORS)},
            },
            "evidence_quality": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status", "issues"],
                "properties": {
                    "status": {"type": "string", "enum": sorted(V1_EVIDENCE_STATUSES)},
                    "issues": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": sorted(V1_EVIDENCE_ISSUES)},
                    },
                },
            },
        },
    }


def judge_output_schema_v1() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(JUDGE_FIELDS_V1),
        "properties": {
            "same_duplicate_group": {"type": "string", "enum": list(DuplicateAnswer)},
            "a_can_replace_b": {"type": "string", "enum": list(DuplicateAnswer)},
            "b_can_replace_a": {"type": "string", "enum": list(DuplicateAnswer)},
            "relation_type": {"type": "string", "enum": list(V1_RELATION_TYPES)},
            "material_difference": {"type": "string", "enum": list(MaterialDifference)},
            "fuzzy_scope": {"type": "string", "enum": list(FuzzyScope)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason_codes": _reason_schema(),
            "evidence": {"type": "array", "maxItems": 4, "items": _evidence_schema()},
        },
    }


def _validate_string_array(value: Any, *, field: str, allowed: frozenset[str]) -> list[str]:
    require(isinstance(value, list), "JUDGE_SCHEMA_INVALID", f"{field} must be a list", field=field)
    require(
        all(isinstance(item, str) for item in value),
        "JUDGE_SCHEMA_INVALID",
        f"{field} items must be strings",
        field=field,
    )
    require(len(value) == len(set(value)), "JUDGE_SCHEMA_INVALID", f"{field} items must be unique", field=field)
    require(
        all(item in allowed for item in value),
        "JUDGE_SCHEMA_INVALID",
        f"{field} contains an unknown code",
        field=field,
    )
    return value


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
            isinstance(item["quote"], str) and len(item["quote"]) <= 240,
            "JUDGE_SCHEMA_INVALID",
            "evidence quote is invalid",
            index=index,
        )


def _validate_reasons(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "JUDGE_SCHEMA_INVALID", "reason_codes must be an object", field="reason_codes")
    expected = {
        "material_differences",
        "overlap_sources",
        "primary_risk_factor",
        "secondary_risk_factors",
        "evidence_quality",
    }
    require(set(value) == expected, "JUDGE_SCHEMA_INVALID", "reason_codes fields differ", field="reason_codes")
    _validate_string_array(
        value["material_differences"],
        field="reason_codes.material_differences",
        allowed=V1_MATERIAL_DIFFERENCE_CODES,
    )
    _validate_string_array(
        value["overlap_sources"],
        field="reason_codes.overlap_sources",
        allowed=V1_OVERLAP_SOURCE_CODES,
    )
    primary = value["primary_risk_factor"]
    require(
        isinstance(primary, str) and primary in V1_PRIMARY_RISK_FACTORS,
        "JUDGE_SCHEMA_INVALID",
        "primary_risk_factor is invalid",
        field="reason_codes.primary_risk_factor",
    )
    secondary = _validate_string_array(
        value["secondary_risk_factors"],
        field="reason_codes.secondary_risk_factors",
        allowed=V1_SECONDARY_RISK_FACTORS,
    )
    require(
        primary not in secondary,
        "JUDGE_CONSISTENCY_INVALID",
        "primary risk factor cannot also be secondary",
    )
    require(
        primary != "NONE" or not secondary,
        "JUDGE_CONSISTENCY_INVALID",
        "NONE primary risk factor cannot have secondary risks",
    )
    quality = value["evidence_quality"]
    require(
        isinstance(quality, dict) and set(quality) == {"status", "issues"},
        "JUDGE_SCHEMA_INVALID",
        "evidence_quality fields differ",
        field="reason_codes.evidence_quality",
    )
    require(
        isinstance(quality["status"], str) and quality["status"] in V1_EVIDENCE_STATUSES,
        "JUDGE_SCHEMA_INVALID",
        "evidence quality status is invalid",
        field="reason_codes.evidence_quality.status",
    )
    issues = _validate_string_array(
        quality["issues"],
        field="reason_codes.evidence_quality.issues",
        allowed=V1_EVIDENCE_ISSUES,
    )
    require(
        quality["status"] != "INSUFFICIENT" or bool(issues),
        "JUDGE_CONSISTENCY_INVALID",
        "insufficient evidence quality requires at least one issue",
    )
    require(
        quality["status"] != "SUFFICIENT" or "INSUFFICIENT_VISIBLE_EVIDENCE" not in issues,
        "JUDGE_CONSISTENCY_INVALID",
        "sufficient evidence cannot include INSUFFICIENT_VISIBLE_EVIDENCE",
    )
    return value


def _validate_consistency(value: dict[str, Any]) -> None:
    same = value["same_duplicate_group"]
    a_to_b = value["a_can_replace_b"]
    b_to_a = value["b_can_replace_a"]
    relation = value["relation_type"]
    material = value["material_difference"]
    scope = value["fuzzy_scope"]
    reasons = value["reason_codes"]
    deltas = reasons["material_differences"]
    overlaps = reasons["overlap_sources"]
    risks = {reasons["primary_risk_factor"], *reasons["secondary_risk_factors"]}
    quality = reasons["evidence_quality"]

    if same == DuplicateAnswer.UNRESOLVED:
        require(
            a_to_b == b_to_a == DuplicateAnswer.UNRESOLVED
            and relation == "UNRESOLVED"
            and material == MaterialDifference.UNRESOLVED
            and scope == FuzzyScope.UNRESOLVED,
            "JUDGE_CONSISTENCY_INVALID",
            "an unresolved group decision requires every core decision to be UNRESOLVED",
        )
        require(
            quality["status"] == "INSUFFICIENT" and value["confidence"] <= 0.5,
            "JUDGE_CONSISTENCY_INVALID",
            "unresolved decisions require insufficient evidence and confidence at most 0.5",
        )
        require(
            reasons["primary_risk_factor"] in {"EXTRACTION_OR_PAYLOAD_LIMIT", "PARSER_ARTIFACT_DOMINANCE"},
            "JUDGE_CONSISTENCY_INVALID",
            "unresolved decisions require an evidence-related primary risk factor",
        )
        return

    require(
        DuplicateAnswer.UNRESOLVED not in {a_to_b, b_to_a}
        and relation != "UNRESOLVED"
        and material != MaterialDifference.UNRESOLVED
        and scope != FuzzyScope.UNRESOLVED,
        "JUDGE_CONSISTENCY_INVALID",
        "a resolved group decision cannot mix in UNRESOLVED core fields",
    )
    require(
        quality["status"] == "SUFFICIENT",
        "JUDGE_CONSISTENCY_INVALID",
        "resolved decisions require sufficient visible evidence",
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

    equivalent_relations = {
        "EXACT",
        "CANONICAL_EXACT",
        "TRANSLATION_EQUIVALENT",
        "NEAR_SURFACE",
        "SEMANTIC_PARAPHRASE",
    }
    if relation in equivalent_relations:
        require(
            same == a_to_b == b_to_a == DuplicateAnswer.YES,
            "JUDGE_CONSISTENCY_INVALID",
            "equivalent relations require duplicate grouping and bidirectional replacement",
        )
        require(
            material in {MaterialDifference.NONE, MaterialDifference.MINOR},
            "JUDGE_CONSISTENCY_INVALID",
            "equivalent relations cannot have a major material difference",
        )
    if relation in {"EXACT", "CANONICAL_EXACT"}:
        require(
            material == MaterialDifference.NONE and not deltas,
            "JUDGE_CONSISTENCY_INVALID",
            "exact relations require no material differences",
        )
    if relation == "TRANSLATION_EQUIVALENT":
        require(
            "TRANSLATED_MAIN_CONTENT" in overlaps and "TRANSLATION_EQUIVALENCE" in risks,
            "JUDGE_CONSISTENCY_INVALID",
            "translation equivalence requires translated-content overlap and risk labels",
        )
    if relation == "CONTAINMENT":
        require(
            same == DuplicateAnswer.YES and {a_to_b, b_to_a} == {DuplicateAnswer.YES, DuplicateAnswer.NO},
            "JUDGE_CONSISTENCY_INVALID",
            "containment requires exactly one safe replacement direction",
        )
    if relation in {"TEMPLATE_SIBLINGS", "RELATED_NON_DUPLICATE", "UNRELATED"}:
        require(
            same == DuplicateAnswer.NO
            and a_to_b == b_to_a == DuplicateAnswer.NO
            and material == MaterialDifference.MAJOR,
            "JUDGE_CONSISTENCY_INVALID",
            "non-duplicate relations require distinct grouping, no replacement, and a major difference",
        )
    if relation == "TEMPLATE_SIBLINGS":
        require(
            "SHARED_PAGE_TEMPLATE" in overlaps and "TEMPLATE_SLOT_COLLISION" in risks,
            "JUDGE_CONSISTENCY_INVALID",
            "template siblings require template overlap and slot-collision risk labels",
        )

    if material == MaterialDifference.NONE:
        require(not deltas, "JUDGE_CONSISTENCY_INVALID", "NONE material difference requires no delta codes")
    else:
        require(bool(deltas), "JUDGE_CONSISTENCY_INVALID", "MINOR or MAJOR difference requires a delta code")
    require(
        not (material == MaterialDifference.MAJOR and a_to_b == DuplicateAnswer.YES and b_to_a == DuplicateAnswer.YES),
        "JUDGE_CONSISTENCY_INVALID",
        "MAJOR difference cannot be bidirectionally replaceable",
    )


def validate_judge_output_v1(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "JUDGE_SCHEMA_INVALID", "judge output must be a JSON object")
    actual_fields = set(value)
    require(
        actual_fields == JUDGE_FIELDS_V1,
        "JUDGE_SCHEMA_INVALID",
        "judge output fields differ",
        expected_fields=sorted(JUDGE_FIELDS_V1),
        actual_fields=sorted(actual_fields),
        missing_fields=sorted(JUDGE_FIELDS_V1 - actual_fields),
        extra_fields=sorted(actual_fields - JUDGE_FIELDS_V1),
    )
    for field in ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a"):
        require(
            _is_enum_value(value[field], DuplicateAnswer),
            "JUDGE_SCHEMA_INVALID",
            "invalid ternary answer",
            field=field,
        )
    require(
        isinstance(value["relation_type"], str) and value["relation_type"] in V1_RELATION_TYPES,
        "JUDGE_SCHEMA_INVALID",
        "invalid relation_type",
    )
    require(
        _is_enum_value(value["material_difference"], MaterialDifference),
        "JUDGE_SCHEMA_INVALID",
        "invalid material_difference",
    )
    require(
        _is_enum_value(value["fuzzy_scope"], FuzzyScope),
        "JUDGE_SCHEMA_INVALID",
        "invalid fuzzy_scope",
    )
    confidence = value["confidence"]
    require(
        isinstance(confidence, int | float) and not isinstance(confidence, bool),
        "JUDGE_SCHEMA_INVALID",
        "confidence must be numeric",
    )
    require(0.0 <= confidence <= 1.0, "JUDGE_SCHEMA_INVALID", "confidence is outside [0,1]")
    _validate_reasons(value["reason_codes"])
    _validate_evidence(value["evidence"])
    _validate_consistency(value)
    return value


def unresolved_judge_output_v1() -> dict[str, Any]:
    return {
        "same_duplicate_group": DuplicateAnswer.UNRESOLVED,
        "a_can_replace_b": DuplicateAnswer.UNRESOLVED,
        "b_can_replace_a": DuplicateAnswer.UNRESOLVED,
        "relation_type": "UNRESOLVED",
        "material_difference": MaterialDifference.UNRESOLVED,
        "fuzzy_scope": FuzzyScope.UNRESOLVED,
        "confidence": 0.0,
        "reason_codes": {
            "material_differences": [],
            "overlap_sources": [],
            "primary_risk_factor": "EXTRACTION_OR_PAYLOAD_LIMIT",
            "secondary_risk_factors": [],
            "evidence_quality": {
                "status": "INSUFFICIENT",
                "issues": ["INSUFFICIENT_VISIBLE_EVIDENCE"],
            },
        },
        "evidence": [],
    }
