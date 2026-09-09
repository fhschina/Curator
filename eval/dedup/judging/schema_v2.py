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

"""V2 judge output schema with unambiguous MinHash diagnostics."""

from __future__ import annotations

from typing import Any, Final

from eval.dedup.contracts import DuplicateAnswer, MaterialDifference, RelationType
from eval.dedup.validation import require

JUDGE_SCHEMA_V2: Final = "dedup-judge-output-v2"
JUDGE_FIELDS_V2: Final = {
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
    "primary_material_difference",
    "expected_minhash_action",
    "surface_evidence_sufficiency",
    "expected_minhash_outcome",
    "dominant_overlap_source",
    "primary_risk_factor",
    "evidence_quality",
    "confidence",
    "reason_codes",
    "evidence",
}

EXPECTED_MINHASH_ACTIONS: Final = ("GROUP", "KEEP_SEPARATE", "UNCERTAIN", "UNRESOLVED")
SURFACE_EVIDENCE_SUFFICIENCY: Final = ("SUFFICIENT", "BORDERLINE", "INSUFFICIENT", "UNRESOLVED")
EXPECTED_MINHASH_OUTCOMES: Final = (
    "EXPECTED_TRUE_POSITIVE",
    "EXPECTED_TRUE_NEGATIVE",
    "FALSE_POSITIVE_RISK",
    "FALSE_NEGATIVE_RISK",
    "AMBIGUOUS",
    "UNRESOLVED",
)
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
EVIDENCE_QUALITY_VALUES: Final = ("SUFFICIENT", "INSUFFICIENT")


def derive_minhash_diagnostics(same_duplicate_group: str, expected_action: str) -> tuple[str, str]:
    """Derive surface sufficiency and outcome from semantic and algorithmic decisions."""

    if same_duplicate_group == DuplicateAnswer.UNRESOLVED or expected_action == "UNRESOLVED":
        return "UNRESOLVED", "UNRESOLVED"
    if expected_action == "UNCERTAIN":
        return "BORDERLINE", "AMBIGUOUS"
    mapping = {
        (DuplicateAnswer.YES, "GROUP"): ("SUFFICIENT", "EXPECTED_TRUE_POSITIVE"),
        (DuplicateAnswer.NO, "KEEP_SEPARATE"): ("SUFFICIENT", "EXPECTED_TRUE_NEGATIVE"),
        (DuplicateAnswer.NO, "GROUP"): ("INSUFFICIENT", "FALSE_POSITIVE_RISK"),
        (DuplicateAnswer.YES, "KEEP_SEPARATE"): ("INSUFFICIENT", "FALSE_NEGATIVE_RISK"),
    }
    require(
        (same_duplicate_group, expected_action) in mapping,
        "JUDGE_CONSISTENCY_INVALID",
        "MinHash diagnostics cannot be derived from the selected decisions",
    )
    return mapping[(same_duplicate_group, expected_action)]


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


def judge_output_schema_v2() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(JUDGE_FIELDS_V2),
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
            "expected_minhash_action": {"type": "string", "enum": list(EXPECTED_MINHASH_ACTIONS)},
            "surface_evidence_sufficiency": {
                "type": "string",
                "enum": list(SURFACE_EVIDENCE_SUFFICIENCY),
            },
            "expected_minhash_outcome": {"type": "string", "enum": list(EXPECTED_MINHASH_OUTCOMES)},
            "dominant_overlap_source": {"type": "string", "enum": list(DOMINANT_OVERLAP_SOURCES)},
            "primary_risk_factor": {"type": "string", "enum": list(PRIMARY_RISK_FACTORS)},
            "evidence_quality": {"type": "string", "enum": list(EVIDENCE_QUALITY_VALUES)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
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
    action = value["expected_minhash_action"]

    if same == DuplicateAnswer.UNRESOLVED:
        require(
            a_to_b == b_to_a == DuplicateAnswer.UNRESOLVED
            and relation == RelationType.UNRESOLVED
            and material == MaterialDifference.UNRESOLVED
            and value["primary_material_difference"] == "UNRESOLVED"
            and action == "UNRESOLVED"
            and value["surface_evidence_sufficiency"] == "UNRESOLVED"
            and value["expected_minhash_outcome"] == "UNRESOLVED",
            "JUDGE_CONSISTENCY_INVALID",
            "an unresolved group decision requires every decision field to be UNRESOLVED",
        )
        require(
            value["evidence_quality"] == "INSUFFICIENT" and value["confidence"] <= 0.5,
            "JUDGE_CONSISTENCY_INVALID",
            "unresolved decisions require insufficient evidence and confidence at most 0.5",
        )
        require(not value["evidence"], "JUDGE_CONSISTENCY_INVALID", "unresolved decisions cannot retain evidence")
        return

    require(
        DuplicateAnswer.UNRESOLVED not in {a_to_b, b_to_a}
        and relation != RelationType.UNRESOLVED
        and material != MaterialDifference.UNRESOLVED
        and value["primary_material_difference"] != "UNRESOLVED"
        and action != "UNRESOLVED",
        "JUDGE_CONSISTENCY_INVALID",
        "a resolved group decision cannot mix in UNRESOLVED fields",
    )
    require(
        value["evidence_quality"] == "SUFFICIENT",
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

    if relation in {RelationType.EXACT, RelationType.CANONICAL_EXACT, RelationType.NEAR_SURFACE}:
        require(
            same == a_to_b == b_to_a == DuplicateAnswer.YES and material != MaterialDifference.MAJOR,
            "JUDGE_CONSISTENCY_INVALID",
            "equivalent relations require bidirectional replacement without a major difference",
        )
    if relation == RelationType.CONTAINMENT:
        require(
            same == DuplicateAnswer.YES and {a_to_b, b_to_a} == {DuplicateAnswer.YES, DuplicateAnswer.NO},
            "JUDGE_CONSISTENCY_INVALID",
            "containment requires exactly one safe replacement direction",
        )
    if relation in {RelationType.RELATED_NON_DUPLICATE, RelationType.UNRELATED}:
        require(
            same == DuplicateAnswer.NO
            and a_to_b == b_to_a == DuplicateAnswer.NO
            and material == MaterialDifference.MAJOR,
            "JUDGE_CONSISTENCY_INVALID",
            "non-duplicate relations require no replacement and a major difference",
        )
    require(
        not (material == MaterialDifference.MAJOR and a_to_b == b_to_a == DuplicateAnswer.YES),
        "JUDGE_CONSISTENCY_INVALID",
        "MAJOR difference cannot be bidirectionally replaceable",
    )
    require(
        (material == MaterialDifference.NONE) == (value["primary_material_difference"] == "NONE"),
        "JUDGE_CONSISTENCY_INVALID",
        "material severity NONE must agree with the primary material-difference diagnosis",
    )
    if relation == RelationType.EXACT:
        require(
            action == "GROUP",
            "JUDGE_CONSISTENCY_INVALID",
            "exact documents must have expected_minhash_action=GROUP",
        )
    expected_sufficiency, expected_outcome = derive_minhash_diagnostics(same, action)
    require(
        value["surface_evidence_sufficiency"] == expected_sufficiency
        and value["expected_minhash_outcome"] == expected_outcome,
        "JUDGE_CONSISTENCY_INVALID",
        "derived MinHash diagnostics disagree with the semantic and expected-action decisions",
    )
    if relation != RelationType.EXACT:
        require(
            {item["side"] for item in value["evidence"]} == {"A", "B"},
            "JUDGE_CONSISTENCY_INVALID",
            "resolved non-exact decisions require aligned quote evidence from both documents",
        )


def validate_judge_output_v2(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "JUDGE_SCHEMA_INVALID", "judge output must be a JSON object")
    actual_fields = set(value)
    require(
        actual_fields == JUDGE_FIELDS_V2,
        "JUDGE_SCHEMA_INVALID",
        "judge output fields differ",
        expected_fields=sorted(JUDGE_FIELDS_V2),
        actual_fields=sorted(actual_fields),
        missing_fields=sorted(JUDGE_FIELDS_V2 - actual_fields),
        extra_fields=sorted(actual_fields - JUDGE_FIELDS_V2),
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
        "expected_minhash_action": EXPECTED_MINHASH_ACTIONS,
        "surface_evidence_sufficiency": SURFACE_EVIDENCE_SUFFICIENCY,
        "expected_minhash_outcome": EXPECTED_MINHASH_OUTCOMES,
        "dominant_overlap_source": DOMINANT_OVERLAP_SOURCES,
        "primary_risk_factor": PRIMARY_RISK_FACTORS,
        "primary_material_difference": PRIMARY_MATERIAL_DIFFERENCES,
        "evidence_quality": EVIDENCE_QUALITY_VALUES,
    }
    for field, allowed in enum_fields.items():
        require(
            isinstance(value[field], str) and value[field] in allowed,
            "JUDGE_SCHEMA_INVALID",
            f"invalid {field}",
            field=field,
        )
    confidence = value["confidence"]
    require(
        isinstance(confidence, int | float) and not isinstance(confidence, bool) and 0.0 <= confidence <= 1.0,
        "JUDGE_SCHEMA_INVALID",
        "confidence must be numeric and inside [0,1]",
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


def unresolved_judge_output_v2() -> dict[str, Any]:
    return {
        "same_duplicate_group": "UNRESOLVED",
        "a_can_replace_b": "UNRESOLVED",
        "b_can_replace_a": "UNRESOLVED",
        "relation_type": "UNRESOLVED",
        "material_difference": "UNRESOLVED",
        "primary_material_difference": "UNRESOLVED",
        "expected_minhash_action": "UNRESOLVED",
        "surface_evidence_sufficiency": "UNRESOLVED",
        "expected_minhash_outcome": "UNRESOLVED",
        "dominant_overlap_source": "UNRESOLVED",
        "primary_risk_factor": "EXTRACTION_OR_PAYLOAD_LIMIT",
        "evidence_quality": "INSUFFICIENT",
        "confidence": 0.0,
        "reason_codes": ["INSUFFICIENT_EVIDENCE"],
        "evidence": [],
    }
