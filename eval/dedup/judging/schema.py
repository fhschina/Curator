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

"""Versioned judge output schemas and strict local validation."""

from __future__ import annotations

from typing import Any, Final

from eval.dedup.contracts import (
    REASON_CODES,
    DuplicateAnswer,
    FuzzyScope,
    MaterialDifference,
    RelationType,
)
from eval.dedup.judging.schema_v1 import (
    JUDGE_SCHEMA_V1,
    judge_output_schema_v1,
    unresolved_judge_output_v1,
    validate_judge_output_v1,
)
from eval.dedup.validation import DedupEvaluationError, require

JUDGE_SCHEMA_V0: Final = "dedup-judge-output-v0"
SUPPORTED_JUDGE_SCHEMA_VERSIONS: Final = frozenset({JUDGE_SCHEMA_V0, JUDGE_SCHEMA_V1})

JUDGE_FIELDS: Final = {
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


def judge_output_schema(schema_version: str = JUDGE_SCHEMA_V0) -> dict[str, Any]:
    if schema_version == JUDGE_SCHEMA_V1:
        return judge_output_schema_v1()
    require(
        schema_version == JUDGE_SCHEMA_V0,
        "UNSUPPORTED_JUDGE_SCHEMA_VERSION",
        "judge output schema version is not supported",
        schema_version=schema_version,
    )
    evidence = {
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
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(JUDGE_FIELDS),
        "properties": {
            "same_duplicate_group": {"type": "string", "enum": list(DuplicateAnswer)},
            "a_can_replace_b": {"type": "string", "enum": list(DuplicateAnswer)},
            "b_can_replace_a": {"type": "string", "enum": list(DuplicateAnswer)},
            "relation_type": {"type": "string", "enum": list(RelationType)},
            "material_difference": {"type": "string", "enum": list(MaterialDifference)},
            "fuzzy_scope": {"type": "string", "enum": list(FuzzyScope)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason_codes": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(REASON_CODES)},
            },
            "evidence": {"type": "array", "maxItems": 4, "items": evidence},
        },
    }


def validate_judge_output(value: Any, schema_version: str = JUDGE_SCHEMA_V0) -> dict[str, Any]:
    """Validate types, enums, evidence bounds, and cross-field consistency."""

    if schema_version == JUDGE_SCHEMA_V1:
        return validate_judge_output_v1(value)
    judge_output_schema(schema_version)
    require(isinstance(value, dict), "JUDGE_SCHEMA_INVALID", "judge output must be a JSON object")
    actual_fields = set(value)
    require(
        actual_fields == JUDGE_FIELDS,
        "JUDGE_SCHEMA_INVALID",
        "judge output fields differ",
        expected_fields=sorted(JUDGE_FIELDS),
        actual_fields=sorted(actual_fields),
        missing_fields=sorted(JUDGE_FIELDS - actual_fields),
        extra_fields=sorted(actual_fields - JUDGE_FIELDS),
    )
    for field in ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a"):
        require(
            isinstance(value[field], str) and value[field] in DuplicateAnswer,
            "JUDGE_SCHEMA_INVALID",
            "invalid ternary answer",
            field=field,
        )
    require(
        isinstance(value["relation_type"], str) and value["relation_type"] in RelationType,
        "JUDGE_SCHEMA_INVALID",
        "invalid relation_type",
    )
    require(
        isinstance(value["material_difference"], str) and value["material_difference"] in MaterialDifference,
        "JUDGE_SCHEMA_INVALID",
        "invalid material_difference",
    )
    require(
        isinstance(value["fuzzy_scope"], str) and value["fuzzy_scope"] in FuzzyScope,
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
    reasons = value["reason_codes"]
    require(isinstance(reasons, list), "JUDGE_SCHEMA_INVALID", "reason_codes must be a list")
    require(all(isinstance(reason, str) for reason in reasons), "JUDGE_SCHEMA_INVALID", "reason codes must be strings")
    require(len(reasons) == len(set(reasons)), "JUDGE_SCHEMA_INVALID", "reason_codes must be unique")
    require(all(reason in REASON_CODES for reason in reasons), "JUDGE_SCHEMA_INVALID", "unknown reason code")
    evidence = value["evidence"]
    require(
        isinstance(evidence, list) and len(evidence) <= 4,
        "JUDGE_SCHEMA_INVALID",
        "evidence must contain at most four items",
    )
    for index, item in enumerate(evidence):
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
            and isinstance(item["end_char"], int)
            and 0 <= item["start_char"] <= item["end_char"],
            "JUDGE_SCHEMA_INVALID",
            "evidence offsets are invalid",
            index=index,
        )
        require(
            isinstance(item["quote"], str) and len(item["quote"]) <= 240,
            "JUDGE_SCHEMA_INVALID",
            "evidence quote is invalid",
        )
    require(
        not (
            value["relation_type"] == RelationType.UNRELATED and value["same_duplicate_group"] == DuplicateAnswer.YES
        ),
        "JUDGE_CONSISTENCY_INVALID",
        "UNRELATED cannot be in the same duplicate group",
    )
    require(
        not (
            value["material_difference"] == MaterialDifference.MAJOR
            and value["a_can_replace_b"] == DuplicateAnswer.YES
            and value["b_can_replace_a"] == DuplicateAnswer.YES
        ),
        "JUDGE_CONSISTENCY_INVALID",
        "MAJOR difference cannot be silently bidirectionally replaceable",
    )
    return value


def unresolved_judge_output(
    reason: str = "INSUFFICIENT_EVIDENCE",
    schema_version: str = JUDGE_SCHEMA_V0,
) -> dict[str, Any]:
    if schema_version == JUDGE_SCHEMA_V1:
        return unresolved_judge_output_v1()
    judge_output_schema(schema_version)
    require(reason in REASON_CODES, "INVALID_REASON_CODE", "unresolved output reason is invalid")
    return {
        "same_duplicate_group": DuplicateAnswer.UNRESOLVED,
        "a_can_replace_b": DuplicateAnswer.UNRESOLVED,
        "b_can_replace_a": DuplicateAnswer.UNRESOLVED,
        "relation_type": RelationType.UNRESOLVED,
        "material_difference": MaterialDifference.UNRESOLVED,
        "fuzzy_scope": FuzzyScope.UNRESOLVED,
        "confidence": 0.0,
        "reason_codes": [reason],
        "evidence": [],
    }


def parse_judge_json(text: str) -> Any:
    """Parse a provider response as strict JSON without normalizing it."""

    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DedupEvaluationError("JUDGE_INVALID_JSON", "judge response is not strict JSON", line=exc.lineno) from exc


def flatten_reason_codes(value: Any) -> list[str]:
    """Flatten either schema version into namespaced labels for reports and dashboards."""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if not isinstance(value, dict):
        return []
    flattened = [f"MATERIAL_DELTA:{item}" for item in value.get("material_differences", [])]
    flattened.extend(f"OVERLAP_SOURCE:{item}" for item in value.get("overlap_sources", []))
    primary = value.get("primary_risk_factor")
    if isinstance(primary, str):
        flattened.append(f"PRIMARY_RISK:{primary}")
    flattened.extend(f"SECONDARY_RISK:{item}" for item in value.get("secondary_risk_factors", []))
    quality = value.get("evidence_quality")
    if isinstance(quality, dict):
        status = quality.get("status")
        if isinstance(status, str):
            flattened.append(f"EVIDENCE_STATUS:{status}")
        flattened.extend(f"EVIDENCE_ISSUE:{item}" for item in quality.get("issues", []))
    return flattened


def parse_and_validate_json(
    text: str,
    schema_version: str = JUDGE_SCHEMA_V0,
) -> dict[str, Any]:
    return validate_judge_output(parse_judge_json(text), schema_version)
