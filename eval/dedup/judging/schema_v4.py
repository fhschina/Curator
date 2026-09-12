# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Retention-direction contract independent of difference severity; legacy readers stay frozen."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eval.dedup.judging import schema as legacy
from eval.dedup.judging import schema_v3 as v3
from eval.dedup.validation import require

JUDGE_SCHEMA_V4 = "dedup-judge-output-v4"
JUDGE_FIELDS_V4 = v3.JUDGE_FIELDS_V3
ADDITIONS = ("MAIN_CONTENT_ADDITION_DELETION", "NON_MAIN_CONTENT_ADDITION_DELETION")


def judge_output_schema_v4() -> dict:
    result = deepcopy(v3.judge_output_schema_v3())
    result["properties"]["primary_material_difference"]["enum"].append(ADDITIONS[1])
    return result


def validate_judge_output_v4(value: Any) -> dict:
    require(
        isinstance(value, dict) and set(value) == JUDGE_FIELDS_V4, "JUDGE_V4_FIELDS", "exact V4 output fields required"
    )
    for key, spec in judge_output_schema_v4()["properties"].items():
        if "enum" in spec:
            require(
                isinstance(value[key], str) and value[key] in spec["enum"],
                "JUDGE_V4_ENUM",
                "unknown categorical value",
                field=key,
            )
    codes = value["reason_codes"]
    require(
        isinstance(codes, list) and all(isinstance(c, str) for c in codes) and len(codes) == len(set(codes)),
        "JUDGE_V4_REASONS",
        "unique string reason codes required",
    )
    v3._validate_evidence(value["evidence"])
    a, b, group = (value[k] for k in ("a_can_replace_b", "b_can_replace_a", "same_duplicate_group"))
    relation, severity, difference = (
        value[k] for k in ("relation_type", "material_difference", "primary_material_difference")
    )
    if group == "UNRESOLVED":
        v3._validate_consistency(value)
        return value
    require(
        a in {"YES", "NO"} and b in {"YES", "NO"} and group == ("YES" if "YES" in (a, b) else "NO"),
        "JUDGE_V4_DIRECTIONS",
        "group must be derived from two resolved directions",
    )
    require(
        "UNRESOLVED" not in (relation, severity, difference, value["dominant_overlap_source"]),
        "JUDGE_V4_UNRESOLVED",
        "resolved output cannot mix unresolved semantic fields",
    )
    require(
        (severity == "NONE") == (difference == "NONE"), "JUDGE_V4_MATERIAL", "NONE severity and difference must agree"
    )
    if relation in {"EXACT", "CANONICAL_EXACT", "NEAR_SURFACE"}:
        require(
            a == b == "YES" and severity in ({"NONE", "MINOR"} if relation == "NEAR_SURFACE" else {"NONE"}),
            "JUDGE_V4_EQUIVALENCE",
            "equivalence requires both directions without major difference",
        )
    elif relation == "CONTAINMENT":
        require(
            {a, b} == {"YES", "NO"} and severity in {"MINOR", "MAJOR"} and difference in ADDITIONS,
            "JUDGE_V4_CONTAINMENT",
            "containment requires one direction and an actual main/non-main content addition",
        )
    else:
        require(
            a == b == "NO" and severity in {"MINOR", "MAJOR"},
            "JUDGE_V4_NEGATIVE",
            "nonduplicates require two unsafe directions and a nonzero difference",
        )
        if relation == "VERSION_RELATED":
            require(severity == "MAJOR", "JUDGE_V4_VERSION", "a genuine state/version conflict is major")
    if relation != "EXACT":
        require(
            {e["side"] for e in value["evidence"]} == {"A", "B"},
            "JUDGE_V4_EVIDENCE",
            "non-exact decisions require bilateral exact evidence",
        )
    return value


def unresolved_judge_output_v4(reason: str = "INSUFFICIENT_EVIDENCE") -> dict:
    result = v3.unresolved_judge_output_v3()
    result["reason_codes"] = [reason]
    return validate_judge_output_v4(result)


def read_versioned_output(value: Any, schema_version: str) -> dict:
    """Explicit dispatch, never infer the version or rewrite an old output as V4."""
    if schema_version == JUDGE_SCHEMA_V4:
        return validate_judge_output_v4(value)
    return legacy.validate_judge_output(value, schema_version)
