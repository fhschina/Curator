# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Discriminated retained-meaning results with explicit compilation to frozen V2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from eval.dedup.judging.coverage_selection import compile_selection, selection_schema, span_inventory
from eval.dedup.judging.coverage_witness import _object, _refs, _text
from eval.dedup.judging.payload_transport import validate_payload_transport
from eval.dedup.validation import require, sha256_json

CONTRACT = "dedup-retained-coverage-v3"
_SIDES = ("a_meaning_in_b", "b_meaning_in_a")


def typed_coverage_schema() -> dict:
    base = selection_schema()
    old_side = base["properties"][_SIDES[0]]["properties"]
    common = {k: deepcopy(old_side[k]) for k in ("reviewed_unique_ids", "coverage_explanation")}
    defs = {
        "CoveredSide": _object(
            {
                **deepcopy(common),
                "harmless_unique_ids": _text(
                    "Own unique IDs whose ENTIRE meaning is harmless UI/repetition; never mark a mixed retained clause harmless."
                ),
                "opposite_support_ids": _text(
                    "Opposite/shared IDs: actual semantic counterparts for every non-harmless unique meaning; optional explanatory context only if all unique meaning is harmless or absent."
                ),
                "status": _text("Every retained meaning of this side is preserved by the opposite side.", "COVERED"),
            }
        ),
        "UncoveredSide": _object(
            {
                **deepcopy(common),
                **{
                    k: deepcopy(old_side[k])
                    for k in (
                        "source_span_id",
                        "counterpart_span_id",
                        "retention_consequence",
                        "uncovered_type",
                        "counterpart_relation",
                    )
                },
                "status": _text("A concrete retained proposition is not preserved by the opposite side.", "UNCOVERED"),
            }
        ),
        "UnresolvedSide": _object(
            {
                **deepcopy(common),
                "checked_opposite_ids": _text(
                    "Opposite/shared IDs examined without reaching a semantic decision; may be empty."
                ),
                "status": _text("Coverage remains unknown, not a negative or a positive.", "UNRESOLVED"),
            }
        ),
    }
    defs["UncoveredSide"]["properties"]["uncovered_type"]["enum"].remove("NONE")
    defs["UncoveredSide"]["properties"]["counterpart_relation"]["enum"].remove("NOT_APPLICABLE")
    union = {
        "oneOf": [{"$ref": f"#/$defs/{name}"} for name in defs],
        "discriminator": {
            "propertyName": "status",
            "mapping": {
                "COVERED": "#/$defs/CoveredSide",
                "UNCOVERED": "#/$defs/UncoveredSide",
                "UNRESOLVED": "#/$defs/UnresolvedSide",
            },
        },
    }
    props = {
        k: deepcopy(base["properties"][k])
        for k in (
            "contract_version",
            "input_status",
            "scope_explanation",
            "anchor_a_ids",
            "anchor_b_ids",
            "record_scope",
        )
    }
    props["contract_version"]["enum"] = [CONTRACT]
    return {**_object({**props, **{field: deepcopy(union) for field in _SIDES}}), "$defs": defs}


def validate_typed_shape(value: Any) -> None:
    from jsonschema import Draft202012Validator

    errors = list(Draft202012Validator(typed_coverage_schema()).iter_errors(value))
    require(not errors, "TYPED_COVERAGE_CONTRACT_INVALID", "exact typed branch fields, enums and types required")


@dataclass(frozen=True)
class CompiledCoverage:
    selection: dict
    audit: dict


def compile_typed_coverage(value: Any, payload: dict) -> CompiledCoverage:
    """Derive encoding fields, never infer a verdict from prose or repair a legacy response."""
    validate_typed_shape(value)
    spans = span_inventory(payload)
    compiled = {k: deepcopy(v) for k, v in value.items() if k not in _SIDES}
    compiled["contract_version"] = selection_schema()["properties"]["contract_version"]["enum"][0]
    audit = {}
    for side, field in zip(("A", "B"), _SIDES, strict=True):
        item = value[field]
        opposite = "B" if side == "A" else "A"
        unique = {sid for sid, s in spans.items() if s.get("side") == side and s["kind"] != "SHARED"}
        derived = {
            "status": item["status"],
            "reviewed_unique_ids": item["reviewed_unique_ids"],
            "coverage_explanation": item["coverage_explanation"],
            "coverage_mode": "NOT_COVERED",
            "coverage_counterpart_ids": "",
            "uncovered_type": "NONE",
            "source_span_id": "",
            "counterpart_span_id": "",
            "counterpart_relation": "NOT_APPLICABLE",
            "retention_consequence": "",
        }
        context = []
        if item["status"] == "COVERED":
            harmless = set(_refs(item["harmless_unique_ids"], spans, side))
            support = _refs(item["opposite_support_ids"], spans, opposite)
            require(
                harmless <= unique, "TYPED_HARMLESS_IDS", "harmless IDs must be own unique, never shared or foreign"
            )
            retained = unique - harmless
            require(
                not retained or bool(support),
                "TYPED_COUNTERPART_MISSING",
                "non-harmless unique meanings need independent semantic counterparts",
            )
            if not unique:
                mode = "NO_UNIQUE_SPANS"
            elif not retained:
                mode = "HARMLESS_ONLY"
            else:
                mode = "MIXED" if harmless else "SEMANTIC_COUNTERPARTS"
                derived["coverage_counterpart_ids"] = item["opposite_support_ids"]
            derived["coverage_mode"] = mode
            context = support if not retained else []
            audit[field] = {"harmless_unique_ids": sorted(harmless), "retained_unique_ids": sorted(retained)}
        elif item["status"] == "UNCOVERED":
            derived.update(
                {
                    k: item[k]
                    for k in (
                        "uncovered_type",
                        "source_span_id",
                        "counterpart_span_id",
                        "counterpart_relation",
                        "retention_consequence",
                    )
                }
            )
        else:
            context = _refs(item["checked_opposite_ids"], spans, opposite)
        audit.setdefault(field, {}).update(derived_mode=derived["coverage_mode"], context_only_ids=context)
        compiled[field] = derived
    compile_selection(compiled, payload)
    return CompiledCoverage(compiled, {"compiler_contract": CONTRACT, "side_derivations": audit})


def bind_typed_response(value: Any, observed: Any) -> tuple[dict, dict]:
    """Only a strictly valid raw response can authorize known Arrow null padding in its echo."""
    validate_typed_shape(value)
    schema = typed_coverage_schema()
    side_fields = set().union(*(d["properties"] for d in schema["$defs"].values()))
    require(
        isinstance(observed, dict) and observed.keys() == value.keys(),
        "FOLLOWUP_BOUNDARY_TYPED_RESPONSE",
        "root response fields changed",
    )
    for field in _SIDES:
        require(
            isinstance(observed.get(field), dict) and observed[field].keys() <= side_fields,
            "FOLLOWUP_BOUNDARY_TYPED_RESPONSE",
            "only known typed side fields may acquire null padding",
        )
    changes = validate_payload_transport(value, observed)
    return deepcopy(value), {
        "transport_contract": "dedup-typed-coverage-binding-v1",
        "raw_response_sha256": sha256_json(value),
        "parsed_column_sha256": sha256_json(observed),
        "representation_changes": changes,
    }
