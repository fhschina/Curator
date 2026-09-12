# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Selected-source proof encoding with independent bilateral context witnesses."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.judging import coverage_selection
from eval.dedup.judging import critic_retention_v2 as previous
from eval.dedup.validation import require

CONTRACT = "dedup-critic-retention-v3"
FIELDS = (
    "a_loss_span_id",
    "b_loss_span_id",
    "a_context_span_id",
    "b_context_span_id",
    "conflict",
    "overlap_basis",
    "shared_anchor_ids",
    "explanation",
)


def response_schema() -> dict:
    properties = {
        key: {"type": "string", "minLength": 0 if "loss" in key else 1} for key in FIELDS if key.endswith("span_id")
    }
    properties.update(
        conflict={"type": "string", "enum": ["NONE", *previous.veto.CONFLICTS]},
        overlap_basis={"type": "string", "enum": list(previous.OVERLAP)},
        shared_anchor_ids={"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        explanation={"type": "string", "minLength": 1},
    )
    return {"type": "object", "additionalProperties": False, "required": list(FIELDS), "properties": properties}


def compile_review(value: dict, payload: dict) -> dict:
    require(
        isinstance(value, dict) and set(value) == set(FIELDS),
        "RETENTION_V3_SCHEMA",
        "one concrete result with exact fields",
    )
    require(
        all(isinstance(value[k], str) for k in FIELDS if k != "shared_anchor_ids")
        and value["conflict"] in ("NONE", *previous.veto.CONFLICTS)
        and value["overlap_basis"] in previous.OVERLAP
        and value["explanation"].strip(),
        "RETENTION_V3_SCHEMA",
        "IDs and categories are strings; no arrays, booleans or generated schema objects",
    )
    spans = coverage_selection.span_inventory(payload)
    compiled = {key: deepcopy(value[key]) for key in ("conflict", "overlap_basis", "shared_anchor_ids", "explanation")}
    for side in ("a", "b"):
        source, context = value[f"{side}_loss_span_id"], value[f"{side}_context_span_id"]
        require(bool(context), "RETENTION_V3_CONTEXT", "select actual checked context on each side, even without loss")
        selected = []
        if source:
            quote = coverage_selection._selected_quote(source, spans, side.upper(), unique=True)
            selected.append({"span_id": source, "quote": quote})
        quote = coverage_selection._selected_quote(context, spans, side.upper(), unique=False)
        if context != source:
            selected.append({"span_id": context, "quote": quote})
        compiled[f"{side}_evidence"] = selected
        compiled[f"{side}_has_uncovered_content"] = bool(source) or value["conflict"] != "NONE"
    return compiled


def apply_review(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    routing = previous.veto.route(main, payload)
    if routing != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return deepcopy(main), routing
    return previous.apply_review(main, payload, compile_review(value, payload))
