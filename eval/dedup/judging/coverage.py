# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Selected coverage-review contract for Judge v0.7."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.core.validation import require
from eval.dedup.judging import coverage_selection
from eval.dedup.judging import critic_intervention as veto

CONTRACT = "dedup-critic-retention-v4"
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
OVERLAP = ("RETAINED_CONTENT", "INTERFACE_ONLY", "UNCERTAIN")


def response_schema() -> dict:
    properties = {
        key: {"type": "string", "minLength": 0 if "loss" in key else 1} for key in FIELDS if key.endswith("span_id")
    }
    properties.update(
        conflict={"type": "string", "enum": ["NONE", *veto.CONFLICTS]},
        overlap_basis={"type": "string", "enum": list(OVERLAP)},
        shared_anchor_ids={"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        explanation={"type": "string", "minLength": 1},
    )
    return {"type": "object", "additionalProperties": False, "required": list(FIELDS), "properties": properties}


def _compile_review(value: dict, payload: dict) -> dict:
    require(
        isinstance(value, dict) and set(value) == set(FIELDS),
        "RETENTION_V3_SCHEMA",
        "one concrete result with exact fields",
    )
    require(
        all(isinstance(value[key], str) for key in FIELDS if key != "shared_anchor_ids")
        and value["conflict"] in ("NONE", *veto.CONFLICTS)
        and value["overlap_basis"] in OVERLAP
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


def _proof(value: dict) -> dict:
    a_loss, b_loss = value["a_has_uncovered_content"], value["b_has_uncovered_content"]
    conflict, overlap = value["conflict"], value["overlap_basis"]
    require(
        type(a_loss) is bool and type(b_loss) is bool and conflict in ("NONE", *veto.CONFLICTS) and overlap in OVERLAP,
        "RETENTION_SCHEMA",
        "loss flags must be booleans and categories must belong to the bound contract",
    )
    if conflict != "NONE":
        require(a_loss and b_loss, "RETENTION_CONFLICT", "an incompatible retained slot makes both contents unsafe")
        action, basis = "REJECT_BOTH", conflict
    elif overlap == "UNCERTAIN":
        action, basis = "ABSTAIN", "INSUFFICIENT_EVIDENCE"
    elif overlap == "INTERFACE_ONLY" and (a_loss or b_loss):
        action, basis = "REJECT_EMPTY_ANCHOR", "EMPTY_SHARED_ANCHOR"
    else:
        action = {
            (False, False): "KEEP_MAIN",
            (True, False): "REJECT_B_REPLACES_A",
            (False, True): "REJECT_A_REPLACES_B",
            (True, True): "REJECT_BOTH",
        }[(a_loss, b_loss)]
        basis = "NO_MATERIAL_OBJECTION" if action == "KEEP_MAIN" else "UNCOVERED_CONTENT"
    return {
        "contract_version": veto.CONTRACT,
        "action": action,
        "basis": basis,
        **{key: deepcopy(value[key]) for key in ("a_evidence", "b_evidence", "shared_anchor_ids", "explanation")},
    }


def apply_review(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    routing = veto.route(main, payload)
    if routing != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return deepcopy(main), routing
    compiled = _compile_review(value, payload)
    spans = {item["span_id"]: item for item in payload["semantic_diff_evidence"]["spans"]}
    for side in ("a", "b"):
        if compiled[f"{side}_has_uncovered_content"] and compiled["conflict"] == "NONE":
            require(
                any(spans[item["span_id"]]["kind"] == f"{side.upper()}_ONLY" for item in compiled[f"{side}_evidence"]),
                "RETENTION_SOURCE_WITNESS",
                "a source loss requires a unique witness from that same source, never a shared span",
            )
    return veto.apply_review(main, payload, _proof(compiled))
