# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Source-oriented loss proofs over an immutable v0.6.2.12 main decision."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.judging import critic_intervention as veto
from eval.dedup.validation import require

CONTRACT = "dedup-critic-retention-v2"
OVERLAP = ("RETAINED_CONTENT", "INTERFACE_ONLY", "UNCERTAIN")


def format_only_schema() -> dict:
    schema = deepcopy(veto.response_schema())
    schema["required"].remove("contract_version")
    del schema["properties"]["contract_version"]
    return schema


def response_schema() -> dict:
    witness_fields = veto.response_schema()["properties"]
    properties = {
        "a_has_uncovered_content": {"type": "boolean"},
        "b_has_uncovered_content": {"type": "boolean"},
        "conflict": {"type": "string", "enum": ["NONE", *veto.CONFLICTS]},
        "overlap_basis": {"type": "string", "enum": list(OVERLAP)},
        **{
            key: deepcopy(witness_fields[key])
            for key in ("a_evidence", "b_evidence", "shared_anchor_ids", "explanation")
        },
    }
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def proof(value: dict, *, format_only: bool = False) -> dict:
    """Derive veto directions; metadata and inverse direction names are code-owned."""
    schema = format_only_schema() if format_only else response_schema()
    require(
        isinstance(value, dict) and set(value) == set(schema["required"]),
        "RETENTION_SCHEMA",
        "exact response fields required; metadata is supplied by the bound request contract",
    )
    if format_only:
        return {"contract_version": veto.CONTRACT, **deepcopy(value)}
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


def apply_review(main: dict, payload: dict, value: dict | None, *, format_only: bool = False) -> tuple[dict, str]:
    routing = veto.route(main, payload)
    if routing != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return deepcopy(main), routing
    derived = proof(value, format_only=format_only)
    if not format_only:
        # Even abstention/empty-anchor responses may not call shared text one-sided loss.
        spans = {s["span_id"]: s for s in payload["semantic_diff_evidence"]["spans"]}
        for side in ("a", "b"):
            if value[f"{side}_has_uncovered_content"] and value["conflict"] == "NONE":
                evidence = value[f"{side}_evidence"]
                require(
                    isinstance(evidence, list)
                    and any(
                        isinstance(item, dict)
                        and isinstance(item.get("span_id"), str)
                        and spans.get(item["span_id"], {}).get("kind") == f"{side.upper()}_ONLY"
                        for item in evidence
                    ),
                    "RETENTION_SOURCE_WITNESS",
                    "a source loss requires a unique witness from that same source, never a shared span",
                )
    return veto.apply_review(main, payload, derived)
