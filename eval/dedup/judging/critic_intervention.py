# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental veto-only critic layered over unchanged v0.6.2.12 main decisions."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.record_scope import complete_visible_equality
from eval.dedup.judging.schema_v3 import unresolved_judge_output_v3, validate_judge_output_v3
from eval.dedup.validation import require

CONTRACT = "dedup-v06212-critic-intervention-v1"
ACTIONS = ("KEEP_MAIN", "REJECT_A_REPLACES_B", "REJECT_B_REPLACES_A", "REJECT_BOTH", "REJECT_EMPTY_ANCHOR", "ABSTAIN")
CONFLICTS = {
    "IDENTITY_CONFLICT": ("DOCUMENT_IDENTITY_CHANGE", "TEMPLATE_SLOT_COLLISION"),
    "STATE_CONFLICT": ("OTHER_MATERIAL", "IDENTIFIER_UNDERWEIGHTING"),
    "POLICY_CONFLICT": ("LEGAL_CONTEXT_CHANGE", "LEGAL_CONTEXT_COLLISION"),
    "ROLE_CONFLICT": ("PAGE_ROLE_CHANGE", "PAGE_ROLE_COLLISION"),
    "MEMBERSHIP_CONFLICT": ("RESULT_SET_CHANGE", "LIST_SNAPSHOT_COLLISION"),
}
BASES = ("NO_MATERIAL_OBJECTION", "UNCOVERED_CONTENT", "EMPTY_SHARED_ANCHOR", "INSUFFICIENT_EVIDENCE", *CONFLICTS)


def response_schema() -> dict:
    witness = {
        "type": "object",
        "additionalProperties": False,
        "required": ["span_id", "quote"],
        "properties": {"span_id": {"type": "string"}, "quote": {"type": "string", "minLength": 1, "maxLength": 240}},
    }
    fields = {
        "contract_version": {"type": "string", "enum": [CONTRACT]},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "basis": {"type": "string", "enum": list(BASES)},
        "a_evidence": {"type": "array", "maxItems": 2, "items": witness},
        "b_evidence": {"type": "array", "maxItems": 2, "items": witness},
        "shared_anchor_ids": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        "explanation": {"type": "string", "minLength": 1},
    }
    return {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}


def _witness(item: dict, side: str, spans: dict, payload: dict) -> dict:
    require(
        isinstance(item, dict) and set(item) == {"span_id", "quote"},
        "CRITIC_SCOPE_WITNESS",
        "exact witness fields required",
    )
    sid, quote = item["span_id"], item["quote"]
    require(
        isinstance(sid, str) and sid in spans and isinstance(quote, str) and 0 < len(quote) <= 240,
        "CRITIC_SCOPE_WITNESS",
        "known span and bounded exact quote required",
    )
    span = spans[sid]
    require(
        span["kind"] == "SHARED" or span.get("side") == side,
        "CRITIC_SCOPE_SIDE",
        "witness belongs to the other document",
    )
    prefix = f"{side.lower()}_" if span["kind"] == "SHARED" else ""
    text, start = span[f"{prefix}text"], span[f"{prefix}start_char"]
    require(quote in text, "CRITIC_SCOPE_QUOTE", "quote must be verbatim inside its own span")
    start += text.index(quote)
    evidence = {"side": side, "start_char": start, "end_char": start + len(quote), "quote": quote}
    validate_evidence_offsets({"evidence": [evidence]}, payload)
    return evidence


def validate_review(value: dict, payload: dict) -> list[dict]:
    """Validate proof structure and alignment, not the model's semantic claims."""
    require(
        isinstance(value, dict) and set(value) == set(response_schema()["required"]),
        "CRITIC_SCOPE_SCHEMA",
        "exact response keys required; legacy critic output is not a scope proof",
    )
    require(
        value["contract_version"] == CONTRACT and value["action"] in ACTIONS and value["basis"] in BASES,
        "CRITIC_SCOPE_SCHEMA",
        "unknown contract, action or basis",
    )
    require(
        isinstance(value["explanation"], str) and bool(value["explanation"].strip()),
        "CRITIC_SCOPE_SCHEMA",
        "nonempty explanation required",
    )
    packet = payload.get("semantic_diff_evidence", {})
    spans = {s["span_id"]: s for s in packet.get("spans", [])}
    anchors = value["shared_anchor_ids"]
    require(
        isinstance(anchors, list)
        and all(isinstance(s, str) for s in anchors)
        and len(anchors) == len(set(anchors))
        and set(anchors) <= spans.keys()
        and all(spans[s]["kind"] == "SHARED" for s in anchors),
        "CRITIC_SCOPE_ANCHOR",
        "shared-anchor IDs must be unique existing shared spans",
    )
    evidence, unique = [], {"A": set(), "B": set()}
    for side in ("A", "B"):
        items = value[f"{side.lower()}_evidence"]
        require(isinstance(items, list) and len(items) <= 2, "CRITIC_SCOPE_WITNESS", "at most two witnesses per side")
        for item in items:
            evidence.append(_witness(item, side, spans, payload))
            if spans[item["span_id"]]["kind"] == f"{side}_ONLY":
                unique[side].add(item["span_id"])
    action, basis = value["action"], value["basis"]
    if action in {"KEEP_MAIN", "ABSTAIN"}:
        expected = "NO_MATERIAL_OBJECTION" if action == "KEEP_MAIN" else "INSUFFICIENT_EVIDENCE"
        require(basis == expected, "CRITIC_SCOPE_ACTION_BASIS", "non-veto action cannot conceal a material veto")
        return evidence
    require(
        packet.get("status") == "COMPLETE" and payload.get("long_document_evidence", {}).get("truncated") is False,
        "CRITIC_SCOPE_INCOMPLETE",
        "no conclusive veto from an incomplete packet",
    )
    require(
        {r["side"] for r in evidence} == {"A", "B"},
        "CRITIC_SCOPE_BILATERAL",
        "veto needs its own exact evidence on both sides",
    )
    if action == "REJECT_EMPTY_ANCHOR":
        shared = {sid for sid, span in spans.items() if span["kind"] == "SHARED"}
        require(
            basis == "EMPTY_SHARED_ANCHOR" and set(anchors) == shared and bool(shared) and any(unique.values()),
            "CRITIC_SCOPE_EMPTY_ANCHOR",
            "refuting the anchor requires all shared spans and a unique addition witness",
        )
    elif basis == "UNCOVERED_CONTENT":
        sources = {"REJECT_A_REPLACES_B": ("B",), "REJECT_B_REPLACES_A": ("A",), "REJECT_BOTH": ("A", "B")}[action]
        require(
            all(unique[s] for s in sources),
            "CRITIC_SCOPE_DIRECTION",
            "cite uncovered content in the document that would be discarded",
        )
    else:
        require(
            basis in CONFLICTS and action == "REJECT_BOTH" and any(unique.values()),
            "CRITIC_SCOPE_CONFLICT",
            "an actual incompatible retained value requires both directions rejected and a decisive delta",
        )
    return evidence


def main_decision(raw_main: dict, payload: dict) -> dict:
    return adapt_ndd_judge_output(raw_main, "dedup-judge-output-v3", payload=payload, record_binding_policy="v6-route")


def route(main: dict, payload: dict) -> str:
    validate_judge_output_v3(main)
    validate_evidence_offsets(main, payload)
    if main["same_duplicate_group"] != "YES":
        return "PRESERVE_MAIN_NEGATIVE_OR_UNRESOLVED"
    if complete_visible_equality(payload):
        return "PRESERVE_COMPLETE_EXACT_INPUT"
    return "REVIEW_POSITIVE_DIRECTIONS_ONLY"


def apply_review(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    """The critic may remove a YES direction, never create one or silently reverse it."""
    routing = route(main, payload)
    if routing != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return deepcopy(main), routing
    evidence = validate_review(value, payload)
    action, basis = value["action"], value["basis"]
    if action == "KEEP_MAIN":
        return deepcopy(main), "NO_SUPPORTED_OBJECTION_KEEP_MAIN"
    if action == "ABSTAIN":
        result = unresolved_judge_output_v3()
        result["reason_codes"].append("CRITIC_INTERVENTION:SEMANTIC_ABSTENTION")
        return result, "EXPLICIT_CRITIC_ABSTENTION"
    if action == "REJECT_EMPTY_ANCHOR" and main["relation_type"] != "CONTAINMENT":
        return deepcopy(main), "EMPTY_ANCHOR_VETO_OUTSIDE_CONTAINMENT_SCOPE"
    rejected = {
        "REJECT_A_REPLACES_B": {"a_can_replace_b"},
        "REJECT_B_REPLACES_A": {"b_can_replace_a"},
        "REJECT_BOTH": {"a_can_replace_b", "b_can_replace_a"},
        "REJECT_EMPTY_ANCHOR": {"a_can_replace_b", "b_can_replace_a"},
    }[action]
    if not any(main[k] == "YES" for k in rejected):
        return deepcopy(main), "OBJECTION_ONLY_TO_ALREADY_UNSAFE_DIRECTION"
    result = deepcopy(main)
    result.update(dict.fromkeys(rejected, "NO"))
    same = "YES" in {result["a_can_replace_b"], result["b_can_replace_a"]}
    primary, risk = CONFLICTS.get(basis, ("OTHER_MATERIAL", "BOILERPLATE_DOMINATED_SIMILARITY"))
    result.update(
        same_duplicate_group="YES" if same else "NO",
        relation_type="CONTAINMENT"
        if same
        else "VERSION_RELATED"
        if basis == "STATE_CONFLICT"
        else "RELATED_NON_DUPLICATE",
        material_difference="MAJOR",
        primary_material_difference="MAIN_CONTENT_ADDITION_DELETION" if same else primary,
        primary_risk_factor="CONTAINMENT_ASYMMETRY" if same else risk,
        confidence_tier="MEDIUM",
        evidence=evidence,
        reason_codes=[f"CRITIC_INTERVENTION:{action}", f"CRITIC_INTERVENTION_BASIS:{basis}"],
    )
    validate_judge_output_v3(result)
    validate_evidence_offsets(result, payload)
    return result, "SUPPORTED_DIRECTIONAL_VETO"
