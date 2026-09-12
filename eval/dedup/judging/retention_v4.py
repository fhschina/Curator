# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental source-retention main/critic adapter with no record-profile veto."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.judging import coverage_selection as selection
from eval.dedup.judging import critic_intervention as witnesses
from eval.dedup.judging import critic_subject_binding as subjects
from eval.dedup.judging import critic_subject_proof_verifier as verifier
from eval.dedup.judging import critic_subject_scope as scope
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.schema_v2 import DOMINANT_OVERLAP_SOURCES
from eval.dedup.judging.schema_v4 import unresolved_judge_output_v4, validate_judge_output_v4
from eval.dedup.validation import require

CONTRACT = "dedup-retention-directions-v4"
LOSS_KINDS = ("NONE", "MAIN_ADDITION", "NON_MAIN_ADDITION", "UNRESOLVED")
CONFLICTS = {**witnesses.CONFLICTS, "NEGATION_CONFLICT": ("NEGATION_CHANGE", "OTHER")}


def response_schema() -> dict:
    properties = {}
    for side in ("a", "b"):
        properties[f"{side}_loss_kind"] = {"type": "string", "enum": list(LOSS_KINDS)}
        properties[f"{side}_loss_span_id"] = {"type": "string"}
        properties[f"{side}_context_span_id"] = {"type": "string"}
    properties.update(
        shared_basis={"type": "string", "enum": ["PRESENT", "NONE", "UNRESOLVED"]},
        conflict={"type": "string", "enum": ["NONE", *CONFLICTS]},
        material_difference={"type": "string", "enum": ["NONE", "MINOR", "MAJOR", "UNRESOLVED"]},
        dominant_overlap_source={"type": "string", "enum": list(DOMINANT_OVERLAP_SOURCES)},
        confidence_tier={"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        explanation={"type": "string", "minLength": 1},
    )
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def input_route(payload: dict) -> str:
    if payload.get("long_document_evidence", {}).get("truncated") is not False:
        return "TRUNCATED_INPUT"
    if payload.get("semantic_diff_evidence", {}).get("status") != "COMPLETE":
        return "INCOMPLETE_SPAN_PACKET"
    a, b = (payload[f"document_{s}"]["text"] for s in ("a", "b"))
    if not all(isinstance(t, str) and t.strip() for t in (a, b)):
        return "EMPTY_OR_MISSING_INPUT"
    selection.span_inventory(payload)
    return "EXACT_INPUT" if a == b else "MODEL_REVIEW"


def exact_output() -> dict:
    value = {
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
        "same_duplicate_group": "YES",
        "relation_type": "EXACT",
        "material_difference": "NONE",
        "primary_material_difference": "NONE",
        "dominant_overlap_source": "MAIN_CONTENT",
        "primary_risk_factor": "NONE",
        "confidence_tier": "HIGH",
        "reason_codes": ["RETENTION_V4:COMPLETE_EXACT_INPUT"],
        "evidence": [],
    }
    return validate_judge_output_v4(value)


def checked_review(value: dict, payload: dict) -> list[dict]:
    schema = response_schema()
    require(
        isinstance(value, dict) and set(value) == set(schema["required"]),
        "RETENTION_V4_FIELDS",
        "exact new retention fields required; legacy ledgers are not this contract",
    )
    for key, spec in schema["properties"].items():
        require(
            isinstance(value[key], str) and ("enum" not in spec or value[key] in spec["enum"]),
            "RETENTION_V4_VALUE",
            "invalid retention field",
            field=key,
        )
    require(bool(value["explanation"].strip()), "RETENTION_V4_EXPLANATION", "nonempty explanation required")
    spans = selection.span_inventory(payload)
    evidence = []
    for side in ("a", "b"):
        loss, sid, context = (value[f"{side}_{key}"] for key in ("loss_kind", "loss_span_id", "context_span_id"))
        require(
            (loss in {"MAIN_ADDITION", "NON_MAIN_ADDITION"}) == bool(sid),
            "RETENTION_V4_LOSS",
            "an uncovered side requires its own unique loss witness; NONE/UNRESOLVED must not assert loss",
        )
        require(bool(context), "RETENTION_V4_CONTEXT", "both sides need an independently checked context")
        for selected, unique in ((sid, True), (context, False)):
            if selected:
                quote = selection._selected_quote(selected, spans, side.upper(), unique=unique)
                item = witnesses._witness({"span_id": selected, "quote": quote}, side.upper(), spans, payload)
                if item not in evidence:
                    evidence.append(item)
    if value["conflict"] != "NONE":
        require(
            value["material_difference"] == "MAJOR", "RETENTION_V4_CONFLICT", "actual incompatible meaning is major"
        )
        require(
            any(spans[value[f"{s}_context_span_id"]]["kind"] != "SHARED" for s in ("a", "b")),
            "RETENTION_V4_CONFLICT",
            "at least one unique context must ground the actual conflict",
        )
    require(len(evidence) <= 4, "RETENTION_V4_EVIDENCE", "bounded exact evidence required")
    return evidence


def adapt_main(value: dict | None, payload: dict) -> dict:
    routing = input_route(payload)
    if routing == "EXACT_INPUT":
        return exact_output()
    if routing != "MODEL_REVIEW":
        return unresolved_judge_output_v4("RETENTION_V4:" + routing)
    evidence = checked_review(value, payload)
    losses = (value["a_loss_kind"], value["b_loss_kind"])
    if "UNRESOLVED" in (
        *losses,
        value["shared_basis"],
        value["material_difference"],
        value["dominant_overlap_source"],
    ):
        return unresolved_judge_output_v4("RETENTION_V4:EXPLICIT_COVERAGE_UNCERTAINTY")
    conflict = value["conflict"]
    if conflict != "NONE":
        a = b = "NO"
        relation = "VERSION_RELATED" if conflict == "STATE_CONFLICT" else "RELATED_NON_DUPLICATE"
        difference, risk = CONFLICTS[conflict]
    else:
        a, b = ("YES" if loss == "NONE" else "NO" for loss in reversed(losses))
        if "YES" in (a, b):
            require(
                value["shared_basis"] == "PRESENT",
                "RETENTION_V4_NONEMPTY_BASIS",
                "replacement needs positive evidence of retained content, including non-main text",
            )
        if a == b == "YES":
            relation, difference, risk = (
                "NEAR_SURFACE",
                "NONE" if value["material_difference"] == "NONE" else "OTHER_MATERIAL",
                "NONE",
            )
        elif a != b:
            kind = next(loss for loss in losses if loss != "NONE")
            relation, difference, risk = (
                "CONTAINMENT",
                "MAIN_CONTENT_ADDITION_DELETION" if kind == "MAIN_ADDITION" else "NON_MAIN_CONTENT_ADDITION_DELETION",
                "CONTAINMENT_ASYMMETRY",
            )
        else:
            relation = "UNRELATED" if value["shared_basis"] == "NONE" else "RELATED_NON_DUPLICATE"
            difference, risk = "OTHER_MATERIAL", "OTHER"
        if "MAIN_ADDITION" in losses:
            require(
                value["material_difference"] == "MAJOR", "RETENTION_V4_SEVERITY", "uncovered main content is major"
            )
    result = {
        "a_can_replace_b": a,
        "b_can_replace_a": b,
        "same_duplicate_group": "YES" if "YES" in (a, b) else "NO",
        "relation_type": relation,
        "material_difference": value["material_difference"],
        "primary_material_difference": difference,
        "dominant_overlap_source": value["dominant_overlap_source"],
        "primary_risk_factor": risk,
        "confidence_tier": "MEDIUM" if value["confidence_tier"] == "HIGH" else value["confidence_tier"],
        "reason_codes": [
            f"RETENTION_V4:A_LOSS:{losses[0]}",
            f"RETENTION_V4:B_LOSS:{losses[1]}",
            f"RETENTION_V4:CONFLICT:{conflict}",
        ],
        "evidence": evidence,
    }
    validate_judge_output_v4(result)
    validate_evidence_offsets(result, payload)
    return result


def critic_route(main: dict, payload: dict) -> str:
    validate_judge_output_v4(main)
    validate_evidence_offsets(main, payload)
    if main["same_duplicate_group"] != "YES":
        return "PRESERVE_NEGATIVE_OR_UNRESOLVED"
    return "REVIEW_POSITIVE_DIRECTIONS_ONLY" if input_route(payload) == "MODEL_REVIEW" else "PRESERVE_INPUT_ROUTE"


def apply_critic(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    routing = critic_route(main, payload)
    if routing != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return deepcopy(main), routing
    proposal = adapt_main(value, payload)
    if proposal["same_duplicate_group"] == "UNRESOLVED":
        return proposal, "EXPLICIT_CRITIC_UNCERTAINTY"
    fields = ("a_can_replace_b", "b_can_replace_a")
    rejected = [k for k in fields if main[k] == "YES" and proposal[k] == "NO"]
    if not rejected:
        return deepcopy(main), "NO_SUPPORTED_DIRECTIONAL_OBJECTION"
    result = deepcopy(proposal)
    for k in fields:
        result[k] = "YES" if main[k] == proposal[k] == "YES" else "NO"
    if result["a_can_replace_b"] == result["b_can_replace_a"] == "NO" and proposal["same_duplicate_group"] == "YES":
        result.update(
            same_duplicate_group="NO",
            relation_type="RELATED_NON_DUPLICATE",
            primary_material_difference="OTHER_MATERIAL",
        )
        result["material_difference"] = (
            "MAJOR" if "MAJOR" in (main["material_difference"], proposal["material_difference"]) else "MINOR"
        )
        # Keep both source-loss witnesses when different stages refute opposite directions.
        losses = [
            next(e for e in public["evidence"] if e["side"] == side)
            for public, side in (
                (main, "A" if main["b_can_replace_a"] == "NO" else "B"),
                (proposal, "A" if proposal["b_can_replace_a"] == "NO" else "B"),
            )
        ]
        result["evidence"] = losses
    result["reason_codes"].append("RETENTION_V4:CRITIC_SUPPORTED_VETO")
    validate_judge_output_v4(result)
    validate_evidence_offsets(result, payload)
    return result, "SUPPORTED_DIRECTIONAL_VETO"


def subject_route(main: dict, payload: dict) -> str:
    if critic_route(main, payload) != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return "PRESERVE_SUBJECT_ROUTE"
    kinds = {s["kind"] for s in selection.span_inventory(payload).values()}
    return (
        "REVIEW_BILATERAL_SUBJECTS" if {"A_ONLY", "B_ONLY"} <= kinds else "PRESERVE_WITHOUT_BILATERAL_UNIQUE_SUBJECTS"
    )


def subject_proof(main: dict, payload: dict, proposal: dict | None) -> dict | None:
    if subject_route(main, payload) != "REVIEW_BILATERAL_SUBJECTS":
        return None
    proof = subjects.compile_review(proposal, payload)
    return proof if proof is not None and proposal["binding_type"] in scope.VETO_BINDINGS else None


def apply_subject_verification(
    main: dict, payload: dict, proposal: dict | None, review: dict | None
) -> tuple[dict, str]:
    proof = subject_proof(main, payload, proposal)
    if proof is None:
        return deepcopy(main), "NO_IN_SCOPE_SUBJECT_PROPOSAL"
    schema = verifier.response_schema()
    require(
        isinstance(review, dict) and set(review) == set(schema["required"]),
        "RETENTION_V4_VERIFIER",
        "exact fixed-proposal verification fields required",
    )
    for key, spec in schema["properties"].items():
        require(
            isinstance(review[key], str) and ("enum" not in spec or review[key] in spec["enum"]),
            "RETENTION_V4_VERIFIER",
            "invalid fixed-proposal field",
        )
    require(bool(review["explanation"].strip()), "RETENTION_V4_VERIFIER", "nonempty verification explanation required")
    if review["comparison"] != "SUPPORTED_DIFFERENT_NAMED_TARGETS":
        return deepcopy(main), "UNSUPPORTED_FIXED_PROPOSAL_KEEP_COVERAGE"
    require(
        review["a_subject_kind"] == review["b_subject_kind"] == "NAMED_ACTUAL_TARGET",
        "RETENTION_V4_VERIFIER",
        "both selected targets must be named actual targets",
    )
    difference, risk = CONFLICTS[proof["basis"]]
    result = deepcopy(main)
    result.update(
        a_can_replace_b="NO",
        b_can_replace_a="NO",
        same_duplicate_group="NO",
        relation_type="RELATED_NON_DUPLICATE",
        material_difference="MAJOR",
        primary_material_difference=difference,
        primary_risk_factor=risk,
        confidence_tier="MEDIUM",
        evidence=witnesses.validate_review(proof, payload),
        reason_codes=["RETENTION_V4:VERIFIED_FIXED_SUBJECT_VETO"],
    )
    validate_judge_output_v4(result)
    validate_evidence_offsets(result, payload)
    return result, "VERIFIED_FIXED_SUBJECT_VETO"
