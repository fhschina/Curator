# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Accept already-selected conflict values alongside their shared predicate context."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.judging import retention_v4 as previous
from eval.dedup.validation import require

CONTRACT = "dedup-retention-directions-v4-proof1"
input_route = previous.input_route
critic_route = previous.critic_route
subject_route = previous.subject_route
subject_proof = previous.subject_proof
apply_subject_verification = previous.apply_subject_verification
response_schema = previous.response_schema


def conflict_evidence(value: dict, payload: dict) -> list[dict]:
    require(
        isinstance(value, dict) and value.get("conflict") in previous.CONFLICTS,
        "RETENTION_PROOF_CONFLICT",
        "a declared actual conflict is required",
    )
    # Validate all original selected spans. Only the obsolete context-field-only
    # uniqueness check is omitted; no ID, quote, side or loss declaration changes.
    structural = deepcopy(value)
    structural["conflict"] = "NONE"
    evidence = previous.checked_review(structural, payload)
    spans = previous.selection.span_inventory(payload)
    selected = [value[f"{s}_{field}"] for s in ("a", "b") for field in ("loss_span_id", "context_span_id")]
    require(value["material_difference"] == "MAJOR", "RETENTION_PROOF_SEVERITY", "actual conflicts remain major")
    require(
        any(sid and spans[sid]["kind"] != "SHARED" for sid in selected),
        "RETENTION_PROOF_UNIQUE",
        "a declared conflict must cite an actual unique value/context in its existing proof",
    )
    return evidence


def adapt_main(value: dict | None, payload: dict) -> dict:
    if input_route(payload) != "MODEL_REVIEW" or not isinstance(value, dict) or value.get("conflict") == "NONE":
        return previous.adapt_main(value, payload)
    evidence = conflict_evidence(value, payload)
    if "UNRESOLVED" in (
        value["a_loss_kind"],
        value["b_loss_kind"],
        value["shared_basis"],
        value["dominant_overlap_source"],
    ):
        return previous.unresolved_judge_output_v4("RETENTION_V4:EXPLICIT_COVERAGE_UNCERTAINTY")
    difference, risk = previous.CONFLICTS[value["conflict"]]
    result = {
        "a_can_replace_b": "NO",
        "b_can_replace_a": "NO",
        "same_duplicate_group": "NO",
        "relation_type": "VERSION_RELATED" if value["conflict"] == "STATE_CONFLICT" else "RELATED_NON_DUPLICATE",
        "material_difference": "MAJOR",
        "primary_material_difference": difference,
        "primary_risk_factor": risk,
        "dominant_overlap_source": value["dominant_overlap_source"],
        "confidence_tier": "MEDIUM" if value["confidence_tier"] == "HIGH" else value["confidence_tier"],
        "reason_codes": [
            f"RETENTION_V4:A_LOSS:{value['a_loss_kind']}",
            f"RETENTION_V4:B_LOSS:{value['b_loss_kind']}",
            f"RETENTION_V4:CONFLICT:{value['conflict']}",
            "RETENTION_V4_PROOF1:EXISTING_SELECTED_CONFLICT_EVIDENCE",
        ],
        "evidence": evidence,
    }
    previous.validate_judge_output_v4(result)
    previous.validate_evidence_offsets(result, payload)
    return result


def apply_critic(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    route = critic_route(main, payload)
    if route != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return deepcopy(main), route
    if not isinstance(value, dict) or value.get("conflict") == "NONE":
        return previous.apply_critic(main, payload, value)
    result = adapt_main(value, payload)
    if result["same_duplicate_group"] == "UNRESOLVED":
        return result, "EXPLICIT_CRITIC_UNCERTAINTY"
    result["reason_codes"].append("RETENTION_V4:CRITIC_SUPPORTED_VETO")
    return result, "SUPPORTED_DIRECTIONAL_VETO"
