# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Separate unique additions from contextual conflicts over shared assertions."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.judging import retention_v4 as retention
from eval.dedup.judging import retention_v4_proof as proof
from eval.dedup.validation import require

CONTRACT = "dedup-retention-context-v1"
RENAMES = {
    **{f"{side}_addition_{field}": f"{side}_loss_{field}" for side in ("a", "b") for field in ("kind", "span_id")},
    "context_conflict": "conflict",
}
input_route = retention.input_route
critic_route = retention.critic_route
subject_route = retention.subject_route
subject_proof = retention.subject_proof
apply_subject_verification = retention.apply_subject_verification
validate_evidence_offsets = retention.validate_evidence_offsets


def response_schema() -> dict:
    old = retention.response_schema()
    reverse = {v: k for k, v in RENAMES.items()}
    properties = {reverse.get(k, k): spec for k, spec in old["properties"].items()}
    return {**old, "properties": properties, "required": list(properties)}


def proof_review(value: dict) -> dict:
    """Map typed fields to the frozen proof adapter, never infer fields from prose."""
    require(
        isinstance(value, dict) and set(value) == set(response_schema()["required"]),
        "RETENTION_CONTEXT_FIELDS",
        "explicit addition/context contract required; legacy loss responses cannot be reinterpreted",
    )
    return {RENAMES.get(k, k): deepcopy(v) for k, v in value.items()}


def annotate(public: dict) -> dict:
    result = deepcopy(public)
    result["reason_codes"] = [
        code.replace("RETENTION_V4:A_LOSS:", "RETENTION_CONTEXT:A_ADDITION:")
        .replace("RETENTION_V4:B_LOSS:", "RETENTION_CONTEXT:B_ADDITION:")
        .replace("RETENTION_V4:CONFLICT:", "RETENTION_CONTEXT:CONFLICT:")
        for code in result["reason_codes"]
    ]
    return result


def adapt_main(value: dict | None, payload: dict) -> dict:
    if input_route(payload) != "MODEL_REVIEW":
        return retention.adapt_main(None, payload)
    # A supported conflict independently refutes both directions. NONE additions
    # mean no separate unique claim, not semantic equivalence or safe replacement.
    return annotate(proof.adapt_main(proof_review(value), payload))


def apply_critic(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    route = critic_route(main, payload)
    if route != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return deepcopy(main), route
    public, rule = proof.apply_critic(main, payload, proof_review(value))
    return annotate(public), rule
