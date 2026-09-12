# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Verify a fixed subject proposal; never discover a new veto or change its targets."""

import json
from copy import deepcopy

from eval.dedup.judging import coverage_selection
from eval.dedup.judging import critic_subject_kinds as kinds
from eval.dedup.judging import critic_subject_scope as scope
from eval.dedup.validation import require

CONTRACT = "dedup-critic-subject-proof-verifier-v4"
COMPARISONS = ("SUPPORTED_DIFFERENT_NAMED_TARGETS", "UNSUPPORTED_COMPARISON", "UNCERTAIN")


def source_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "subject_proposal"}


def proposed_result(base: dict, payload: dict) -> tuple[dict, str]:
    source = source_payload(payload)
    route = scope.route(base, source)
    if route != "REVIEW_BILATERAL_SUBJECTS":
        return deepcopy(base), route
    require(
        isinstance(payload.get("subject_proposal"), dict),
        "SUBJECT_VERIFIER_PROPOSAL",
        "validated upstream proposal required",
    )
    return scope.apply_review(base, source, payload["subject_proposal"])


def route(base: dict, payload: dict) -> str:
    proposed, _ = proposed_result(base, payload)
    if proposed["same_duplicate_group"] == "NO" and base["same_duplicate_group"] == "YES":
        return "VERIFY_FIXED_SUBJECT_VETO"
    return "NO_SUPPORTED_SUBJECT_VETO_TO_VERIFY"


def response_schema() -> dict:
    properties = {f"{side}_subject_kind": {"type": "string", "enum": list(kinds.KINDS)} for side in ("a", "b")}
    properties.update(
        comparison={"type": "string", "enum": list(COMPARISONS)}, explanation={"type": "string", "minLength": 1}
    )
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def apply_review(base: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    routing = route(base, payload)
    if routing != "VERIFY_FIXED_SUBJECT_VETO":
        return deepcopy(base), routing
    require(
        isinstance(value, dict)
        and set(value) == set(response_schema()["required"])
        and all(value[f"{s}_subject_kind"] in kinds.KINDS for s in ("a", "b"))
        and value["comparison"] in COMPARISONS
        and isinstance(value["explanation"], str)
        and bool(value["explanation"].strip()),
        "SUBJECT_VERIFIER_SCHEMA",
        "exact fixed-proposal verification fields required",
    )
    if value["comparison"] == "SUPPORTED_DIFFERENT_NAMED_TARGETS":
        require(
            all(value[f"{s}_subject_kind"] == "NAMED_ACTUAL_TARGET" for s in ("a", "b")),
            "SUBJECT_VERIFIER_CONSISTENCY",
            "a supported comparison requires two named actual targets",
        )
        return proposed_result(base, payload)[0], "VERIFIED_FIXED_SUBJECT_VETO"
    return deepcopy(base), "UNSUPPORTED_FIXED_PROPOSAL_KEEP_COVERAGE"


def messages(payload: dict, system: str) -> list[dict]:
    source, proposal = source_payload(payload), payload["subject_proposal"]
    scope.previous.compile_review(proposal, source)
    spans = coverage_selection.span_inventory(source)
    selected = {"binding_type_hypothesis": proposal["binding_type"]}
    for side in ("a", "b"):
        selected[side] = {}
        for kind in ("subject", "predicate"):
            sid = proposal[f"{side}_{kind}_span_id"]
            selected[side][kind] = {
                "span_id": sid,
                "quote": coverage_selection._selected_quote(sid, spans, side.upper(), unique=kind == "subject"),
            }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "DOCUMENT A\n"
            + source["document_a"]["text"]
            + "\n\nDOCUMENT B\n"
            + source["document_b"]["text"]
            + "\n\nFIXED COMPARISON TO VERIFY (untrusted hypothesis, not a conclusion)\n"
            + json.dumps(selected, ensure_ascii=False),
        },
    ]
