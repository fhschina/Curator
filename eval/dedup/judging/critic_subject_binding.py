# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional, bilateral subject-conflict veto over an already fixed coverage result."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.core.validation import require
from eval.dedup.judging import coverage_selection
from eval.dedup.judging import critic_intervention as veto

CONTRACT = "dedup-critic-subject-binding-v1"
BINDINGS = ("NONE", "LIABILITY_PARTY", "POLICY_SERVICE", "ACCESS_TARGET", "FAILED_OBJECT", "RECORD_SUBJECT")
RELATIONS = ("SAME", "DIFFERENT", "UNCERTAIN", "NOT_APPLICABLE")


def route(main: dict, payload: dict) -> str:
    prior = veto.route(main, payload)
    if prior != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        return prior
    kinds = {s["kind"] for s in coverage_selection.span_inventory(payload).values()}
    if not {"A_ONLY", "B_ONLY"} <= kinds:
        return "PRESERVE_WITHOUT_BILATERAL_UNIQUE_SUBJECTS"
    if payload.get("long_document_evidence", {}).get("truncated") is not False:
        return "PRESERVE_INCOMPLETE_SUBJECT_CONTEXT"
    return "REVIEW_BILATERAL_SUBJECTS"


def response_schema(payload: dict | None = None) -> dict:
    fields = {
        key: {"type": "string"}
        for key in ("a_subject_span_id", "b_subject_span_id", "a_predicate_span_id", "b_predicate_span_id")
    }
    if payload is not None:
        spans = coverage_selection.span_inventory(payload)
        for side in ("a", "b"):
            fields[f"{side}_subject_span_id"]["enum"] = [
                "",
                *[sid for sid, s in spans.items() if s["kind"] == side.upper() + "_ONLY"],
            ]
            fields[f"{side}_predicate_span_id"]["enum"] = [
                "",
                *[sid for sid, s in spans.items() if s["kind"] in ("SHARED", side.upper() + "_ONLY")],
            ]
    fields.update(
        binding_type={"type": "string", "enum": list(BINDINGS)},
        target_relation={"type": "string", "enum": list(RELATIONS)},
        explanation={"type": "string", "minLength": 1},
    )
    return {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}


def compile_review(value: dict, payload: dict) -> dict | None:
    require(
        isinstance(value, dict)
        and set(value) == set(response_schema()["required"])
        and all(isinstance(v, str) for v in value.values())
        and value["binding_type"] in BINDINGS
        and value["target_relation"] in RELATIONS
        and bool(value["explanation"].strip()),
        "SUBJECT_BINDING_SCHEMA",
        "exact source-comparison fields required",
    )
    spans = coverage_selection.span_inventory(payload)
    evidence = {}
    for side in ("a", "b"):
        evidence[side] = []
        for kind in ("subject", "predicate"):
            sid = value[f"{side}_{kind}_span_id"]
            if sid:
                quote = coverage_selection._selected_quote(sid, spans, side.upper(), unique=kind == "subject")
                witness = {"span_id": sid, "quote": quote}
                if witness not in evidence[side]:
                    evidence[side].append(witness)
    if value["target_relation"] != "DIFFERENT":
        return None
    require(
        value["binding_type"] != "NONE" and all(value[key] for key in value if key.endswith("span_id")),
        "SUBJECT_BINDING_BILATERAL",
        "a veto needs both own-unique subjects and both predicate witnesses",
    )
    basis = "ROLE_CONFLICT" if value["binding_type"] == "ACCESS_TARGET" else "IDENTITY_CONFLICT"
    proof = {
        "contract_version": veto.CONTRACT,
        "action": "REJECT_BOTH",
        "basis": basis,
        "a_evidence": evidence["a"],
        "b_evidence": evidence["b"],
        "shared_anchor_ids": [],
        "explanation": value["explanation"],
    }
    veto.validate_review(proof, payload)
    return proof


def apply_review(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    routing = route(main, payload)
    if routing != "REVIEW_BILATERAL_SUBJECTS":
        return deepcopy(main), routing
    proof = compile_review(value, payload)
    if proof is None:
        return deepcopy(main), "NO_SUPPORTED_SUBJECT_VETO_KEEP_COVERAGE"
    return veto.apply_review(main, payload, proof)


def messages(payload: dict, system: str) -> list[dict]:
    spans = coverage_selection.span_inventory(payload)
    blocks = []
    for side in ("A", "B"):
        ordered = []
        for sid, span in spans.items():
            if span["kind"] != "SHARED" and span.get("side") != side:
                continue
            prefix = side.lower() + "_" if span["kind"] == "SHARED" else ""
            start, end, text = (span[prefix + k] for k in ("start_char", "end_char", "text"))
            ordered.append((start, f"[{sid} {span['kind']} chars {start}:{end}] {text}"))
        blocks.append(f"DOCUMENT {side}\n" + "\n".join(text for _, text in sorted(ordered)))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(blocks)}]
