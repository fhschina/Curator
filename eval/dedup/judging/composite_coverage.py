# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Grounded composite-document coverage, separate from historical record-scope contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eval.dedup.judging.coverage_selection import compile_selection, span_inventory
from eval.dedup.judging.coverage_witness import _MATERIAL, _object, _refs, _span_evidence, _text, parse_coverage
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.payload_transport import validate_payload_transport
from eval.dedup.judging.record_scope import complete_visible_equality
from eval.dedup.judging.schema_v2 import DOMINANT_OVERLAP_SOURCES
from eval.dedup.judging.schema_v3 import unresolved_judge_output_v3, validate_judge_output_v3
from eval.dedup.judging.typed_coverage import CONTRACT as TYPED_CONTRACT
from eval.dedup.judging.typed_coverage import compile_typed_coverage, typed_coverage_schema
from eval.dedup.validation import require, sha256_json

CONTRACT = "dedup-retained-coverage-v4"
POLICY = "dedup-composite-containment-policy-v1"
ROUTING = "dedup-fresh-main-composite-routing-v1"
SIDES = {"A": "a_meaning_in_b", "B": "b_meaning_in_a"}


def composite_schema() -> dict:
    old = typed_coverage_schema()
    props = {k: deepcopy(old["properties"][k]) for k in ("contract_version", "input_status", *SIDES.values())}
    props["contract_version"] = _text(
        "Immutable composite coverage contract; historical responses are invalid.", CONTRACT
    )
    for side in ("a", "b"):
        props[f"anchor_{side}_ids"] = _text(
            f"{side.upper()} or S IDs grounding shared substantive X, or the actual message/context if there is no X. "
            "Identify X, not the newly added independent Y. Anchors need not be lexically identical."
        )
    props.update(
        basis_explanation=_text(
            "Identify the retained X on BOTH sides, or explain why overlap is only non-main/template."
        ),
        shared_basis=_text(
            "Shared substantive X permits independent Y additions; a common owner/template alone does not establish X.",
            "SHARED_SUBSTANTIVE_CONTENT",
            "NON_MAIN_MESSAGES",
            "NO_SHARED_SUBSTANTIVE_CONTENT",
            "UNRESOLVED",
        ),
        hard_conflict=_object(
            {
                "kind": _text(
                    "Actual incompatible retained values/functions, not merely an additional independent record.",
                    "NONE",
                    *_MATERIAL,
                    "UNRESOLVED",
                ),
                "a_span_id": _text(
                    "ONE A/shared span containing the actual conflicting value; empty for NONE/UNRESOLVED."
                ),
                "b_span_id": _text(
                    "ONE B/shared span containing the actual conflicting value; empty for NONE/UNRESOLVED."
                ),
                "explanation": _text(
                    "State the incompatible identities, states, functions, membership or permission; absence is not contradiction."
                ),
            }
        ),
        translation_status=_text(
            "Complete faithful translation requires bilateral coverage without a conflict.",
            "COMPLETE_FAITHFUL",
            "OTHER",
            "UNRESOLVED",
        ),
        dominant_overlap_source=_text(
            "Describe where most visible overlap comes from, separately from whether substantive X exists.",
            *DOMINANT_OVERLAP_SOURCES,
        ),
    )
    return {**_object(props), "$defs": deepcopy(old["$defs"])}


def validate_composite_shape(value: Any) -> None:
    from jsonschema import Draft202012Validator

    require(
        not list(Draft202012Validator(composite_schema()).iter_errors(value)),
        "COMPOSITE_CONTRACT_INVALID",
        "exact V4 fields, branch membership, types and enums required",
    )


def bind_composite_response(value: Any, observed: Any) -> tuple[dict, dict]:
    validate_composite_shape(value)
    schema = composite_schema()
    side_fields = set().union(*(d["properties"] for d in schema["$defs"].values()))
    require(
        isinstance(observed, dict) and observed.keys() == value.keys(),
        "FOLLOWUP_BOUNDARY_COMPOSITE_RESPONSE",
        "root fields changed",
    )
    for field in SIDES.values():
        require(
            isinstance(observed[field], dict) and observed[field].keys() <= side_fields,
            "FOLLOWUP_BOUNDARY_COMPOSITE_RESPONSE",
            "only known side branch fields may acquire null padding",
        )
    require(
        isinstance(observed["hard_conflict"], dict)
        and observed["hard_conflict"].keys() == value["hard_conflict"].keys(),
        "FOLLOWUP_BOUNDARY_COMPOSITE_RESPONSE",
        "conflict proof fields changed",
    )
    changes = validate_payload_transport(value, observed)
    return deepcopy(value), {"contract": CONTRACT, "raw_sha256": sha256_json(value), "representation_changes": changes}


def validate_composite(value: dict, payload: dict) -> tuple[dict, list[dict]]:
    """Validate citations/consistency; semantic truth still requires evaluation."""
    validate_composite_shape(value)
    spans = span_inventory(payload)
    # Reuse the frozen typed-side compiler solely for inventory/coverage proofs.
    # UNRESOLVED is a neutral legacy scope placeholder, never a new semantic verdict.
    proof = {
        "contract_version": TYPED_CONTRACT,
        "input_status": value["input_status"],
        "record_scope": "UNRESOLVED",
        "scope_explanation": value["basis_explanation"],
        **{k: deepcopy(value[k]) for k in ("anchor_a_ids", "anchor_b_ids", *SIDES.values())},
    }
    compiled = compile_typed_coverage(proof, payload)
    evidence = parse_coverage(compile_selection(compiled.selection, payload), payload).evidence
    conflict = value["hard_conflict"]
    require(
        bool(conflict["explanation"].strip()), "COMPOSITE_CONFLICT_REASON", "conflict assessment needs a rationale"
    )
    uncovered = [v for k, v in value.items() if k in SIDES.values() and v["status"] == "UNCOVERED"]
    if conflict["kind"] in {"NONE", "UNRESOLVED"}:
        require(
            not conflict["a_span_id"] and not conflict["b_span_id"],
            "COMPOSITE_INACTIVE_CONFLICT",
            "no inactive conflict witnesses",
        )
        require(
            not any(i["counterpart_relation"] == "CONTRADICTS" for i in uncovered),
            "COMPOSITE_CONFLICT_MISSING",
            "a contradiction needs an explicit paired conflict proof",
        )
    else:
        for side in SIDES:
            sid = conflict[f"{side.lower()}_span_id"]
            require(
                sid in spans and _refs(sid, spans, side) == [sid],
                "COMPOSITE_CONFLICT_REFERENCE",
                "one actual source on each side required",
            )
        require(
            any(
                value[field]["status"] == "UNCOVERED"
                and value[field]["uncovered_type"] == conflict["kind"]
                and value[field]["counterpart_relation"] == "CONTRADICTS"
                and value[field]["source_span_id"] == conflict[f"{side.lower()}_span_id"]
                and value[field]["counterpart_span_id"] == conflict[f"{'b' if side == 'A' else 'a'}_span_id"]
                for side, field in SIDES.items()
            ),
            "COMPOSITE_CONFLICT_COVERAGE",
            "paired conflict must match a concrete directional loss witness",
        )
        evidence = [_span_evidence(spans[conflict[f"{s.lower()}_span_id"]], s) for s in SIDES] + evidence
    if value["input_status"] == "UNRESOLVED":
        require(
            value["shared_basis"] == "UNRESOLVED"
            and conflict["kind"] == "UNRESOLVED"
            and value["translation_status"] == value["dominant_overlap_source"] == "UNRESOLVED",
            "COMPOSITE_INCOMPLETE",
            "an incomplete review cannot assert resolved semantic fields",
        )
    if value["translation_status"] == "COMPLETE_FAITHFUL":
        require(
            value["input_status"] == "COMPLETE"
            and value["shared_basis"] in {"SHARED_SUBSTANTIVE_CONTENT", "NON_MAIN_MESSAGES"}
            and all(value[f]["status"] == "COVERED" for f in SIDES.values())
            and conflict["kind"] == "NONE",
            "COMPOSITE_TRANSLATION",
            "complete translation cannot have a retained loss or unresolved/conflicting basis",
        )
    return compiled.selection, evidence


def _public(target: dict, *, overlap: str, evidence: list[dict], reasons: list[str], payload: dict) -> dict:
    selected = [next(e for e in evidence if e["side"] == s) for s in SIDES] if evidence else []
    for item in evidence:
        if item not in selected and len(selected) < 4:
            selected.append(item)
    result = {
        **unresolved_judge_output_v3(),
        **target,
        "dominant_overlap_source": overlap,
        "confidence_tier": "MEDIUM",
        "reason_codes": reasons,
        "evidence": selected,
    }
    validate_judge_output_v3(result)
    validate_evidence_offsets(result, payload)
    return result


def owned_output(payload: dict) -> dict | None:
    if (
        payload.get("long_document_evidence", {}).get("truncated") is not False
        or payload.get("semantic_diff_evidence", {}).get("status") != "COMPLETE"
        or not all(
            isinstance(payload.get(f"document_{s}", {}).get("text"), str) and payload[f"document_{s}"]["text"].strip()
            for s in ("a", "b")
        )
    ):
        return unresolved_judge_output_v3()
    if complete_visible_equality(payload):
        return _public(
            {
                "same_duplicate_group": "YES",
                "a_can_replace_b": "YES",
                "b_can_replace_a": "YES",
                "relation_type": "EXACT",
                "material_difference": "NONE",
                "primary_material_difference": "NONE",
                "primary_risk_factor": "NONE",
            },
            overlap="LOCAL_PASSAGE",
            evidence=[],
            reasons=["COMPOSITE_RULE:COMPLETE_VISIBLE_EQUALITY", "COMPOSITE_TEXT_ROLE:NOT_INFERRED"],
            payload=payload,
        )
    return None


def adapt_composite(value: dict, payload: dict) -> dict:
    compiled, evidence = validate_composite(value, payload)
    owned = owned_output(payload)
    if owned is not None:
        return owned
    a, b = (value[f] for f in SIDES.values())
    basis, conflict = value["shared_basis"], value["hard_conflict"]["kind"]
    if (
        value["input_status"] == "UNRESOLVED"
        or basis == "UNRESOLVED"
        or conflict == "UNRESOLVED"
        or value["dominant_overlap_source"] == "UNRESOLVED"
        or value["translation_status"] == "UNRESOLVED"
        or "UNRESOLVED" in {a["status"], b["status"]}
    ):
        return unresolved_judge_output_v3()
    losses = [i for i in (a, b) if i["status"] == "UNCOVERED"]
    if not losses and basis == "NO_SHARED_SUBSTANTIVE_CONTENT":
        return unresolved_judge_output_v3()
    if not losses:
        chrome = any(compiled[f]["coverage_mode"] in {"HARMLESS_ONLY", "MIXED"} for f in SIDES.values())
        minor = chrome and value["translation_status"] != "COMPLETE_FAITHFUL"
        target = {
            "same_duplicate_group": "YES",
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "NEAR_SURFACE",
            "material_difference": "MINOR" if minor else "NONE",
            "primary_material_difference": "OTHER_MATERIAL" if minor else "NONE",
            "primary_risk_factor": "TRANSLATION_EQUIVALENCE"
            if value["translation_status"] == "COMPLETE_FAITHFUL"
            else "NONE",
        }
        rule = "BILATERAL_COVERAGE"
    elif (
        basis == "SHARED_SUBSTANTIVE_CONTENT"
        and conflict == "NONE"
        and len(losses) == 1
        and losses[0]["uncovered_type"] == "MAIN_CONTENT"
        and losses[0]["counterpart_relation"] == "NO_EQUIVALENT_FOUND"
    ):
        target = {
            "same_duplicate_group": "YES",
            "a_can_replace_b": "YES" if b["status"] == "COVERED" else "NO",
            "b_can_replace_a": "YES" if a["status"] == "COVERED" else "NO",
            "relation_type": "CONTAINMENT",
            "material_difference": "MAJOR",
            "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
            "primary_risk_factor": "CONTAINMENT_ASYMMETRY",
        }
        rule = "SHARED_X_WITH_ONE_SIDED_CONTENT"
    else:
        kind = (
            conflict
            if conflict != "NONE"
            else next((i["uncovered_type"] for i in losses if i["uncovered_type"] in _MATERIAL), "MAIN_CONTENT")
        )
        primary, risk = _MATERIAL.get(kind, ("MAIN_CONTENT_ADDITION_DELETION", "OTHER"))
        if basis == "NO_SHARED_SUBSTANTIVE_CONTENT" and conflict == "NONE":
            risk = (
                "BOILERPLATE_DOMINATED_SIMILARITY"
                if value["dominant_overlap_source"] not in {"NONE", "MAIN_CONTENT"}
                else "OTHER"
            )
        target = {
            "same_duplicate_group": "NO",
            "a_can_replace_b": "NO",
            "b_can_replace_a": "NO",
            "relation_type": "VERSION_RELATED" if kind == "STATE_VERSION" else "RELATED_NON_DUPLICATE",
            "material_difference": "MAJOR",
            "primary_material_difference": primary,
            "primary_risk_factor": risk,
        }
        rule = "ACTUAL_CONFLICT" if conflict != "NONE" else "NO_SAFE_COMPOSITE_REPLACEMENT"
    return _public(
        target,
        overlap=value["dominant_overlap_source"],
        evidence=evidence,
        payload=payload,
        reasons=[
            f"COMPOSITE_RULE:{rule}",
            f"COMPOSITE_POLICY:{POLICY}",
            f"COMPOSITE_BASIS:{basis}",
            f"COVERAGE_A_IN_B:{a['status']}",
            f"COVERAGE_B_IN_A:{b['status']}",
            f"COMPOSITE_CONFLICT:{conflict}",
        ],
    )


def critic_route(main: dict, payload: dict) -> tuple[str, dict | None]:
    """Fresh V4 main only; historical NO responses cannot silently bypass rejudging."""
    public = adapt_composite(main, payload)
    owned = owned_output(payload)
    if owned is not None:
        return "OWNED_INPUT", owned
    if public["same_duplicate_group"] != "YES":
        return "MAIN_NOT_POSITIVE", public
    return "NEEDS_COMPOSITE_CRITIC", None


def finalize_composite(main: dict, critic: dict | None, payload: dict) -> dict:
    _, owned = critic_route(main, payload)
    if owned is not None:
        require(
            critic is None, "COMPOSITE_UNREQUESTED_CRITIC", "owned main result must not fabricate a critic response"
        )
        return owned
    require(critic is not None, "COMPOSITE_CRITIC_MISSING", "positive main requires its fresh V4 audit")
    return adapt_composite(critic, payload)
