# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""V5 permits grounded shared-context contradictions without weakening content-loss witnesses."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eval.dedup.judging import composite_coverage as base
from eval.dedup.judging.coverage_selection import span_inventory
from eval.dedup.judging.coverage_witness import _MATERIAL, _refs, _span_evidence, _text
from eval.dedup.judging.payload_transport import validate_payload_transport
from eval.dedup.judging.schema_v3 import unresolved_judge_output_v3
from eval.dedup.validation import require, sha256_json

CONTRACT = "dedup-retained-coverage-v5"
ROUTING = "dedup-fresh-main-context-routing-v2"
SIDES = base.SIDES
owned_output = base.owned_output


def context_schema() -> dict:
    schema = base.composite_schema()
    schema["properties"]["contract_version"] = _text("Immutable contextual coverage contract.", CONTRACT)
    schema["$defs"]["UncoveredSide"]["properties"]["source_span_id"]["description"] = (
        "ONE own-unique loss ID. Exception: a SHARED ID may ground CONTRADICTS only when it matches this side's "
        "hard_conflict witness, its counterpart matches the opposite hard_conflict witness, and that opposite "
        "witness is an actual opposite-unique difference. Shared text alone never proves missing content."
    )
    schema["properties"]["translation_status"]["description"] = (
        "WHOLE-PAIR translation, not translation of just X. COMPLETE_FAITHFUL requires both sides COVERED and "
        "no actual conflict. Translated X plus retained Y on only one side MUST be OTHER, never COMPLETE_FAITHFUL."
    )
    return schema


def validate_context_shape(value: Any) -> None:
    from jsonschema import Draft202012Validator

    require(
        not list(Draft202012Validator(context_schema()).iter_errors(value)),
        "CONTEXT_CONTRACT_INVALID",
        "exact V5 fields, typed branches and enums required",
    )


def bind_context_response(value: Any, observed: Any) -> tuple[dict, dict]:
    validate_context_shape(value)
    fields = set().union(*(d["properties"] for d in context_schema()["$defs"].values()))
    require(
        isinstance(observed, dict) and observed.keys() == value.keys(),
        "FOLLOWUP_BOUNDARY_CONTEXT_RESPONSE",
        "root fields changed",
    )
    require(
        all(isinstance(observed[f], dict) and observed[f].keys() <= fields for f in SIDES.values()),
        "FOLLOWUP_BOUNDARY_CONTEXT_RESPONSE",
        "only known typed-branch null padding is permitted",
    )
    require(
        isinstance(observed["hard_conflict"], dict)
        and observed["hard_conflict"].keys() == value["hard_conflict"].keys(),
        "FOLLOWUP_BOUNDARY_CONTEXT_RESPONSE",
        "conflict fields changed",
    )
    changes = validate_payload_transport(value, observed)
    return deepcopy(value), {"contract": CONTRACT, "raw_sha256": sha256_json(value), "representation_changes": changes}


def _one(sid: str, spans: dict, side: str, *, optional: bool = False) -> list[str]:
    if optional and not sid:
        return []
    require(
        sid in spans and _refs(sid, spans, side) == [sid],
        "CONTEXT_REFERENCE",
        "one actual source on the specified side required",
    )
    return [sid]


def _shared_conflict_evidence(value: dict, payload: dict, spans: dict) -> list[dict]:
    """Validate the exceptional branch directly; never relabel a contradicted side COVERED."""
    require(
        payload["semantic_diff_evidence"]["status"] == "COMPLETE"
        and payload["long_document_evidence"]["truncated"] is False
        and value["input_status"] == "COMPLETE",
        "CONTEXT_INPUT_INCOMPLETE",
        "resolved loss needs complete input",
    )
    require(bool(value["basis_explanation"].strip()), "COVERAGE_REASON_MISSING", "shared basis needs an explanation")
    conflict = value["hard_conflict"]
    require(
        conflict["kind"] in _MATERIAL and bool(conflict["explanation"].strip()),
        "CONTEXT_CONFLICT_REQUIRED",
        "shared source requires an explicit actual conflict",
    )
    evidence = []
    for side in SIDES:
        sid = conflict[f"{side.lower()}_span_id"]
        _one(sid, spans, side)
        evidence.append(_span_evidence(spans[sid], side))
    require(
        any(spans[conflict[f"{s.lower()}_span_id"]]["kind"] != "SHARED" for s in SIDES),
        "CONTEXT_DIFFERENCE_REQUIRED",
        "two shared witnesses alone do not establish the differing value",
    )
    for side, field in SIDES.items():
        opposite = "B" if side == "A" else "A"
        anchors = _refs(value[f"anchor_{side.lower()}_ids"], spans, side)
        require(bool(anchors), "COVERAGE_ANCHOR_MISSING", "bilateral original context required")
        evidence.append(_span_evidence(spans[anchors[0]], side))
        item = value[field]
        unique = {sid for sid, span in spans.items() if span.get("side") == side and span["kind"] != "SHARED"}
        reviewed = set(_refs(item["reviewed_unique_ids"], spans, side))
        require(
            reviewed <= unique and bool(item["coverage_explanation"].strip()),
            "COVERAGE_REVIEW_INCOMPLETE",
            "valid reviewed inventory and explanation required",
        )
        require(
            item["status"] == "UNRESOLVED" or reviewed == unique,
            "COVERAGE_REVIEW_INCOMPLETE",
            "resolved sides must review all own-unique spans",
        )
        if item["status"] == "COVERED":
            harmless = set(_refs(item["harmless_unique_ids"], spans, side))
            support = _refs(item["opposite_support_ids"], spans, opposite)
            require(harmless <= unique, "TYPED_HARMLESS_IDS", "harmless IDs must be own unique")
            require(
                not (unique - harmless) or bool(support), "TYPED_COUNTERPART_MISSING", "retained meanings need support"
            )
        elif item["status"] == "UNRESOLVED":
            _refs(item["checked_opposite_ids"], spans, opposite)
        else:
            source, counterpart = item["source_span_id"], item["counterpart_span_id"]
            _one(source, spans, side)
            _one(counterpart, spans, opposite, optional=item["counterpart_relation"] != "CONTRADICTS")
            require(
                bool(item["retention_consequence"].strip()),
                "COVERAGE_WITNESS_MISSING",
                "concrete retained loss required",
            )
            if spans[source]["kind"] == "SHARED":
                require(
                    item["counterpart_relation"] == "CONTRADICTS"
                    and item["uncovered_type"] == conflict["kind"]
                    and source == conflict[f"{side.lower()}_span_id"]
                    and counterpart == conflict[f"{opposite.lower()}_span_id"]
                    and spans[counterpart]["kind"] != "SHARED",
                    "CONTEXT_SHARED_WITNESS",
                    "shared loss must match a paired actual conflict with a unique opposite difference",
                )
            evidence.append(_span_evidence(spans[source], side))
            if counterpart:
                evidence.append(_span_evidence(spans[counterpart], opposite))
    require(
        value["translation_status"] != "COMPLETE_FAITHFUL",
        "COMPOSITE_TRANSLATION",
        "whole-pair complete translation cannot have a contradiction or retained loss",
    )
    return evidence


def adapt_context(value: dict, payload: dict) -> dict:
    validate_context_shape(value)
    spans = span_inventory(payload)
    has_shared_loss = any(
        value[f]["status"] == "UNCOVERED" and spans.get(value[f]["source_span_id"], {}).get("kind") == "SHARED"
        for f in SIDES.values()
    )
    if not has_shared_loss:
        # V5 keeps the ordinary branch semantics exactly; only the explicitly
        # validated in-memory adapter view has a V4 tag, never the stored raw response.
        return base.adapt_composite({**deepcopy(value), "contract_version": base.CONTRACT}, payload)
    evidence = _shared_conflict_evidence(value, payload, spans)
    owned = owned_output(payload)
    if owned is not None:
        return owned
    if any(value[k] == "UNRESOLVED" for k in ("shared_basis", "translation_status", "dominant_overlap_source")) or any(
        value[f]["status"] == "UNRESOLVED" for f in SIDES.values()
    ):
        return unresolved_judge_output_v3()
    kind = value["hard_conflict"]["kind"]
    primary, risk = _MATERIAL[kind]
    return base._public(
        {
            "same_duplicate_group": "NO",
            "a_can_replace_b": "NO",
            "b_can_replace_a": "NO",
            "relation_type": "VERSION_RELATED" if kind == "STATE_VERSION" else "RELATED_NON_DUPLICATE",
            "material_difference": "MAJOR",
            "primary_material_difference": primary,
            "primary_risk_factor": risk,
        },
        overlap=value["dominant_overlap_source"],
        evidence=evidence,
        payload=payload,
        reasons=[
            "COMPOSITE_RULE:SHARED_CONTEXT_CONFLICT",
            f"CONTEXT_CONTRACT:{CONTRACT}",
            f"COMPOSITE_POLICY:{base.POLICY}",
            f"COMPOSITE_BASIS:{value['shared_basis']}",
            *[f"COVERAGE_{s}_IN_{'B' if s == 'A' else 'A'}:{value[f]['status']}" for s, f in SIDES.items()],
            f"COMPOSITE_CONFLICT:{kind}",
        ],
    )


def critic_route(main: dict, payload: dict) -> tuple[str, dict | None]:
    public = adapt_context(main, payload)
    if owned_output(payload) is not None:
        return "OWNED_INPUT", public
    if public["same_duplicate_group"] != "YES":
        return "MAIN_NOT_POSITIVE", public
    return "NEEDS_CONTEXT_CRITIC", None


def finalize_context(main: dict, critic: dict | None, payload: dict) -> dict:
    _, retained = critic_route(main, payload)
    if retained is not None:
        require(critic is None, "COMPOSITE_UNREQUESTED_CRITIC", "cannot fabricate an unrequested critic")
        return retained
    require(critic is not None, "COMPOSITE_CRITIC_MISSING", "fresh positive main requires critic")
    return adapt_context(critic, payload)
