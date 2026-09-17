# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""V2 retained-coverage witnesses select original spans instead of regenerating quotes."""

from __future__ import annotations

import re
from copy import deepcopy

from eval.dedup.core.validation import require
from eval.dedup.judging import coverage_witness as original
from eval.dedup.judging.payload import validate_evidence_offsets

CONTRACT = "dedup-retained-coverage-v2"
COLUMN = original.COLUMN


def selection_schema() -> dict:
    schema = original.coverage_schema()
    schema["properties"]["contract_version"]["enum"] = [CONTRACT]
    schema["properties"]["record_scope"]["enum"].remove("IDENTICAL_TEXT")
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        properties = deepcopy(schema["properties"][field]["properties"])
        for key in ("source_ids", "source_quote", "counterpart_ids", "counterpart_quote"):
            properties.pop(key)
        properties["source_span_id"] = original._text(
            "ONE own unique span ID grounding the decisive loss; empty unless UNCOVERED. Never a list or a quote."
        )
        properties["counterpart_span_id"] = original._text(
            "ONE opposite/shared span ID actually checked, or empty if genuinely no counterpart exists."
        )
        schema["properties"][field] = original._object(properties)
    return schema


def span_inventory(payload: dict) -> dict[str, dict]:
    """Validate every selectable span, including uncited ones; never truncate or repair text."""
    packet = payload.get("semantic_diff_evidence", {})
    spans = packet.get("spans")
    require(isinstance(spans, list), "FOLLOWUP_BOUNDARY_SELECTION_PACKET", "span inventory missing")
    result = {}
    for span in spans:
        require(isinstance(span, dict), "FOLLOWUP_BOUNDARY_SELECTION_PACKET", "invalid span")
        sid, kind = span.get("span_id"), span.get("kind")
        require(
            isinstance(sid, str) and re.fullmatch(r"[ABS]\d{3}", sid) is not None and sid not in result,
            "FOLLOWUP_BOUNDARY_SELECTION_PACKET",
            "span IDs must be valid and unique",
        )
        sides = ("A", "B") if kind == "SHARED" else (span.get("side"),)
        require(
            (kind == "SHARED" and sid.startswith("S"))
            or (kind in {"A_ONLY", "B_ONLY"} and sides[0] == kind[0] == sid[0]),
            "FOLLOWUP_BOUNDARY_SELECTION_PACKET",
            "span kind and side disagree",
        )
        for side in sides:
            prefix = side.lower() + "_" if kind == "SHARED" else ""
            text = span.get(prefix + "text")
            start, end = span.get(prefix + "start_char"), span.get(prefix + "end_char")
            document = payload.get("document_" + side.lower(), {}).get("text")
            require(
                isinstance(text, str)
                and 0 < len(text) <= 240
                and isinstance(document, str)
                and type(start) is int
                and type(end) is int
                and 0 <= start < end <= len(document)
                and document[start:end] == text,
                "FOLLOWUP_BOUNDARY_SELECTION_PACKET",
                "complete selected spans must align and fit without truncation",
            )
        result[sid] = span
    return result


def _selected_quote(sid: str, spans: dict, side: str, *, unique: bool) -> str:
    if not sid:
        return ""
    require(sid in spans, "SELECTION_REFERENCE_INVALID", "select one existing span ID, not a list or range")
    span = spans[sid]
    require(
        (span["kind"] != "SHARED" and span["side"] == side) or (not unique and span["kind"] == "SHARED"),
        "SELECTION_REFERENCE_SIDE",
        "witness must select the required document and own-unique source",
    )
    evidence = original._span_evidence(span, side)
    return evidence["quote"]


def compile_selection(value: dict, payload: dict) -> dict:
    """Explicit V2 compilation is not a permissive parser for historical V1 model responses."""
    original._shape(value, selection_schema())
    spans = span_inventory(payload)
    compiled = deepcopy(value)
    compiled["contract_version"] = original.CONTRACT
    for side, field in (("A", "a_meaning_in_b"), ("B", "b_meaning_in_a")):
        item = compiled[field]
        source = item.pop("source_span_id")
        counterpart = item.pop("counterpart_span_id")
        item.update(
            source_ids=source,
            source_quote=_selected_quote(source, spans, side, unique=True),
            counterpart_ids=counterpart,
            counterpart_quote=_selected_quote(counterpart, spans, "B" if side == "A" else "A", unique=False),
        )
    original.parse_coverage(compiled, payload)
    return compiled


def adapt_selection(main: dict, value: dict, payload: dict) -> dict:
    public = original.adapt_coverage_witness(main, compile_selection(value, payload), payload)
    validate_evidence_offsets(public, payload)
    return public
