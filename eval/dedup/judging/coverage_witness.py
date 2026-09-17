# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Versioned, grounded retained-meaning coverage review; historical critics cannot supply it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from eval.dedup.core.validation import DedupEvaluationError, require
from eval.dedup.judging.local_ndd import _optional_v3_span_ledger, adapt_ndd_judge_output
from eval.dedup.judging.output_schema import JUDGE_SCHEMA, unresolved_judge_output, validate_judge_output
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.record_scope import complete_visible_equality
from eval.dedup.judging.scoped_critic import expand_citations

CONTRACT = "dedup-retained-coverage-v1"
COLUMN = "qwen_dedup_coverage_witness"
_REFERENCE_LIST = re.compile(r"[ABS]\d{3}(?:-[ABS]?\d{3})?(?:[ ,]+[ABS]\d{3}(?:-[ABS]?\d{3})?)*")
_MATERIAL = {
    "POLICY_MEANING": ("LEGAL_CONTEXT_CHANGE", "LEGAL_CONTEXT_COLLISION"),
    "RECORD_IDENTITY": ("DOCUMENT_IDENTITY_CHANGE", "TEMPLATE_SLOT_COLLISION"),
    "PAGE_ROLE": ("PAGE_ROLE_CHANGE", "PAGE_ROLE_COLLISION"),
    "STATE_VERSION": ("OTHER_MATERIAL", "IDENTIFIER_UNDERWEIGHTING"),
    "MEMBERSHIP": ("RESULT_SET_CHANGE", "LIST_SNAPSHOT_COLLISION"),
    "OTHER_RETAINED": ("OTHER_MATERIAL", "OTHER"),
}


def _text(description: str, *options: str) -> dict:
    return {
        "type": "string",
        "maxLength": 4096,
        "description": description,
        **({"enum": list(options)} if options else {}),
    }


def _object(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def coverage_schema() -> dict:
    side = _object(
        {
            "status": _text(
                "Coverage of THIS side's retained meaning by the opposite side.", "COVERED", "UNCOVERED", "UNRESOLVED"
            ),
            "reviewed_unique_ids": _text(
                "Every unique span on THIS side, e.g. A001-A006,A009. Empty iff none; no S IDs."
            ),
            "coverage_mode": _text(
                "How unique meanings are accounted for.",
                "SEMANTIC_COUNTERPARTS",
                "HARMLESS_ONLY",
                "MIXED",
                "NO_UNIQUE_SPANS",
                "NOT_COVERED",
            ),
            "coverage_counterpart_ids": _text(
                "Opposite-side or shared IDs supporting covered meanings. Empty for harmless-only/no-unique/unresolved."
            ),
            "coverage_explanation": _text(
                "Explain actual opposite-side entailment or why only harmless UI/repetition remains; citations alone do not establish it."
            ),
            "uncovered_type": _text(
                "Strongest retained uncovered meaning, not the lexical difference type.",
                "NONE",
                "MAIN_CONTENT",
                *_MATERIAL,
            ),
            "source_ids": _text(
                "Own-side unique IDs grounding the decisive uncovered proposition; optional own-side shared context. Empty without a witness."
            ),
            "source_quote": {
                **_text(
                    "Exact nonempty substring of a cited own-side unique span, at most 240 characters. Empty without a witness."
                ),
                "maxLength": 240,
            },
            "counterpart_ids": _text(
                "Closest opposite/shared IDs checked against that witness, or empty if genuinely no counterpart exists."
            ),
            "counterpart_quote": {
                **_text(
                    "Exact substring of a cited opposite/shared span, at most 240 characters; empty iff counterpart_ids empty."
                ),
                "maxLength": 240,
            },
            "counterpart_relation": _text(
                "No counterpart is not a conflicting value.", "NO_EQUIVALENT_FOUND", "CONTRADICTS", "NOT_APPLICABLE"
            ),
            "retention_consequence": _text(
                "What specific meaning would be lost, whose identity/function changes, or which value conflicts. Empty without a witness."
            ),
        }
    )
    return _object(
        {
            "contract_version": _text("Immutable response contract.", CONTRACT),
            "input_status": _text(
                "Untruncated complete input required for a resolved review.", "COMPLETE", "UNRESOLVED"
            ),
            "record_scope": _text(
                "Same owner/template does not bind a specific record.",
                "SAME_SUBSTANTIVE_RECORD",
                "NON_MAIN_MESSAGES",
                "DISTINCT_OR_UNBOUND_RECORDS",
                "IDENTICAL_TEXT",
                "UNRESOLVED",
            ),
            "anchor_a_ids": _text(
                "Independent A-side record/context anchor: A or S IDs, not the unique field that first introduces the record."
            ),
            "anchor_b_ids": _text("Independent B-side record/context anchor: B or S IDs."),
            "scope_explanation": _text(
                "Identify the actual represented record or message on BOTH sides and what the anchors establish."
            ),
            "a_meaning_in_b": side,
            "b_meaning_in_a": side,
        }
    )


def _shape(value: Any, schema: dict, field: str = "coverage") -> None:
    if schema["type"] == "object":
        require(
            isinstance(value, dict) and set(value) == set(schema["properties"]),
            "COVERAGE_CONTRACT_INVALID",
            "coverage fields differ; an old critic cannot substitute for the new contract",
            field=field,
        )
        for key, child in schema["properties"].items():
            _shape(value[key], child, f"{field}.{key}")
    else:
        require(
            isinstance(value, str)
            and ("enum" not in schema or value in schema["enum"])
            and len(value) <= schema.get("maxLength", 4096),
            "COVERAGE_CONTRACT_INVALID",
            "coverage field has an invalid type, enum or length",
            field=field,
        )


def _refs(value: str, spans: dict, side: str) -> list[str]:
    require(
        not value or _REFERENCE_LIST.fullmatch(value),
        "COVERAGE_REFERENCE_INVALID",
        "use explicit IDs or same-side ranges",
    )
    ids, issues = expand_citations(value, set(spans))
    require(not issues, "COVERAGE_REFERENCE_INVALID", "unknown or reversed span range", issues=issues)
    require(
        all(spans[s]["kind"] == "SHARED" or spans[s]["side"] == side for s in ids),
        "COVERAGE_REFERENCE_SIDE",
        "reference belongs to the wrong document",
        side=side,
    )
    return ids


def _span_evidence(span: dict, side: str, quote: str | None = None) -> dict:
    prefix = f"{side.lower()}_" if span["kind"] == "SHARED" else ""
    text, start = span[f"{prefix}text"], span[f"{prefix}start_char"]
    excerpt = text[:240] if quote is None else quote
    require(bool(excerpt) and excerpt in text, "COVERAGE_QUOTE_INVALID", "quote is not in its cited span", side=side)
    start += text.index(excerpt)
    return {"side": side, "start_char": start, "end_char": start + len(excerpt), "quote": excerpt}


def _quote_evidence(quote: str, ids: list[str], spans: dict, side: str, *, unique: bool) -> dict:
    for sid in ids:
        span = spans[sid]
        if unique and span["kind"] == "SHARED":
            continue
        prefix = f"{side.lower()}_" if span["kind"] == "SHARED" else ""
        if quote and quote in span[f"{prefix}text"]:
            return _span_evidence(span, side, quote)
    raise DedupEvaluationError(
        "COVERAGE_QUOTE_INVALID", "quote must occur in the independently cited source", side=side
    )


@dataclass(frozen=True)
class CoverageReview:
    value: dict
    evidence: list[dict]


def parse_coverage(value: Any, payload: dict) -> CoverageReview:
    """Validate provenance and logical consistency, not the model's semantic entailment claim."""
    _shape(value, coverage_schema())
    packet = payload.get("semantic_diff_evidence", {})
    complete = (
        packet.get("status") == "COMPLETE"
        and payload.get("long_document_evidence", {}).get("truncated") is False
        and all(isinstance(payload.get(f"document_{side}", {}).get("text"), str) for side in ("a", "b"))
    )
    require(
        value["input_status"] != "COMPLETE" or complete, "COVERAGE_INPUT_INCOMPLETE", "cannot resolve incomplete input"
    )
    require(bool(value["scope_explanation"].strip()), "COVERAGE_REASON_MISSING", "record scope needs an explanation")
    spans = {span["span_id"]: span for span in packet.get("spans", [])}
    require(len(spans) == len(packet.get("spans", [])), "COVERAGE_PACKET_INVALID", "duplicate span IDs")
    evidence, anchors = [], {}
    for side in ("A", "B"):
        anchors[side] = _refs(value[f"anchor_{side.lower()}_ids"], spans, side)
        if complete:
            for span in spans.values():
                if span["kind"] == "SHARED" or span.get("side") == side:
                    prefix = f"{side.lower()}_" if span["kind"] == "SHARED" else ""
                    text = payload[f"document_{side.lower()}"]["text"]
                    start, end = span[f"{prefix}start_char"], span[f"{prefix}end_char"]
                    require(
                        type(start) is int
                        and type(end) is int
                        and 0 <= start < end <= len(text)
                        and text[start:end] == span[f"{prefix}text"],
                        "COVERAGE_PACKET_INVALID",
                        "span must align exactly with original input",
                        side=side,
                    )
        evidence.extend(_span_evidence(spans[s], side) for s in anchors[side][:1])
    if value["input_status"] == "COMPLETE":
        require(
            bool(anchors["A"] and anchors["B"]), "COVERAGE_ANCHOR_MISSING", "each side needs its own context anchor"
        )
    for side, field in (("A", "a_meaning_in_b"), ("B", "b_meaning_in_a")):
        opposite = "B" if side == "A" else "A"
        item = value[field]
        reviewed = _refs(item["reviewed_unique_ids"], spans, side)
        unique = {sid for sid, span in spans.items() if span["kind"] != "SHARED" and span.get("side") == side}
        require(set(reviewed) <= unique, "COVERAGE_UNIQUE_IDS", "reviewed IDs must be this side's unique spans")
        require(
            bool(item["coverage_explanation"].strip()),
            "COVERAGE_REASON_MISSING",
            "each side needs a coverage explanation",
        )
        covered_refs = _refs(item["coverage_counterpart_ids"], spans, opposite)
        source_refs = _refs(item["source_ids"], spans, side)
        counterparts = _refs(item["counterpart_ids"], spans, opposite)
        if item["status"] != "UNRESOLVED":
            require(
                value["input_status"] == "COMPLETE" and set(reviewed) == unique,
                "COVERAGE_REVIEW_INCOMPLETE",
                "resolved coverage must account for every unique span",
            )
        if item["status"] == "UNCOVERED":
            require(
                item["coverage_mode"] == "NOT_COVERED"
                and item["uncovered_type"] != "NONE"
                and item["counterpart_relation"] != "NOT_APPLICABLE"
                and bool(item["retention_consequence"].strip()),
                "COVERAGE_WITNESS_MISSING",
                "uncovered meaning needs a typed, consequential source witness",
            )
            evidence.insert(0, _quote_evidence(item["source_quote"], source_refs, spans, side, unique=True))
            require(
                bool(counterparts) == bool(item["counterpart_quote"]),
                "COVERAGE_COUNTERPART_MISSING",
                "counterpart quote and references must be supplied together",
            )
            require(
                item["counterpart_relation"] != "CONTRADICTS" or bool(counterparts),
                "COVERAGE_COUNTERPART_MISSING",
                "contradiction requires an actual opposite-side value",
            )
            if counterparts:
                evidence.insert(
                    1, _quote_evidence(item["counterpart_quote"], counterparts, spans, opposite, unique=False)
                )
        else:
            require(
                item["uncovered_type"] == "NONE"
                and not any(
                    (
                        source_refs,
                        item["source_quote"],
                        counterparts,
                        item["counterpart_quote"],
                        item["retention_consequence"],
                    )
                )
                and item["counterpart_relation"] == "NOT_APPLICABLE",
                "COVERAGE_INACTIVE_WITNESS",
                "covered or unresolved review must not contain a material witness",
            )
            if item["status"] == "COVERED":
                modes = {"SEMANTIC_COUNTERPARTS", "HARMLESS_ONLY", "MIXED"} if unique else {"NO_UNIQUE_SPANS"}
                require(
                    item["coverage_mode"] in modes,
                    "COVERAGE_MODE_INVALID",
                    "coverage mode contradicts unique span membership",
                )
                require(
                    bool(covered_refs) == (item["coverage_mode"] in {"SEMANTIC_COUNTERPARTS", "MIXED"}),
                    "COVERAGE_COUNTERPART_MISSING",
                    "semantic coverage requires independently cited counterparts",
                )
            else:
                require(
                    item["coverage_mode"] == "NOT_COVERED" and not covered_refs,
                    "COVERAGE_MODE_INVALID",
                    "unresolved cannot claim coverage",
                )
    if value["input_status"] == "UNRESOLVED":
        require(
            value["record_scope"] == "UNRESOLVED"
            and not any(anchors.values())
            and all(value[field]["status"] == "UNRESOLVED" for field in ("a_meaning_in_b", "b_meaning_in_a")),
            "COVERAGE_UNRESOLVED_INCONSISTENT",
            "incomplete review cannot contain resolved scope or coverage",
        )
    if value["record_scope"] == "IDENTICAL_TEXT":
        require(
            complete_visible_equality(payload),
            "COVERAGE_FALSE_IDENTITY",
            "exact identity requires complete equal nonempty original text",
        )
    validate_evidence_offsets({"evidence": evidence}, payload)
    return CoverageReview(value, evidence)


def adapt_coverage_witness(main: dict, value: Any, payload: dict) -> dict:  # noqa: PLR0911
    """Use directional coverage witnesses without reopening negative or unknown main decisions."""
    review = parse_coverage(value, payload)
    if (
        payload.get("semantic_diff_evidence", {}).get("status") != "COMPLETE"
        or payload.get("long_document_evidence", {}).get("truncated") is not False
    ):
        return unresolved_judge_output()
    source = adapt_ndd_judge_output(main, JUDGE_SCHEMA, payload=payload, record_binding_policy="v6-route")
    if source["same_duplicate_group"] != "YES":
        return source
    if complete_visible_equality(payload):
        return validate_judge_output(
            {
                **source,
                "relation_type": "EXACT",
                "material_difference": "NONE",
                "primary_material_difference": "NONE",
                "evidence": [],
            }
        )
    ledger = _optional_v3_span_ledger(main, payload)
    require(
        ledger is not None and ledger[2] is None,
        "COVERAGE_MAIN_LEDGER_INVALID",
        "supported positive main ledger is required",
    )
    a, b = value["a_meaning_in_b"], value["b_meaning_in_a"]
    if (
        value["input_status"] == "UNRESOLVED"
        or value["record_scope"] == "UNRESOLVED"
        or "UNRESOLVED" in {a["status"], b["status"]}
    ):
        return unresolved_judge_output()
    non_main = ledger[0]["span_content_profile_a"] == ledger[0]["span_content_profile_b"] == "NON_MAIN_ONLY"
    if (value["record_scope"] == "NON_MAIN_MESSAGES" and not non_main) or (
        value["record_scope"] == "SAME_SUBSTANTIVE_RECORD" and non_main
    ):
        return unresolved_judge_output()
    uncovered = [item for item in (a, b) if item["status"] == "UNCOVERED"]
    if not uncovered:
        if value["record_scope"] == "DISTINCT_OR_UNBOUND_RECORDS":
            return unresolved_judge_output()
        chrome = any(item["coverage_mode"] in {"HARMLESS_ONLY", "MIXED"} for item in (a, b))
        target = {
            "same_duplicate_group": "YES",
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "NEAR_SURFACE",
            "material_difference": "MINOR" if chrome else "NONE",
            "primary_material_difference": "OTHER_MATERIAL" if chrome else "NONE",
            "primary_risk_factor": "NONE",
        }
        rule = "BILATERAL_COVERAGE"
    elif (
        not non_main
        and value["record_scope"] == "SAME_SUBSTANTIVE_RECORD"
        and len(uncovered) == 1
        and uncovered[0]["uncovered_type"] == "MAIN_CONTENT"
        and uncovered[0]["counterpart_relation"] == "NO_EQUIVALENT_FOUND"
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
        rule = "GROUNDED_SAME_RECORD_EXTENSION"
    else:
        kind = next(
            (item["uncovered_type"] for item in uncovered if item["uncovered_type"] in _MATERIAL), "OTHER_RETAINED"
        )
        primary, risk = _MATERIAL[kind]
        if value["record_scope"] == "DISTINCT_OR_UNBOUND_RECORDS":
            primary, risk = "DOCUMENT_IDENTITY_CHANGE", "TEMPLATE_SLOT_COLLISION"
        target = {
            "same_duplicate_group": "NO",
            "a_can_replace_b": "NO",
            "b_can_replace_a": "NO",
            "relation_type": "VERSION_RELATED" if kind == "STATE_VERSION" else "RELATED_NON_DUPLICATE",
            "material_difference": "MAJOR",
            "primary_material_difference": primary,
            "primary_risk_factor": risk,
        }
        rule = "UNCOVERED_RETAINED_MEANING"
    evidence = []
    for side in ("A", "B"):
        evidence.append(next(item for item in review.evidence if item["side"] == side))
    for item in review.evidence:
        if item not in evidence and len(evidence) < 4:
            evidence.append(item)
    public = {
        **source,
        **target,
        "confidence_tier": "MEDIUM",
        "evidence": evidence,
        "reason_codes": [
            f"COVERAGE_RULE:{rule}",
            f"COVERAGE_A_IN_B:{a['status']}",
            f"COVERAGE_B_IN_A:{b['status']}",
            f"COVERAGE_RECORD_SCOPE:{value['record_scope']}",
        ],
    }
    validate_judge_output(public)
    validate_evidence_offsets(public, payload)
    return public
