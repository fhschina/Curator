# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Evidence-scoped arbitration for the immutable V0.6.2.10 boundary critic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from eval.dedup.validation import require

BOUNDARY_CRITIC_OPTIONS = {
    "non_main_delta_subtype": {
        "NOT_APPLICABLE",
        "EQUIVALENT_MESSAGE_OR_WRAPPER",
        "POLICY_PROPOSITION_CHANGE",
        "CONSENT_OR_LEGAL_STATE_CHANGE",
        "COOKIE_INVENTORY_CHANGE",
        "PAGE_CONTEXT_OR_IDENTITY_CHANGE",
        "UNRESOLVED",
    },
    "translation_delta_direction": {"NOT_TRANSLATION", "EQUIVALENT", "A_ADDS", "B_ADDS", "CONFLICTING", "UNRESOLVED"},
    "record_binding_verdict": {
        "ATOMIC_SAME_RECORD_EXTENSION",
        "CHROME_OR_REDUNDANCY_ONLY",
        "SEMANTIC_EQUIVALENCE",
        "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
        "TWO_SIDED_OR_CONFLICTING",
        "NOT_APPLICABLE",
        "UNRESOLVED",
    },
}
_SPAN_ID = re.compile(r"\b[ABS]\d{3}\b")
_MATERIAL_NON_MAIN = {
    "POLICY_PROPOSITION_CHANGE": "LEGAL_CONTEXT_CHANGE",
    "CONSENT_OR_LEGAL_STATE_CHANGE": "LEGAL_CONTEXT_CHANGE",
    "COOKIE_INVENTORY_CHANGE": "RESULT_SET_CHANGE",
    "PAGE_CONTEXT_OR_IDENTITY_CHANGE": "DOCUMENT_IDENTITY_CHANGE",
}


@dataclass(frozen=True)
class BoundaryReview:
    scores: dict[str, str]
    citations: dict[str, list[str]]
    issues: tuple[str, ...]

    @property
    def evidence_ids(self) -> list[str]:
        # Decisive specialized evidence precedes the generic record verdict.
        return list(dict.fromkeys(span_id for ids in self.citations.values() for span_id in ids))


def parse_boundary_review(value: Any, payload: dict[str, Any] | None) -> BoundaryReview:
    require(isinstance(value, dict), "LOCAL_NDD_OUTPUT_INVALID", "boundary critic is required")
    scores, citations, issues = {}, {}, []
    packet = payload.get("semantic_diff_evidence", {}) if isinstance(payload, dict) else {}
    spans = {span["span_id"]: span for span in packet.get("spans", [])}
    if packet.get("status") != "COMPLETE":
        issues.append("BOUNDARY_PACKET_INCOMPLETE")
    for field, options in BOUNDARY_CRITIC_OPTIONS.items():
        item = value.get(field)
        require(
            isinstance(item, dict)
            and isinstance(item.get("score"), str)
            and item["score"].upper() in options
            and isinstance(item.get("reasoning"), str),
            "LOCAL_NDD_OUTPUT_INVALID",
            "boundary critic score or reasoning is missing or invalid",
            field=field,
        )
        score = item["score"].upper()
        ids = list(dict.fromkeys(_SPAN_ID.findall(item["reasoning"].upper())))
        scores[field], citations[field] = score, ids
        if any(span_id not in spans for span_id in ids):
            issues.append(f"{field}:UNKNOWN_SPAN")
            continue
        if score in {"NOT_APPLICABLE", "NOT_TRANSLATION", "UNRESOLVED"}:
            continue
        has_a = any(spans[s]["kind"] == "A_ONLY" for s in ids)
        has_b = any(spans[s]["kind"] == "B_ONLY" for s in ids)
        bilateral = (has_a and has_b) or any(spans[s]["kind"] == "SHARED" for s in ids)
        if not bilateral:
            issues.append(f"{field}:MISSING_BILATERAL_BASIS")
        if field == "translation_delta_direction" and not (has_a and has_b):
            issues.append(f"{field}:MISSING_TRANSLATION_SIDE")
        if (score in _MATERIAL_NON_MAIN or score == "ATOMIC_SAME_RECORD_EXTENSION") and not (has_a or has_b):
            issues.append(f"{field}:MISSING_DELTA_SPAN")
        if score == "CHROME_OR_REDUNDANCY_ONLY":
            unique_ids = {s for s, span in spans.items() if span["kind"] in {"A_ONLY", "B_ONLY"}}
            if not unique_ids or not unique_ids <= set(ids):
                issues.append(f"{field}:CHROME_COVERAGE_INCOMPLETE")
    return BoundaryReview(scores, citations, tuple(issues))


def _reject(primary: str) -> dict[str, str]:
    return {
        "a_can_replace_b": "NO",
        "b_can_replace_a": "NO",
        "relation_type": "RELATED_NON_DUPLICATE",
        "material_difference": "MAJOR",
        "primary_material_difference": primary,
    }


def _translation_target(direction: str) -> dict[str, str]:
    equivalent = direction == "EQUIVALENT"
    return {
        "a_can_replace_b": "YES" if direction in {"EQUIVALENT", "A_ADDS"} else "NO",
        "b_can_replace_a": "YES" if direction in {"EQUIVALENT", "B_ADDS"} else "NO",
        "relation_type": "NEAR_SURFACE" if equivalent else "CONTAINMENT",
        "material_difference": "NONE" if equivalent else "MAJOR",
        "primary_material_difference": "NONE" if equivalent else "MAIN_CONTENT_ADDITION_DELETION",
        "primary_risk_factor": "TRANSLATION_EQUIVALENCE" if equivalent else "CONTAINMENT_ASYMMETRY",
        "dominant_overlap_source": "MAIN_CONTENT",
        "confidence_tier": "MEDIUM",
    }


def arbitrate_boundary_review(  # noqa: PLR0911 - ordered semantic gates
    main: dict[str, str],
    ledger: dict[str, str],
    review: BoundaryReview,
    unresolved: dict[str, str],
) -> tuple[dict[str, str], str]:
    """Reconcile specialized evidence without rescuing invalid or conflicting main ledgers."""

    if main.get("relation_type") == "UNRESOLVED":
        return main, "BOUNDARY_MAIN_UNRESOLVED"
    if ledger["span_hard_conflict"] != "NONE":
        return main, "BOUNDARY_MAIN_HARD_CONFLICT"
    if review.issues:
        return unresolved, "BOUNDARY_INVALID_CITATIONS"

    non_main = review.scores["non_main_delta_subtype"]
    translation = review.scores["translation_delta_direction"]
    binding = review.scores["record_binding_verdict"]
    if non_main in _MATERIAL_NON_MAIN:
        return _reject(_MATERIAL_NON_MAIN[non_main]), f"BOUNDARY_VETO_{non_main}"
    if non_main == "UNRESOLVED":
        return unresolved, "BOUNDARY_NON_MAIN_UNRESOLVED"

    profiles = {ledger["span_content_profile_a"], ledger["span_content_profile_b"]}
    if profiles == {"NON_MAIN_ONLY"}:
        if non_main != "EQUIVALENT_MESSAGE_OR_WRAPPER":
            return unresolved, "BOUNDARY_NON_MAIN_SCOPE_DISAGREEMENT"
        return main, "BOUNDARY_NON_MAIN_MESSAGE_REVIEWED"

    if "YES" not in {main.get("a_can_replace_b"), main.get("b_can_replace_a")}:
        return main, "BOUNDARY_MAIN_NEGATIVE_PRESERVED"
    if translation == "CONFLICTING":
        return _reject("OTHER_MATERIAL"), "BOUNDARY_TRANSLATION_CONFLICT"
    if translation == "UNRESOLVED":
        return unresolved, "BOUNDARY_TRANSLATION_UNRESOLVED"
    if translation != "NOT_TRANSLATION":
        eligible = (
            profiles == {"SUBSTANTIVE_MAIN"}
            and ledger["span_shared_basis"] == "VERIFIED_SUBSTANTIVE_RECORD"
            and ledger["span_translation_status"] in {"COMPLETE_FAITHFUL", "PARTIAL_OR_ADDITIVE"}
            and binding in {"ATOMIC_SAME_RECORD_EXTENSION", "SEMANTIC_EQUIVALENCE"}
            and {ledger["span_a_delta"], ledger["span_b_delta"]}
            <= {"NONE", "SEMANTICALLY_COVERED", "UNIVERSAL_UI_OR_REPETITION", "SAME_RECORD_CONTENT_EXTENSION"}
        )
        if not eligible:
            return unresolved, "BOUNDARY_TRANSLATION_SCOPE_DISAGREEMENT"
        if main.get("relation_type") == "CONTAINMENT":
            if translation == "EQUIVALENT":
                return {**main, "confidence_tier": "LOW"}, "BOUNDARY_PRESERVED_CITED_TRANSLATION_ADDITION"
            expected = "A_ADDS" if main["a_can_replace_b"] == "YES" else "B_ADDS"
            if expected != translation:
                return unresolved, "BOUNDARY_TRANSLATION_DIRECTION_DISAGREEMENT"
        return _translation_target(translation), f"BOUNDARY_TRANSLATION_{translation}"

    if binding == "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT":
        return _reject("PAGE_ROLE_CHANGE"), "BOUNDARY_SEPARATE_RECORD_VETO"
    if binding == "TWO_SIDED_OR_CONFLICTING":
        return _reject("OTHER_MATERIAL"), "BOUNDARY_TWO_SIDED_VETO"
    if binding == "UNRESOLVED":
        return unresolved, "BOUNDARY_BINDING_UNRESOLVED"
    if main.get("relation_type") != "CONTAINMENT":
        return main, "BOUNDARY_MAIN_EQUIVALENCE_REVIEWED"
    if binding == "CHROME_OR_REDUNDANCY_ONLY":
        return {
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "NEAR_SURFACE",
            "material_difference": "MINOR",
            "primary_material_difference": "OTHER_MATERIAL",
            "confidence_tier": "MEDIUM",
        }, "BOUNDARY_EXHAUSTIVE_CHROME_EQUIVALENCE"
    if binding == "SEMANTIC_EQUIVALENCE":
        return {**main, "confidence_tier": "LOW"}, "BOUNDARY_PRESERVED_CITED_ADDITION"
    if binding == "ATOMIC_SAME_RECORD_EXTENSION":
        side = "A" if main["a_can_replace_b"] == "YES" else "B"
        if any(s.startswith(side) for s in review.citations["record_binding_verdict"]):
            return main, "BOUNDARY_ATOMIC_EXTENSION_CONFIRMED"
        return unresolved, "BOUNDARY_EXTENSION_SIDE_MISSING"
    return unresolved, "BOUNDARY_CONTAINMENT_NOT_CONFIRMED"
