# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Branch-scoped V0.6.2.11 review layered over the frozen V0.6.2.9 arbitration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from eval.dedup.core.validation import require
from eval.dedup.judging.boundary_critic import BOUNDARY_CRITIC_OPTIONS, _reject, _translation_target

LEGACY_BINDING_OPTIONS = {
    "ATOMIC_SAME_RECORD_EXTENSION",
    "BENIGN_NON_RECORD_DELTA",
    "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
    "TWO_SIDED_OR_CONFLICTING",
    "NON_MAIN_POLICY_OR_STATE_CHANGE",
    "NOT_APPLICABLE",
    "UNRESOLVED",
}
MATERIAL_SUBTYPES = {
    "POLICY_PROPOSITION_CHANGE": "LEGAL_CONTEXT_CHANGE",
    "CONSENT_OR_LEGAL_STATE_CHANGE": "LEGAL_CONTEXT_CHANGE",
    "COOKIE_INVENTORY_CHANGE": "RESULT_SET_CHANGE",
    "PAGE_CONTEXT_OR_IDENTITY_CHANGE": "DOCUMENT_IDENTITY_CHANGE",
}
_CITATION = re.compile(r"\b([ABS])(\d{3})(?:\s*(?:[-\u2013\u2014]|THROUGH|TO)\s*([ABS]?)(\d{3}))?\b")


def expand_citations(reasoning: str, known_ids: set[str]) -> tuple[list[str], tuple[str, ...]]:
    """Expand explicit same-side ranges only when every referenced ID exists."""
    citations, issues = [], []
    for match in _CITATION.finditer(reasoning.upper()):
        side, start_text, end_side, end_text = match.groups()
        start, end = int(start_text), int(end_text or start_text)
        if end_side not in {None, "", side} or end < start:
            issues.append("INVALID_SPAN_RANGE")
            continue
        ids = [f"{side}{index:03d}" for index in range(start, end + 1)]
        if not set(ids) <= known_ids:
            issues.append("UNKNOWN_SPAN")
            continue
        citations.extend(ids)
    return list(dict.fromkeys(citations)), tuple(dict.fromkeys(issues))


@dataclass(frozen=True)
class ScopedReview:
    scores: dict[str, str]
    citations: dict[str, list[str]]
    issues: dict[str, tuple[str, ...]]
    unique_ids: frozenset[str]

    @property
    def evidence_ids(self) -> list[str]:
        return list(
            dict.fromkeys(span for field, ids in self.citations.items() if not self.issues.get(field) for span in ids)
        )

    @property
    def legacy_critic(self) -> tuple[str, list[str], str | None]:
        score = self.scores["record_binding_verdict"]
        # V0.6.2.10 recorded outputs are accepted only by the offline replay path.
        score = {
            "CHROME_OR_REDUNDANCY_ONLY": "BENIGN_NON_RECORD_DELTA",
            "SEMANTIC_EQUIVALENCE": "BENIGN_NON_RECORD_DELTA",
        }.get(score, score)
        issues = (*self.issues.get("packet", ()), *self.issues.get("record_binding_verdict", ()))
        return score, self.citations["record_binding_verdict"], ";".join(issues) or None

    def covers_all_deltas(self, field: str) -> bool:
        return bool(self.unique_ids) and not self.issues.get(field) and self.unique_ids <= set(self.citations[field])


def parse_scoped_review(
    value: Any,
    payload: dict[str, Any] | None,
    *,
    translation_enabled: bool = True,
    legacy_binding: bool = True,
) -> ScopedReview:
    require(isinstance(value, dict), "LOCAL_NDD_OUTPUT_INVALID", "scoped critic is required")
    packet = payload.get("semantic_diff_evidence") if isinstance(payload, dict) else None
    packet = packet if isinstance(packet, dict) else {}
    spans = {s["span_id"]: s for s in packet.get("spans", []) if isinstance(s, dict) and s.get("span_id")}
    issues = {} if packet.get("status") == "COMPLETE" else {"packet": ("INCOMPLETE_PACKET",)}
    options = {
        "record_binding_verdict": LEGACY_BINDING_OPTIONS
        if legacy_binding
        else BOUNDARY_CRITIC_OPTIONS["record_binding_verdict"],
        "non_main_delta_subtype": BOUNDARY_CRITIC_OPTIONS["non_main_delta_subtype"],
    }
    if translation_enabled:
        options["translation_delta_direction"] = BOUNDARY_CRITIC_OPTIONS["translation_delta_direction"]
    scores, citations = {}, {}
    for field, choices in options.items():
        item = value.get(field)
        require(
            isinstance(item, dict)
            and isinstance(item.get("score"), str)
            and item["score"].upper() in choices
            and isinstance(item.get("reasoning"), str),
            "LOCAL_NDD_OUTPUT_INVALID",
            "scoped critic score or reasoning is invalid",
            field=field,
        )
        score = item["score"].upper()
        ids, syntax_issues = expand_citations(item["reasoning"], set(spans))
        scores[field], citations[field] = score, ids
        field_issues = list(syntax_issues)
        if score not in {"NOT_APPLICABLE", "NOT_TRANSLATION", "UNRESOLVED"}:
            kinds = {spans[s].get("kind") for s in ids}
            bilateral = "SHARED" in kinds or {"A_ONLY", "B_ONLY"} <= kinds
            if not bilateral:
                field_issues.append("MISSING_BILATERAL_BASIS")
            if field == "translation_delta_direction" and not {"A_ONLY", "B_ONLY"} <= kinds:
                field_issues.append("MISSING_TRANSLATION_SIDE")
            if (score in MATERIAL_SUBTYPES or score == "ATOMIC_SAME_RECORD_EXTENSION") and not kinds & {
                "A_ONLY",
                "B_ONLY",
            }:
                field_issues.append("MISSING_DELTA_SPAN")
        if field_issues:
            issues[field] = tuple(field_issues)
    return ScopedReview(
        scores,
        citations,
        issues,
        frozenset(s for s, span in spans.items() if span.get("kind") in {"A_ONLY", "B_ONLY"}),
    )


def arbitrate_scoped_review(  # noqa: PLR0911 - ordered ownership and evidence gates
    main: dict[str, str],
    ledger: dict[str, str],
    review: ScopedReview,
    *,
    legacy_target: dict[str, str],
    legacy_rule: str | None,
    unresolved: dict[str, str],
) -> tuple[dict[str, str], str]:
    if main.get("relation_type") == "UNRESOLVED":
        return main, "SCOPED_MAIN_UNRESOLVED"
    if "YES" not in {main.get("a_can_replace_b"), main.get("b_can_replace_a")}:
        return main, "SCOPED_VALID_MAIN_NEGATIVE_PRESERVED"
    if review.issues.get("packet"):
        return unresolved, "SCOPED_PACKET_INCOMPLETE"

    subtype = review.scores["non_main_delta_subtype"]
    if subtype in MATERIAL_SUBTYPES:
        if review.issues.get("non_main_delta_subtype"):
            return unresolved, "SCOPED_POLICY_EVIDENCE_INVALID"
        return _reject(MATERIAL_SUBTYPES[subtype]), f"SCOPED_POLICY_VETO_{subtype}"
    if subtype == "UNRESOLVED":
        return unresolved, "SCOPED_POLICY_UNRESOLVED"

    profiles = {ledger["span_content_profile_a"], ledger["span_content_profile_b"]}
    if profiles == {"NON_MAIN_ONLY"}:
        # The main independently established complete-message equivalence. An
        # inactive or redundant supplemental score is not its supporting proof.
        return main, "SCOPED_MAIN_NON_MAIN_EQUIVALENCE_PRESERVED"

    translation = review.scores.get("translation_delta_direction", "NOT_TRANSLATION")
    applicable = (
        profiles == {"SUBSTANTIVE_MAIN"}
        and ledger["span_shared_basis"] == "VERIFIED_SUBSTANTIVE_RECORD"
        and ledger["span_translation_status"] in {"COMPLETE_FAITHFUL", "PARTIAL_OR_ADDITIVE"}
    )
    if applicable and translation != "NOT_TRANSLATION":
        if translation == "UNRESOLVED" or review.issues.get("translation_delta_direction"):
            return unresolved, "SCOPED_TRANSLATION_EVIDENCE_INVALID"
        if translation == "CONFLICTING":
            return _reject("OTHER_MATERIAL"), "SCOPED_TRANSLATION_CONFLICT"
        if main["relation_type"] == "CONTAINMENT":
            expected = "A_ADDS" if main["a_can_replace_b"] == "YES" else "B_ADDS"
            if translation == "EQUIVALENT":
                return {**main, "confidence_tier": "LOW"}, "SCOPED_CITED_TRANSLATION_ADDITION_PRESERVED"
            if translation != expected:
                return unresolved, "SCOPED_TRANSLATION_DIRECTION_DISAGREEMENT"
        elif (
            "YES" not in {legacy_target.get("a_can_replace_b"), legacy_target.get("b_can_replace_a")}
            and legacy_target.get("relation_type") != "UNRESOLVED"
        ):
            return legacy_target, legacy_rule or "SCOPED_RECORD_VETO_PRESERVED"
        return _translation_target(translation), f"SCOPED_TRANSLATION_{translation}"

    binding = review.scores["record_binding_verdict"]
    exhaustive_chrome = (
        binding == "CHROME_OR_REDUNDANCY_ONLY" and review.covers_all_deltas("record_binding_verdict")
    ) or (
        binding == "BENIGN_NON_RECORD_DELTA"
        and subtype == "EQUIVALENT_MESSAGE_OR_WRAPPER"
        and not review.issues.get("record_binding_verdict")
        and review.covers_all_deltas("non_main_delta_subtype")
    )
    if main["relation_type"] == "CONTAINMENT" and not applicable and exhaustive_chrome:
        return {
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "NEAR_SURFACE",
            "material_difference": "MINOR",
            "primary_material_difference": "OTHER_MATERIAL",
            "confidence_tier": "MEDIUM",
        }, "SCOPED_EXHAUSTIVE_CHROME_EQUIVALENCE"
    return legacy_target, legacy_rule or "SCOPED_LEGACY_RECORD_REVIEW"
