# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""V0.6.2.12 evidence-scoped exceptions to non-main message equivalence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.dedup.core.validation import require
from eval.dedup.judging.boundary_critic import _reject
from eval.dedup.judging.scoped_critic import expand_citations

MATERIAL_CONFLICTS = {
    "IDENTITY_OR_SERVICE_TARGET_CHANGE": ("DOCUMENT_IDENTITY_CHANGE", "TEMPLATE_SLOT_COLLISION"),
    "PAGE_ROLE_CHANGE": ("PAGE_ROLE_CHANGE", "PAGE_ROLE_COLLISION"),
    "POLICY_PERMISSION_CHANGE": ("LEGAL_CONTEXT_CHANGE", "LEGAL_CONTEXT_COLLISION"),
    "STATE_CHANGE": ("OTHER_MATERIAL", "IDENTIFIER_UNDERWEIGHTING"),
    "MEMBERSHIP_CHANGE": ("RESULT_SET_CHANGE", "LIST_SNAPSHOT_COLLISION"),
}
RETAINED_CONFLICT_OPTIONS = frozenset({"NONE", "NOT_APPLICABLE", "UNRESOLVED", *MATERIAL_CONFLICTS})
_VETO_VERDICTS = {
    "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
    "TWO_SIDED_OR_CONFLICTING",
    "NON_MAIN_POLICY_OR_STATE_CHANGE",
}


@dataclass(frozen=True)
class RetainedConflict:
    score: str
    citations: list[str]
    issues: tuple[str, ...]


def parse_retained_conflict(value: Any, payload: dict[str, Any] | None) -> RetainedConflict:
    item = value.get("retained_conflict") if isinstance(value, dict) else None
    require(
        isinstance(item, dict)
        and isinstance(item.get("score"), str)
        and item["score"].upper() in RETAINED_CONFLICT_OPTIONS
        and isinstance(item.get("reasoning"), str),
        "LOCAL_NDD_OUTPUT_INVALID",
        "V0.6.2.12 requires its own retained-conflict assessment; historical scores cannot substitute",
        field="retained_conflict",
    )
    packet = payload.get("semantic_diff_evidence", {}) if isinstance(payload, dict) else {}
    packet = packet if isinstance(packet, dict) else {}
    spans = {s["span_id"]: s for s in packet.get("spans", []) if isinstance(s, dict) and s.get("span_id")}
    ids, issues = expand_citations(item["reasoning"], set(spans))
    issues = list(issues)
    if packet.get("status") != "COMPLETE":
        issues.append("INCOMPLETE_PACKET")
    score = item["score"].upper()
    if score in MATERIAL_CONFLICTS:
        kinds = {spans[s]["kind"] for s in ids}
        if "SHARED" not in kinds and not {"A_ONLY", "B_ONLY"} <= kinds:
            issues.append("MISSING_BILATERAL_BASIS")
        if not kinds & {"A_ONLY", "B_ONLY"}:
            issues.append("MISSING_MATERIAL_DELTA")
    return RetainedConflict(score, ids, tuple(issues))


def arbitrate_retained_conflict(
    main: dict[str, str],
    ledger: dict[str, str],
    review: RetainedConflict,
    *,
    critic_verdict: str,
    legacy_target: dict[str, str],
    legacy_rule: str | None,
    unresolved: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    """Only a specific, independently cited material conflict can override non-main ownership."""
    if "YES" not in {main.get("a_can_replace_b"), main.get("b_can_replace_a")} or {
        ledger["span_content_profile_a"],
        ledger["span_content_profile_b"],
    } != {"NON_MAIN_ONLY"}:
        return legacy_target, legacy_rule
    if review.score in {"NONE", "NOT_APPLICABLE"}:
        return main, "RETAINED_COMPLETE_MESSAGE_EQUIVALENCE"
    if review.score == "UNRESOLVED" or review.issues:
        return unresolved, "RETAINED_CONFLICT_UNRESOLVED_OR_UNSUPPORTED"
    if critic_verdict not in _VETO_VERDICTS:
        return unresolved, "RETAINED_CONFLICT_BINDING_DISAGREEMENT"
    primary, risk = MATERIAL_CONFLICTS[review.score]
    return {
        **_reject(primary),
        "primary_risk_factor": risk,
        "confidence_tier": "MEDIUM",
    }, f"RETAINED_CONFLICT_VETO_{review.score}"
