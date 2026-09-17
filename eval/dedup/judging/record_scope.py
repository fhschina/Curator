# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""V0.6.2.13 proof requirements for exact identity and positive record binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.dedup.core.validation import require
from eval.dedup.judging.boundary_critic import _reject
from eval.dedup.judging.retained_conflict import MATERIAL_CONFLICTS, RetainedConflict
from eval.dedup.judging.scoped_critic import expand_citations

RECORD_SCOPE_OPTIONS = frozenset(
    {
        "SAME_SPECIFIC_RECORD",
        "SAME_ORGANIZATION_DESCRIPTION",
        "EQUIVALENT_COMPLETE_MESSAGE",
        "GENERIC_CONTEXT_ONLY",
        "DISTINCT_RECORD",
        "LIST_MEMBERSHIP_CHANGE",
        "NOT_APPLICABLE",
        "UNRESOLVED",
    }
)
_BOUND_RECORDS = {"SAME_SPECIFIC_RECORD", "SAME_ORGANIZATION_DESCRIPTION"}
_SEPARATE_SCOPES = {"GENERIC_CONTEXT_ONLY", "DISTINCT_RECORD", "LIST_MEMBERSHIP_CHANGE"}
_NEGATIVE_VERDICTS = {
    "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
    "TWO_SIDED_OR_CONFLICTING",
    "NON_MAIN_POLICY_OR_STATE_CHANGE",
}


def complete_visible_equality(payload: dict[str, Any] | None) -> bool:
    """Token alignment equality alone cannot prove full, untruncated text identity."""
    if not isinstance(payload, dict):
        return False
    evidence = payload.get("long_document_evidence", {})
    packet = payload.get("semantic_diff_evidence", {})
    if evidence.get("truncated") is not False or packet.get("status") != "COMPLETE":
        return False
    text_a = payload.get("document_a", {}).get("text")
    text_b = payload.get("document_b", {}).get("text")
    return isinstance(text_a, str) and bool(text_a.strip()) and text_a == text_b


@dataclass(frozen=True)
class RecordScope:
    score: str
    citations: list[str]
    issues: tuple[str, ...]
    benign_covers_all_deltas: bool


def parse_record_scope(value: Any, payload: dict[str, Any] | None) -> RecordScope:
    item = value.get("record_scope") if isinstance(value, dict) else None
    require(
        isinstance(item, dict)
        and isinstance(item.get("score"), str)
        and item["score"].upper() in RECORD_SCOPE_OPTIONS
        and isinstance(item.get("reasoning"), str),
        "LOCAL_NDD_OUTPUT_INVALID",
        "V0.6.2.13 requires its own record-scope proof; historical responses cannot supply it",
        field="record_scope",
    )
    packet = payload.get("semantic_diff_evidence", {}) if isinstance(payload, dict) else {}
    spans = {s["span_id"]: s for s in packet.get("spans", []) if isinstance(s, dict) and s.get("span_id")}
    ids, syntax_issues = expand_citations(item["reasoning"], set(spans))
    score, issues = item["score"].upper(), list(syntax_issues)
    if packet.get("status") != "COMPLETE":
        issues.append("INCOMPLETE_PACKET")
    if score not in {"UNRESOLVED", "NOT_APPLICABLE"}:
        kinds = {spans[s]["kind"] for s in ids}
        if "SHARED" not in kinds and not {"A_ONLY", "B_ONLY"} <= kinds:
            issues.append("MISSING_BILATERAL_BASIS")
        if score in _SEPARATE_SCOPES and not kinds & {"A_ONLY", "B_ONLY"}:
            issues.append("MISSING_RECORD_DELTA")
    unique_ids = {s for s, span in spans.items() if span["kind"] in {"A_ONLY", "B_ONLY"}}
    reasoning = value.get("record_binding_verdict", {}).get("reasoning", "")
    coverage_ids, coverage_issues = expand_citations(reasoning, set(spans))
    covered = bool(unique_ids) and not coverage_issues and unique_ids <= set(coverage_ids)
    return RecordScope(score, ids, tuple(issues), covered)


def arbitrate_record_scope(  # noqa: PLR0911 - distinct evidence obligations must fail independently
    main: dict[str, str],
    scope: RecordScope,
    retained: RetainedConflict,
    critic: tuple[str, list[str], str | None],
    *,
    legacy_target: dict[str, str],
    legacy_rule: str | None,
    unresolved: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    """Do not reopen negative/unknown main decisions or turn an unsupported claim into a negative."""
    if "YES" not in {main.get("a_can_replace_b"), main.get("b_can_replace_a")}:
        return legacy_target, legacy_rule
    verdict, _, critic_issue = critic
    if scope.score in _SEPARATE_SCOPES:
        if scope.issues or critic_issue or verdict not in _NEGATIVE_VERDICTS:
            return unresolved, "RECORD_SCOPE_SEPARATION_PROOF_DISAGREEMENT"
        primary = "RESULT_SET_CHANGE" if scope.score == "LIST_MEMBERSHIP_CHANGE" else "DOCUMENT_IDENTITY_CHANGE"
        return _reject(primary), f"RECORD_SCOPE_VETO_{scope.score}"
    if main.get("relation_type") != "CONTAINMENT":
        return legacy_target, legacy_rule
    if verdict in _NEGATIVE_VERDICTS:
        return legacy_target, legacy_rule
    if scope.issues or critic_issue:
        return unresolved, "RECORD_SCOPE_EXTENSION_PROOF_INVALID"
    if verdict == "BENIGN_NON_RECORD_DELTA":
        if (
            scope.score not in _BOUND_RECORDS | {"EQUIVALENT_COMPLETE_MESSAGE"}
            or not scope.benign_covers_all_deltas
            or retained.score not in {"NONE", "NOT_APPLICABLE"}
        ):
            return unresolved, "RECORD_SCOPE_BENIGN_COVERAGE_UNPROVEN"
        return {
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "NEAR_SURFACE",
            "material_difference": "MINOR",
            "primary_material_difference": "OTHER_MATERIAL",
            "confidence_tier": "MEDIUM",
        }, "RECORD_SCOPE_VERIFIED_BENIGN_EQUIVALENCE"
    if verdict == "ATOMIC_SAME_RECORD_EXTENSION" and scope.score in _BOUND_RECORDS:
        return legacy_target, legacy_rule
    return unresolved, "RECORD_SCOPE_EXTENSION_IDENTITY_UNPROVEN"


def arbitrate_record_scope_v8(
    main: dict[str, str],
    scope: RecordScope,
    retained: RetainedConflict,
    critic: tuple[str, list[str], str | None],
    *,
    ledger: dict[str, str],
    legacy_target: dict[str, str],
    legacy_rule: str | None,
    unresolved: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    """Scope cannot invalidate an established veto or manufacture semantic coverage."""
    directions = {legacy_target.get("a_can_replace_b"), legacy_target.get("b_can_replace_a")}
    if directions == {"NO"} or {ledger.get("span_content_profile_a"), ledger.get("span_content_profile_b")} == {
        "NON_MAIN_ONLY"
    }:
        return legacy_target, legacy_rule
    if main.get("relation_type") == "CONTAINMENT" and directions == {"YES", "NO"}:
        if critic[0] == "BENIGN_NON_RECORD_DELTA":
            # Enumerating every delta proves citation coverage, not opposite-side entailment.
            return {**legacy_target, "confidence_tier": "LOW"}, "RECORD_SCOPE_EXTENSION_BENIGN_DISAGREEMENT_PRESERVED"
        if (
            ledger.get("span_translation_status") == "PARTIAL_OR_ADDITIVE"
            and scope.score in _BOUND_RECORDS
            and not scope.issues
            and critic[0] == "ATOMIC_SAME_RECORD_EXTENSION"
        ):
            # Additive translation is already direction-owned by the main ledger; this
            # keeps that result without treating another field's citations as critic proof.
            return legacy_target, legacy_rule
    return arbitrate_record_scope(
        main,
        scope,
        retained,
        critic,
        legacy_target=legacy_target,
        legacy_rule=legacy_rule,
        unresolved=unresolved,
    )


def arbitrate_record_scope_v9(
    main: dict[str, str],
    scope: RecordScope,
    retained: RetainedConflict,
    critic: tuple[str, list[str], str | None],
    *,
    ledger: dict[str, str],
    legacy_target: dict[str, str],
    legacy_rule: str | None,
    unresolved: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    """A substantive profile cannot hide a proved permission change or coverage disagreement."""
    target, rule = arbitrate_record_scope_v8(
        main,
        scope,
        retained,
        critic,
        ledger=ledger,
        legacy_target=legacy_target,
        legacy_rule=legacy_rule,
        unresolved=unresolved,
    )
    if "YES" not in {main.get("a_can_replace_b"), main.get("b_can_replace_a")} or "YES" not in {
        target.get("a_can_replace_b"),
        target.get("b_can_replace_a"),
    }:
        return target, rule
    if retained.score in MATERIAL_CONFLICTS:
        if retained.issues or critic[2] or critic[0] not in _NEGATIVE_VERDICTS:
            return unresolved, "RETENTION_MATERIAL_PROOF_DISAGREEMENT"
        primary, risk = MATERIAL_CONFLICTS[retained.score]
        return {
            **_reject(primary),
            "primary_risk_factor": risk,
            "confidence_tier": "MEDIUM",
        }, f"RETENTION_MATERIAL_VETO_{retained.score}"
    if (
        main.get("a_can_replace_b") == main.get("b_can_replace_a") == "YES"
        and ledger.get("span_content_profile_a") == ledger.get("span_content_profile_b") == "SUBSTANTIVE_MAIN"
        and critic[0] == "ATOMIC_SAME_RECORD_EXTENSION"
        and not critic[2]
        and scope.score in _BOUND_RECORDS
        and not scope.issues
    ):
        # Existing fields do not encode which side's retained propositions remain
        # uncovered. Citations alone cannot supply a replacement direction or veto.
        return unresolved, "RETENTION_EQUIVALENCE_EXTENSION_DISAGREEMENT"
    return target, rule
