# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-grounded, veto-only retained-coverage critic (retention-v4 rules)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from tutorials.eval.dedup.critics.dedup_adapter import (
    _offset,
    adapt_alignment,
    complete_text_equality,
    normalize_main,
    require,
    validate_decision,
)

if TYPE_CHECKING:
    import data_designer.config as dd

CONFLICTS = {
    "IDENTITY_CONFLICT": ("document_identity_change", "template_slot_collision"),
    "STATE_CONFLICT": ("other_material", "identifier_underweighting"),
    "POLICY_CONFLICT": ("legal_context_change", "legal_context_collision"),
    "ROLE_CONFLICT": ("page_role_change", "page_role_collision"),
    "MEMBERSHIP_CONFLICT": ("result_set_change", "list_snapshot_collision"),
}
OVERLAP_BASES = ("RETAINED_CONTENT", "INTERFACE_ONLY", "UNCERTAIN")
ID_FIELDS = ("a_loss_span_id", "b_loss_span_id", "a_context_span_id", "b_context_span_id")


def response_schema() -> dict[str, Any]:
    fields = {name: {"type": "string", "minLength": 0 if "loss" in name else 1} for name in ID_FIELDS}
    fields.update(
        conflict={"type": "string", "enum": ["NONE", *CONFLICTS]},
        overlap_basis={"type": "string", "enum": list(OVERLAP_BASES)},
        shared_anchor_ids={"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        explanation={"type": "string", "minLength": 1},
    )
    return {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}


def _compile_review(value: object, packet: dict[str, Any], pair_id: object) -> tuple[str, list[dict[str, Any]]]:
    require(
        isinstance(value, dict) and set(value) == set(response_schema()["required"]),
        "invalid coverage review schema",
        pair_id=pair_id,
    )
    require(
        all(isinstance(value[key], str) for key in (*ID_FIELDS, "conflict", "overlap_basis", "explanation"))
        and value["conflict"] in ("NONE", *CONFLICTS)
        and value["overlap_basis"] in OVERLAP_BASES
        and bool(value["explanation"].strip()),
        "invalid review field types or categories",
        pair_id=pair_id,
    )
    spans = {span["span_id"]: span for span in packet["spans"]}
    anchors = value["shared_anchor_ids"]
    shared = {sid for sid, span in spans.items() if span["kind"] == "SHARED"}
    require(
        isinstance(anchors, list)
        and all(isinstance(sid, str) for sid in anchors)
        and len(anchors) == len(set(anchors))
        and set(anchors) <= shared,
        "anchors must be unique existing shared IDs",
        pair_id=pair_id,
    )
    evidence, unique = [], set()
    for side in ("A", "B"):
        loss, context = (value[f"{side.lower()}_{name}_span_id"] for name in ("loss", "context"))
        require(bool(context), "both checked context IDs are required", pair_id=pair_id)
        for sid in dict.fromkeys([loss, context]):
            if not sid:
                continue
            require(sid in spans, f"unknown span ID {sid!r}", pair_id=pair_id)
            span = spans[sid]
            own_unique = span["kind"] == f"{side}_ONLY"
            require(
                own_unique or (sid != loss and span["kind"] == "SHARED"),
                f"wrong side or non-unique loss: {sid}",
                pair_id=pair_id,
            )
            prefix = side.lower() + "_" if span["kind"] == "SHARED" else ""
            evidence.append(
                {
                    "side": side,
                    **{key: _offset(span, prefix + key, pair_id) for key in ("start_char", "end_char")},
                    "quote": span[prefix + "text"],
                }
            )
            if own_unique:
                unique.add(side)
    a_loss, b_loss = bool(value["a_loss_span_id"]), bool(value["b_loss_span_id"])
    if value["conflict"] != "NONE":
        action = "REJECT_BOTH"
        require(bool(unique), "conflict requires a unique delta witness", pair_id=pair_id)
    elif value["overlap_basis"] == "UNCERTAIN":
        action = "ABSTAIN"
    elif value["overlap_basis"] == "INTERFACE_ONLY" and (a_loss or b_loss):
        action = "REJECT_EMPTY_ANCHOR"
        require(
            bool(shared) and set(anchors) == shared, "empty-anchor veto requires all shared spans", pair_id=pair_id
        )
    else:
        action = {
            (False, False): "KEEP_MAIN",
            (True, False): "REJECT_B_REPLACES_A",
            (False, True): "REJECT_A_REPLACES_B",
            (True, True): "REJECT_BOTH",
        }[(a_loss, b_loss)]
    if action not in {"KEEP_MAIN", "ABSTAIN"}:
        require(
            packet["status"] == "COMPLETE" and packet["truncated"] is False,
            "veto requires complete, untruncated evidence",
            pair_id=pair_id,
        )
    return action, evidence


def _apply_action(main: dict[str, str], review: dict[str, Any], action: str) -> tuple[dict[str, str], str, str]:
    result = dict(main)
    if action == "KEEP_MAIN":
        return result, action, "NO_SUPPORTED_OBJECTION_KEEP_MAIN"
    if action == "ABSTAIN":
        result = dict.fromkeys(main, "unresolved")
        result.update(primary_risk_factor="extraction_or_payload_limit", confidence_tier="low")
        return result, action, "EXPLICIT_CRITIC_ABSTENTION"
    if action == "REJECT_EMPTY_ANCHOR" and main["relation_type"] != "containment":
        return result, "KEEP_MAIN", "EMPTY_ANCHOR_VETO_OUTSIDE_CONTAINMENT_SCOPE"
    rejected = {
        "REJECT_A_REPLACES_B": ("a_can_replace_b",),
        "REJECT_B_REPLACES_A": ("b_can_replace_a",),
        "REJECT_BOTH": ("a_can_replace_b", "b_can_replace_a"),
        "REJECT_EMPTY_ANCHOR": ("a_can_replace_b", "b_can_replace_a"),
    }[action]
    if not any(main[key] == "yes" for key in rejected):
        return result, "KEEP_MAIN", "OBJECTION_ONLY_TO_ALREADY_UNSAFE_DIRECTION"
    result.update(dict.fromkeys(rejected, "no"))
    same = "yes" in (result["a_can_replace_b"], result["b_can_replace_a"])
    primary, risk = CONFLICTS.get(review["conflict"], ("other_material", "boilerplate_dominated_similarity"))
    result.update(
        relation_type="containment"
        if same
        else "version_related"
        if review["conflict"] == "STATE_CONFLICT"
        else "related_non_duplicate",
        material_difference="major",
        primary_material_difference="main_content_addition_deletion" if same else primary,
        primary_risk_factor="containment_asymmetry" if same else risk,
        confidence_tier="medium",
    )
    return result, action, "SUPPORTED_DIRECTIONAL_VETO"


@dataclass(frozen=True)
class CoverageCritic:
    source_judge: str

    name: ClassVar[str] = "coverage"
    temporary_columns: ClassVar[tuple[str, ...]] = ("_coverage_payload", "_coverage_main")
    prepared_columns: ClassVar[tuple[str, ...]] = ("coverage_should_run", "coverage_reason", *temporary_columns)
    applied_columns: ClassVar[tuple[str, ...]] = (
        "coverage_action",
        "coverage_reason",
        "coverage_evidence",
        "final_decision",
    )
    output_columns: ClassVar[tuple[str, ...]] = (
        "coverage_should_run",
        "coverage_review",
        "coverage_action",
        "coverage_reason",
        "coverage_evidence",
        "final_decision",
    )

    def prepare(self, record: dict[str, Any]) -> dict[str, Any]:
        main = normalize_main(record, self.source_judge)
        packet = adapt_alignment(record)
        if "yes" not in (main["a_can_replace_b"], main["b_can_replace_a"]):
            reason = "PRESERVE_MAIN_NEGATIVE_OR_UNRESOLVED"
        elif complete_text_equality(record, packet):
            reason = "PRESERVE_COMPLETE_EXACT_INPUT"
        else:
            reason = "REVIEW_POSITIVE_DIRECTIONS_ONLY"
        return {
            "coverage_should_run": reason == "REVIEW_POSITIVE_DIRECTIONS_ONLY",
            "coverage_reason": reason,
            "_coverage_payload": packet,
            "_coverage_main": main,
        }

    def build_column(self, *, model_alias: str, prompt: str, system_prompt: str) -> dd.LLMStructuredColumnConfig:
        import data_designer.config as dd

        return dd.LLMStructuredColumnConfig(
            name="coverage_review",
            model_alias=model_alias,
            prompt=prompt,
            system_prompt=system_prompt,
            output_format=response_schema(),
            skip=dd.SkipConfig(when="{{ not coverage_should_run }}"),
        )

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        pair_id = record.get("pair_id")
        should_run = record["coverage_should_run"]
        require(type(should_run) is bool, "coverage_should_run must be boolean", pair_id=pair_id)
        main, review = record["_coverage_main"], record.get("coverage_review")
        if not should_run:
            require(review is None, "skipped row has an unexpected review", pair_id=pair_id)
            return {
                "coverage_action": "SKIP",
                "coverage_reason": record["coverage_reason"],
                "coverage_evidence": [],
                "final_decision": dict(main),
            }
        action, evidence = _compile_review(review, record["_coverage_payload"], pair_id)
        result, action, reason = _apply_action(main, review, action)
        validate_decision(result, pair_id=pair_id)
        return {
            "coverage_action": action,
            "coverage_reason": reason,
            "coverage_evidence": evidence,
            "final_decision": result,
        }
