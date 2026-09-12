# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explain invalid V5 evidence bindings without repairing or weakening the certificate."""

from __future__ import annotations

from eval.dedup.judging.context_coverage import SIDES, adapt_context, validate_context_shape
from eval.dedup.judging.coverage_selection import span_inventory
from eval.dedup.validation import DedupEvaluationError

FEEDBACK_CONTRACT = "dedup-directional-witness-feedback-v1"


def binding_issues(value: dict, payload: dict) -> list[dict]:
    validate_context_shape(value)
    spans = span_inventory(payload)
    unique = {s: sorted(k for k, v in spans.items() if v.get("side") == s and v["kind"] != "SHARED") for s in SIDES}
    shared = sorted(k for k, v in spans.items() if v["kind"] == "SHARED")
    issues = []

    def check(path: str, actual: str, allowed: list[str], rule: str) -> None:
        if actual not in allowed:
            issues.append({"field": path, "actual": actual, "allowed_single_ids": allowed, "rule": rule})

    conflict = value["hard_conflict"]
    has_shared_source = False
    for side, field in SIDES.items():
        other = "B" if side == "A" else "A"
        item = value[field]
        if item["status"] != "UNCOVERED":
            continue
        source, counterpart = item["source_span_id"], item["counterpart_span_id"]
        check(f"{field}.source_span_id", source, unique[side] + shared, f"source belongs to document {side}")
        check(
            f"{field}.counterpart_span_id",
            counterpart,
            unique[other] + shared + ([""] if item["counterpart_relation"] != "CONTRADICTS" else []),
            f"counterpart belongs to document {other}, never document {side}'s unique spans",
        )
        if source in shared:
            has_shared_source = True
            check(
                f"{field}.counterpart_span_id",
                counterpart,
                unique[other],
                f"shared source requires an actual opposite-unique difference on document {other}",
            )
            for s, sid in ((side, source), (other, counterpart)):
                if sid in unique[s] + shared:
                    check(
                        f"hard_conflict.{s.lower()}_span_id",
                        conflict[f"{s.lower()}_span_id"],
                        [sid],
                        "must match the same directional conflict witness after correcting its document ownership",
                    )
    if has_shared_source and all(conflict[f"{s.lower()}_span_id"] in shared for s in SIDES):
        issues.append(
            {
                "field": "hard_conflict",
                "actual": {s: conflict[f"{s.lower()}_span_id"] for s in SIDES},
                "unique_choices_by_document": unique,
                "rule": "at least one root witness must cite the actual differing unique span; two shared roots do not suffice",
            }
        )
    return issues


def explain_invalid_binding(value: dict, payload: dict) -> None:
    """Raise only when the frozen adapter rejects; valid outputs have identical semantics."""
    try:
        adapt_context(value, payload)
    except DedupEvaluationError as original:
        issues = binding_issues(value, payload)
        if issues:
            # These existing safe-feedback keys survive the immutable retry encoder.
            raise DedupEvaluationError(
                "DIRECTIONAL_WITNESS_BINDING",
                "Re-select the cited IDs on their required documents, then make root and directional witnesses agree. "
                "Do not change semantic coverage statuses merely to bypass an ID error. Return the full fenced JSON.",
                field=[i["field"] for i in issues],
                actual_fields=[{"field": i["field"], "value": i["actual"]} for i in issues],
                expected_fields=[{k: v for k, v in i.items() if k != "actual"} for i in issues],
            ) from original
        raise
