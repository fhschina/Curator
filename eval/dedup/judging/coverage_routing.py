# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage-call ownership decisions made before observing any critic response."""

from __future__ import annotations

from dataclasses import dataclass

from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.record_scope import complete_visible_equality
from eval.dedup.judging.schema_v3 import JUDGE_SCHEMA_V3, unresolved_judge_output_v3, validate_judge_output_v3

ROUTING_CONTRACT = "dedup-coverage-call-routing-v1"


@dataclass(frozen=True)
class CoverageRoute:
    route: str
    public_output: dict | None


def route_coverage(main: dict, payload: dict) -> CoverageRoute:
    """Preserve immutable coverage arbitration branches; never skip based on critic quality."""
    if (
        payload.get("semantic_diff_evidence", {}).get("status") != "COMPLETE"
        or payload.get("long_document_evidence", {}).get("truncated") is not False
    ):
        route, public = "INCOMPLETE_INPUT", unresolved_judge_output_v3()
    else:
        source = adapt_ndd_judge_output(main, JUDGE_SCHEMA_V3, payload=payload, record_binding_policy="v6-route")
        if source["same_duplicate_group"] != "YES":
            route, public = "MAIN_" + source["same_duplicate_group"], source
        elif complete_visible_equality(payload):
            route, public = (
                "VISIBLE_EXACT",
                {
                    **source,
                    "relation_type": "EXACT",
                    "material_difference": "NONE",
                    "primary_material_difference": "NONE",
                    "evidence": [],
                },
            )
        else:
            return CoverageRoute("NEEDS_COVERAGE", None)
    validate_judge_output_v3(public)
    validate_evidence_offsets(public, payload)
    return CoverageRoute(route, public)
