# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def pair() -> dict[str, Any]:
    summary = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
    }
    return {
        "pair_id": "pair-1",
        "text_a": "Report: apples",
        "text_b": "Report: pears",
        "truncated": False,
        "pair_semantic_judgment": {
            key: {"score": value, "reasoning": "Original main reasoning."} for key, value in summary.items()
        },
        "semantic_diff": {
            "status": "COMPLETE",
            "truncated": False,
            "truncated_a": False,
            "truncated_b": False,
            "span_counts": {"SHARED": 1, "A_ONLY": 1, "B_ONLY": 1},
            "spans": [
                {
                    "span_id": "S001",
                    "kind": "SHARED",
                    "a_text": "Report:",
                    "b_text": "Report:",
                    "a_start_char": 0,
                    "a_end_char": 7,
                    "b_start_char": 0,
                    "b_end_char": 7,
                },
                {
                    "span_id": "A001",
                    "kind": "A_ONLY",
                    "text": "Report: apples",
                    "start_char": 0,
                    "end_char": 14,
                    "delta_start_char": 8,
                    "delta_end_char": 14,
                },
                {
                    "span_id": "B001",
                    "kind": "B_ONLY",
                    "text": "Report: pears",
                    "start_char": 0,
                    "end_char": 13,
                    "delta_start_char": 8,
                    "delta_end_char": 13,
                },
            ],
        },
    }


@pytest.fixture
def review() -> dict[str, Any]:
    return {
        "a_loss_span_id": "",
        "b_loss_span_id": "",
        "a_context_span_id": "A001",
        "b_context_span_id": "B001",
        "conflict": "NONE",
        "overlap_basis": "RETAINED_CONTENT",
        "shared_anchor_ids": ["S001"],
        "explanation": "Compared both documents.",
    }
