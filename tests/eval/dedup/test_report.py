# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest

from eval.dedup.report import _validate_recommendations
from eval.dedup.validation import DedupEvaluationError


def _recommendations() -> dict:
    return {
        "key_findings": [{"finding": "Candidate-pool positive yield is low.", "evidence_refs": ["yield"]}],
        "risks": [{"finding": "The selected pool is inefficient.", "evidence_refs": ["yield"]}],
        "recommended_actions": [
            {
                "priority": "HIGH",
                "action": "Improve candidate-pool precision.",
                "rationale": "This reduces wasted Judge operations.",
                "evidence_refs": ["yield"],
            }
        ],
    }


def test_recommendation_validation_accepts_bounded_candidate_yield_claim() -> None:
    validated, dropped = _validate_recommendations(_recommendations(), {"yield"})
    assert validated["recommended_actions"][0]["priority"] == "HIGH"
    assert dropped == 0


def test_recommendation_validation_drops_recall_claim_from_candidate_yield() -> None:
    recommendations = _recommendations()
    recommendations["recommended_actions"].append(
        {
            "priority": "LOW",
            "action": "Reduce wasted candidate reviews.",
            "rationale": "The selected candidate pool has low positive yield.",
            "evidence_refs": ["yield"],
        }
    )
    recommendations["recommended_actions"][0]["action"] = "Improve retriever recall."
    validated, dropped = _validate_recommendations(recommendations, {"yield"})
    assert [item["action"] for item in validated["recommended_actions"]] == ["Reduce wasted candidate reviews."]
    assert dropped == 1


def test_recommendation_validation_rejects_empty_collection_after_scope_guard() -> None:
    recommendations = _recommendations()
    recommendations["recommended_actions"][0]["action"] = "Improve retriever recall."
    with pytest.raises(DedupEvaluationError, match="RECOMMENDATION_COLLECTION_EMPTY_AFTER_SCOPE_GUARD"):
        _validate_recommendations(recommendations, {"yield"})
