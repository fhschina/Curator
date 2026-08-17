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

from pathlib import Path

import pytest

from eval.dedup.dashboard import (
    _evidence_status,
    _script_json,
    pair_explorer_destination,
    pair_explorer_html,
    pair_explorer_review_queue,
)
from eval.dedup.report import _representative_record, _validate_recommendations
from eval.dedup.validation import DedupEvaluationError


def test_pair_explorer_destination_tracks_report_label() -> None:
    assert pair_explorer_destination(Path("reports/final_report.md")) == Path("reports/pair_explorer.html")
    assert pair_explorer_destination(Path("reports/final_report.review-v3.md")) == Path(
        "reports/pair_explorer.review-v3.html"
    )


def test_representative_record_uses_slice_median_then_stable_tiebreakers() -> None:
    records = [
        {"pair_id": "cp1_a", "token_length_ratio": 0.1, "evidence": [], "confidence": 1.0},
        {"pair_id": "cp1_b", "token_length_ratio": 0.5, "evidence": [{"quote": "x"}], "confidence": 0.9},
        {"pair_id": "cp1_c", "token_length_ratio": 0.9, "evidence": [], "confidence": 1.0},
    ]
    used: set[str] = set()

    selected = _representative_record(records, used=used, predicate=lambda _: True)

    assert selected is not None
    assert selected["pair_id"] == "cp1_b"
    assert used == {"cp1_b"}


def test_pair_explorer_review_queue_keeps_actionable_pairs_and_report_controls() -> None:
    records = [
        {"pair_id": "wrong", "outcomes": ["wrong_removal"], "judge_status": "valid"},
        {"pair_id": "positive", "outcomes": ["discovered_candidate_fn"], "judge_status": "valid"},
        {"pair_id": "unresolved", "outcomes": ["unresolved"], "judge_status": "valid"},
        {"pair_id": "error", "outcomes": ["judge_error"], "judge_status": "judge_error"},
        {"pair_id": "control", "outcomes": ["safe_removal"], "judge_status": "valid"},
        {"pair_id": "negative", "outcomes": ["hard_negative"], "judge_status": "valid"},
    ]
    examples = {"removal": [records[4]], "cross": []}

    selected = pair_explorer_review_queue(records, examples)

    assert {row["pair_id"] for row in selected} == {"wrong", "positive", "unresolved", "error", "control"}


def test_pair_explorer_escapes_untrusted_text_inside_script_data() -> None:
    record = {
        "pair_id": "cp1_fixture",
        "left": {"excerpt": "</script><script>alert(1)</script>"},
        "right": {"excerpt": "safe"},
    }

    dashboard = pair_explorer_html(evaluation_run_id="fixture", records=[record])

    assert "__PAIR_DATA__" not in dashboard
    assert "__GROUP_CONTEXT_DATA__" not in dashboard
    assert "</script><script>alert(1)</script>" not in dashboard
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)" in dashboard
    assert _script_json(record) in dashboard


def test_pair_explorer_contains_decision_review_and_context_sections() -> None:
    dashboard = pair_explorer_html(
        evaluation_run_id="fixture",
        records=[],
        group_contexts={"7": {"cluster_key": "group:fixture:7"}},
    )

    assert "SUT decision" in dashboard
    assert "Judge verdict" in dashboard
    assert "Evaluation outcome" in dashboard
    assert "Human review" in dashboard
    assert "Export reviews CSV" in dashboard
    assert "SUT group context" in dashboard
    assert "group:fixture:7" in dashboard


def test_evidence_status_reports_partial_repair_coverage() -> None:
    status = _evidence_status(
        {
            "evidence": [{"quote": "retained"}],
            "deterministic_repair_events": [
                {"action": "realign_offsets"},
                {"action": "drop_unalignable"},
                {"action": "drop_unalignable"},
            ],
        },
        judge_status="valid",
    )

    assert status == {"returned": 3, "retained": 1, "realigned": 1, "dropped": 2, "coverage": "PARTIAL"}


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
