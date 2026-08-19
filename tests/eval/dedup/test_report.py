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
from eval.dedup.report import (
    _pipeline_accounting_tree,
    _representative_record,
    _step_5b_generation_markdown,
    _validate_recommendations,
)
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


def test_pair_explorer_contains_language_filter_and_reason_code_chart() -> None:
    dashboard = pair_explorer_html(evaluation_run_id="fixture", records=[])

    assert 'id="language"' in dashboard
    assert "Any language · either document" in dashboard
    assert "Same-language pairs" in dashboard
    assert "Cross-language pairs" in dashboard
    assert "r.left.language===lv||r.right.language===lv" in dashboard
    assert "Reason code distribution" in dashboard
    assert "new Set(r.reason_codes)" in dashboard
    assert "multi-label counts" in dashboard


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


def test_pipeline_accounting_separates_step_5a_from_step_5b_anchors() -> None:
    markers = [
        {},
        {"counts": {"grouped_documents": 2_021_220, "groups": 352_601, "removals": 1_668_619}},
        {"counts": {"singletons": 7_986_841}},
        {"counts": {"rows": 1_000}},
        {
            "counts": {
                "removal_frame_size": 1_668_619,
                "removal_rows": 10_000,
                "cross_lexical_candidates": 27_463,
                "cross_semantic_candidates": 50_000,
                "cross_union_candidates": 73_363,
                "cross_unique_selected_pairs": 10_000,
            }
        },
    ]

    tree = _pipeline_accounting_tree(
        evaluation_manifest={"dataset_row_count": 10_008_061},
        markers=markers,
        judge={"requested": 20_000, "schema_valid": 19_994, "resolved": 19_990, "unresolved": 4, "errors": 6},
    )

    assert "1,668,619 SUT removal decisions\n└── 10,000 uniformly sampled Step 5a" in tree
    assert "1,000 Step 4 anchors (Step 5b only)" in tree
    assert "73,363 cross-channel union records\n    └── 10,000 selected Step 5b" in tree
    assert "anchors\n├── 10,000 Step 5a" not in tree


def test_step_5b_generation_explains_channels_funnel_and_quota_scope() -> None:
    markers = [
        {},
        {},
        {},
        {"counts": {"rows": 1_000}},
        {
            "counts": {
                "cross_lexical_candidates": 27_463,
                "cross_semantic_candidates": 50_000,
                "cross_union_candidates": 73_363,
                "cross_unique_selected_pairs": 10_000,
            }
        },
    ]
    retrieval_config = {
        "selected_lsh": {"bands": 7, "rows_per_band": 1},
        "lexical_trials": [{"bands": 7, "rows_per_band": 1, "median_cross_group_candidates": 31.0}],
        "pilot_candidate_count_target": {"minimum": 20, "maximum": 50},
        "top_k": 50,
    }

    rendered = _step_5b_generation_markdown(
        evaluation_manifest={"upstream_provenance_availability": {"resolved_config": False}},
        markers=markers,
        cross_outcome={"resolved": 9_997},
        retrieval_config=retrieval_config,
    )

    assert "two parallel retrieval channels" in rendered
    assert "7 bands x\n1 row per band" in rendered
    assert "median of\n31 cross-group candidates" in rendered
    assert "does not claim a numeric SUT-to-evaluation threshold change such as 0.8 to 0.7" in rendered
    assert "| Cross-channel union | 73363 |" in rendered
    assert "| Selected unique pairs | 10000 |" in rendered
    assert "not natural corpus prevalence, channel recall" in rendered


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
