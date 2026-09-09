# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from eval.dedup.analysis.policy_review import (
    apply_adjudication_overlay,
    build_policy_review_rows,
    summarize_policy_review,
)


def _prediction(*, same: str, relation: str, ledger_rule: str) -> dict:
    return {
        "canonical_pair_id": "pair-1",
        "same_duplicate_group": same,
        "a_can_replace_b": "NO",
        "b_can_replace_a": "NO",
        "relation_type": relation,
        "material_difference": "MAJOR",
        "confidence_tier": "MEDIUM",
        "reason_codes": [
            "CONTENT_PROFILE_A:SUBSTANTIVE_MAIN",
            "CONTENT_PROFILE_B:NON_MAIN_ONLY",
            "SHARED_CONTENT_BASIS:NONE",
            "HARD_CONFLICT:NONE",
            "DIFFERENCE_LOCATION:A_ONLY_MAIN_ADDITION",
            "RECORD_ALIGNMENT:DIFFERENT_RECORD_OR_ROLE",
            "NON_MAIN_DIFFERENCE:NOT_APPLICABLE",
            "TRANSLATION_STATUS:NOT_TRANSLATION",
            f"SEMANTIC_LEDGER_RULE:{ledger_rule}",
        ],
        "evidence": [
            {"side": "A", "quote": "article body"},
            {"side": "B", "quote": "cookie notice"},
        ],
    }


def test_build_policy_review_rows_flags_policy_boundary_without_relabeling() -> None:
    labels = [
        {
            "review_id": "H0001",
            "canonical_pair_id": "pair-1",
            "stratum_population_n": "10",
            "stratum_sample_n": "2",
            "human_reason_code": "meaningful_addition",
            "human_same_duplicate_group": "YES",
            "human_a_can_replace_b": "YES",
            "human_b_can_replace_a": "NO",
            "human_relation_type": "CONTAINMENT",
            "human_material_difference": "MAJOR",
            "blind_human_reason": "old policy reason",
            "hs_policy_lesson": "lesson",
        }
    ]
    candidate = _prediction(
        same="NO", relation="UNRELATED", ledger_rule="SUBSTANTIVE_NON_MAIN_PROFILE_MISMATCH"
    )
    previous = {
        "canonical_pair_id": "pair-1",
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "NO",
    }
    payloads = [
        {
            "canonical_pair_id": "pair-1",
            "payload": {
                "document_a": {"text": "article body"},
                "document_b": {"text": "cookie notice"},
            },
        }
    ]

    rows = build_policy_review_rows(
        labels=labels,
        candidate_predictions=[candidate],
        previous_predictions=[previous],
        payloads=payloads,
    )

    assert len(rows) == 1
    assert rows[0]["triage_category"] == "POTENTIAL_POLICY_LABEL_CONFLICT"
    assert rows[0]["sample_weight"] == 5.0
    assert rows[0]["review_outcome"] == ""
    assert rows[0]["candidate_evidence_b"] == "cookie notice"
    assert rows[0]["record_alignment"] == "DIFFERENT_RECORD_OR_ROLE"
    assert rows[0]["non_main_difference"] == "NOT_APPLICABLE"
    assert rows[0]["translation_status"] == "NOT_TRANSLATION"
    assert summarize_policy_review(rows)["categories"] == {"POTENTIAL_POLICY_LABEL_CONFLICT": 1}


def test_build_policy_review_rows_omits_primary_matches() -> None:
    label = {
        "review_id": "H0001",
        "canonical_pair_id": "pair-1",
        "human_same_duplicate_group": "NO",
        "human_a_can_replace_b": "NO",
        "human_b_can_replace_a": "NO",
    }
    candidate = _prediction(same="NO", relation="UNRELATED", ledger_rule="HARD_CONFLICT_VETO")
    previous = {
        "canonical_pair_id": "pair-1",
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "NO",
    }
    payload = {
        "canonical_pair_id": "pair-1",
        "payload": {"document_a": {"text": "a"}, "document_b": {"text": "b"}},
    }

    assert not build_policy_review_rows(
        labels=[label],
        candidate_predictions=[candidate],
        previous_predictions=[previous],
        payloads=[payload],
    )


def test_build_policy_review_rows_uses_visible_windows_for_truncated_payload() -> None:
    label = {
        "review_id": "H0001",
        "canonical_pair_id": "pair-1",
        "human_same_duplicate_group": "YES",
        "human_a_can_replace_b": "YES",
        "human_b_can_replace_a": "YES",
    }
    candidate = _prediction(same="NO", relation="UNRELATED", ledger_rule="HARD_CONFLICT_VETO")
    previous = {
        "canonical_pair_id": "pair-1",
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
    }
    payload = {
        "canonical_pair_id": "pair-1",
        "payload": {
            "document_a": {"text": None},
            "document_b": {"text": None},
            "long_document_evidence": {
                "truncated": True,
                "windows": [
                    {"side": "A", "text": "visible A window"},
                    {"side": "B", "text": "visible B window"},
                ],
            },
        },
    }

    rows = build_policy_review_rows(
        labels=[label],
        candidate_predictions=[candidate],
        previous_predictions=[previous],
        payloads=[payload],
    )

    assert rows[0]["document_a"] == "visible A window"
    assert rows[0]["document_b"] == "visible B window"


def test_apply_adjudication_overlay_preserves_frozen_labels() -> None:
    labels = [
        {
            "review_id": "H0001",
            "human_same_duplicate_group": "YES",
            "human_a_can_replace_b": "YES",
            "human_b_can_replace_a": "NO",
            "human_relation_type": "CONTAINMENT",
            "human_material_difference": "MAJOR",
            "human_reason_code": "meaningful_addition",
        },
        {"review_id": "H0002", "human_same_duplicate_group": "NO"},
    ]
    adjudications = [
        {
            "review_id": "H0001",
            "same_duplicate_group": "NO",
            "a_can_replace_b": "NO",
            "b_can_replace_a": "NO",
            "relation_type": "UNRELATED",
            "material_difference": "MAJOR",
            "reason_code": "BOILERPLATE_ONLY_OVERLAP",
            "confidence_tier": "HIGH",
            "reviewer_notes": "The shared text is only a disclaimer",
        }
    ]

    reconciled, summary = apply_adjudication_overlay(labels, adjudications)

    assert reconciled[0]["human_same_duplicate_group"] == "NO"
    assert reconciled[0]["pre_policy_review_human_same_duplicate_group"] == "YES"
    assert reconciled[0]["human_reason_code"] == "meaningful_addition"
    assert reconciled[1] == labels[1]
    assert summary["adjudicated"] == 1
    assert summary["changed_primary_tuple"] == 1
    assert summary["changed_duplicate_group"] == 1


def test_apply_adjudication_overlay_records_explicit_diagnostic_protocol() -> None:
    labels = [
        {
            "review_id": "H0001",
            "human_same_duplicate_group": "YES",
            "human_a_can_replace_b": "YES",
            "human_b_can_replace_a": "NO",
            "human_relation_type": "CONTAINMENT",
            "human_material_difference": "MAJOR",
        }
    ]
    adjudications = [
        {
            "review_id": "H0001",
            "same_duplicate_group": "NO",
            "a_can_replace_b": "NO",
            "b_can_replace_a": "NO",
            "relation_type": "RELATED_NON_DUPLICATE",
            "material_difference": "MAJOR",
            "reason_code": "MATERIAL_NON_MAIN_MESSAGE",
            "confidence_tier": "HIGH",
            "reviewer_notes": "The visible non-main messages differ materially",
            "review_protocol": "DIAGNOSTIC_V062_POLICY",
        }
    ]

    reconciled, summary = apply_adjudication_overlay(labels, adjudications)

    assert reconciled[0]["policy_review_protocol"] == "DIAGNOSTIC_V062_POLICY"
    assert summary["review_protocol"] == "DIAGNOSTIC_V062_POLICY"
