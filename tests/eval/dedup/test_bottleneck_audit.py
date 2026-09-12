# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from copy import deepcopy

import pytest

from eval.dedup.analysis.bottleneck_audit import (
    blind_packets,
    cause_accounting,
    exact_conflicts,
    mixed_panel,
    oracle_sensitivity,
    review_ledger,
    transition_matrix,
)
from eval.dedup.validation import DedupEvaluationError


def fixture_rows():
    labels, mains, finals, payloads = [], [], [], []
    cases = [
        ("NO", "YES", "YES", 2),
        ("YES", "YES", "NO", 3),
        ("YES", "YES", "UNRESOLVED", 5),
        ("YES", "YES", "YES", 7),
        ("NO", "YES", "NO", 11),
    ]
    for i, (truth, main, final, weight) in enumerate(cases):
        labels.append(
            {
                "review_id": f"H{i}",
                "canonical_pair_id": f"p{i}",
                "stratum_population_n": weight * 2,
                "stratum_sample_n": 2,
                "sample_weight": 999,
                "human_same_duplicate_group": truth,
                "human_a_can_replace_b": truth,
                "human_b_can_replace_a": truth,
                "human_relation_type": "NEAR_SURFACE" if truth == "YES" else "UNRELATED",
                "human_material_difference": "NONE",
                "human_reason_code": "test",
            }
        )
        for value, target in ((main, mains), (final, finals)):
            target.append(
                {
                    "canonical_pair_id": f"p{i}",
                    "same_duplicate_group": value,
                    "a_can_replace_b": value,
                    "b_can_replace_a": value,
                    "relation_type": "UNRELATED",
                    "material_difference": "NONE",
                }
            )
        payloads.append(
            {
                "canonical_pair_id": f"p{i}",
                "payload": {
                    "document_a": {"text": "A"},
                    "document_b": {"text": "B"},
                    "long_document_evidence": {"truncated": False},
                    "semantic_diff_evidence": {"status": "COMPLETE", "spans": [{"span_id": "A001"}]},
                },
            }
        )
    finals[3]["b_can_replace_a"] = "NO"
    reviews = {
        f"H{i}": {
            "cause": cause,
            "cluster": "test",
            "note": "explicit evidence review",
            "locus": "MIXED",
            "evidence_ids": ["A001"],
        }
        for i, cause in enumerate(
            ["SEMANTIC_ERROR", "POLICY_REFERENCE_DISPUTE", "ENGINEERING_ERROR", "SEMANTIC_ERROR"]
        )
    }
    return labels, mains, finals, payloads, reviews


def test_confusion_weights_close_without_double_counting_direction_or_unresolved():
    ledger = review_ledger(*fixture_rows())
    result = cause_accounting(ledger)
    causes = result["by_cause"]
    assert result["total_weight"] == 28
    assert causes["SEMANTIC_ERROR"]["error_weight"] == 9
    assert causes["SEMANTIC_ERROR"]["fp_weight"] == 2
    assert causes["SEMANTIC_ERROR"]["fn_weight"] == 0
    assert causes["SEMANTIC_ERROR"]["direction_weight"] == 7
    assert causes["ENGINEERING_ERROR"]["fn_weight"] == 5
    assert causes["ENGINEERING_ERROR"]["unresolved_weight"] == 5
    assert sum(r["precision_loss_contribution_pp"] for r in causes.values()) == pytest.approx(100 * 2 / 9)
    assert sum(r["recall_loss_contribution_pp"] for r in causes.values()) == pytest.approx(100 * 8 / 15)
    assert sum(r["primary_loss_pp"] for r in causes.values()) == pytest.approx(100 * 17 / 28)


def test_component_counts_include_correct_guards_and_retained_direction_errors():
    labels, mains, finals, _, _ = fixture_rows()
    result = transition_matrix(labels, mains, finals)
    assert result["corrected_weight"] == 11
    assert result["regressed_weight"] == 15
    assert result["net_primary_gain_pp"] == pytest.approx(-100 * 4 / 28)
    assert result["group_weight_changes"] == {"lost_true_positive_weight": 8, "removed_false_positive_weight": 11}
    assert sum(v["pairs"] for v in result["transitions"].values()) == 5


def test_oracle_is_per_cause_not_a_label_revision_or_additive_gain():
    labels, mains, finals, payloads, reviews = fixture_rows()
    original = deepcopy((labels, finals))
    result = oracle_sensitivity(labels, finals, review_ledger(labels, mains, finals, payloads, reviews))
    assert result["not_candidate_scores"]
    assert result["not_label_revision"]
    assert result["by_cause"]["ENGINEERING_ERROR"]["duplicate_recall"] == pytest.approx(12 / 15)
    assert (labels, finals) == original


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing_review", "BOTTLENECK_REVIEW_COVERAGE"),
        ("extra_review", "BOTTLENECK_REVIEW_COVERAGE"),
        ("missing_prediction", "BOTTLENECK_MEMBERSHIP"),
        ("unknown_span", "BOTTLENECK_REVIEW_EVIDENCE"),
        ("no_evidence", "BOTTLENECK_REVIEW_EVIDENCE"),
        ("unknown_cause", "BOTTLENECK_REVIEW_CAUSE"),
        ("complete_as_input_limit", "BOTTLENECK_INPUT_LIMIT"),
    ],
)
def test_review_fail_closed(mutation, code):
    labels, mains, finals, payloads, reviews = fixture_rows()
    if mutation == "missing_review":
        del reviews["H0"]
    elif mutation == "extra_review":
        reviews["H4"] = deepcopy(reviews["H0"])
    elif mutation == "missing_prediction":
        finals.pop()
    elif mutation == "unknown_span":
        reviews["H0"]["evidence_ids"] = ["MISSING"]
    elif mutation == "no_evidence":
        reviews["H0"]["evidence_ids"] = []
    elif mutation == "unknown_cause":
        reviews["H0"]["cause"] = "AUTOMATIC_GOLD_FIX"
    elif mutation == "complete_as_input_limit":
        reviews["H0"]["cause"] = "INPUT_LIMITATION"
    with pytest.raises(DedupEvaluationError, match=code):
        review_ledger(labels, mains, finals, payloads, reviews)


def test_input_limit_is_not_automatically_semantic_or_engineering_error():
    labels, mains, finals, payloads, reviews = fixture_rows()
    payloads[2]["payload"]["semantic_diff_evidence"]["status"] = "INCOMPLETE_LIMIT"
    reviews["H2"].update(cause="INPUT_LIMITATION", evidence_ids=[], locus="INPUT")
    ledger = review_ledger(labels, mains, finals, payloads, reviews)
    assert ledger[2]["cause"] == "INPUT_LIMITATION"
    assert ledger[4]["review_status"] == "NOT_REVIEWED"


def test_reversed_documents_normalize_directions_and_ignore_taxonomy_only():
    labels, _, _, payloads, _ = fixture_rows()
    labels, payloads = labels[:2], payloads[:2]
    for label in labels:
        label["human_same_duplicate_group"] = "YES"
    labels[0].update(human_a_can_replace_b="YES", human_b_can_replace_a="NO")
    labels[1].update(human_a_can_replace_b="NO", human_b_can_replace_a="YES")
    payloads[1]["payload"].update(document_a={"text": "B"}, document_b={"text": "A"})
    assert exact_conflicts(labels, payloads) == []
    labels[1].update(human_a_can_replace_b="YES", human_b_can_replace_a="YES")
    result = exact_conflicts(labels, payloads)
    assert len(result) == 1
    assert result[0]["has_both_orientations"]
    assert not result[0]["identical_visible_payload"]
    assert result[0]["minimum_error_weight_for_orientation_consistent_answer"] == 2


@pytest.mark.parametrize(("text", "truncated"), [(None, False), ("", False), ("A", True), ("A ", False)])
def test_missing_truncated_or_merely_similar_text_is_not_an_exact_conflict(text, truncated):
    labels, _, _, payloads, _ = fixture_rows()
    labels, payloads = labels[:2], payloads[:2]
    payloads[1]["payload"]["document_a"]["text"] = text
    payloads[1]["payload"]["long_document_evidence"]["truncated"] = truncated
    assert exact_conflicts(labels, payloads) == []


def test_mixed_panel_preserves_correct_guards_and_rejects_disputes_or_duplicate_text():
    labels, _, _, payloads, reviews = fixture_rows()
    for i, packet in enumerate(payloads):
        packet["payload"]["document_a"]["text"] = f"A{i}"
    spec = {"cases": [{"review_id": "H0", "note": "clear error"}, {"review_id": "H4", "note": "correct guard"}]}
    panel, manifest = mixed_panel(labels, payloads, spec, reviews)
    assert [r["review_id"] for r in panel] == ["H0", "H4"]
    assert manifest["independent_human_confirmed"] is False
    spec["cases"][1]["review_id"] = "H1"
    with pytest.raises(DedupEvaluationError, match="BOTTLENECK_PANEL_DISPUTE"):
        mixed_panel(labels, payloads, spec, reviews)
    spec["cases"][1]["review_id"] = "H4"
    payloads[4]["payload"] = deepcopy(payloads[0]["payload"])
    with pytest.raises(DedupEvaluationError, match="BOTTLENECK_PANEL_DUPLICATE"):
        mixed_panel(labels, payloads, spec, reviews)


def test_blind_export_has_no_predictions_labels_sampling_reasons_or_duplicate_ids():
    from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V1

    payload = {
        "payload_schema_version": VISIBLE_PAYLOAD_V1,
        "document_a": {"text": "hello"},
        "document_b": {"text": "world"},
        "long_document_evidence": {"truncated": False, "windows": []},
    }
    rows = [
        {"canonical_pair_id": pid, "payload": payload, "human_label": "YES", "sampling_reason": "test"}
        for pid in ("p0", "p1")
    ]
    blind, keys = blind_packets(rows, {"p0", "p1"})
    assert len({r["case_id"] for r in blind}) == 2
    assert all(set(r) == {"case_id", "payload"} for r in blind)
    assert {r["canonical_pair_id"] for r in keys} == {"p0", "p1"}
    assert blind_packets(list(reversed(rows)), {"p0", "p1"}) == (blind, keys)
    with pytest.raises(DedupEvaluationError, match="BOTTLENECK_BLIND_MEMBERSHIP"):
        blind_packets(rows, {"missing"})
    payload["human_label"] = "YES"
    with pytest.raises(DedupEvaluationError, match="BOTTLENECK_BLIND_FIELDS"):
        blind_packets(rows, {"p0"})
