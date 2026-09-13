# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import critic_anchor_ids as subject
from eval.dedup.judging import critic_retention_v4 as coverage
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_critic_retention_v3 import fixture


@pytest.mark.parametrize(("canonical", "short"), [("S041", "S41"), ("S021", "S21"), ("S001", "S1"), ("S001", "S01")])
def test_missing_zeros_preserve_evidence_and_decisions(canonical, short):
    payload, main, review = fixture()
    for span in payload["semantic_diff_evidence"]["spans"]:
        if span["span_id"] == "S001":
            span["span_id"] = canonical
    for field in ("a_context_span_id", "b_context_span_id"):
        if review[field] == "S001":
            review[field] = canonical
    review.update(shared_anchor_ids=[canonical], b_loss_span_id="B001")
    expected = coverage.apply_review(main, payload, review)
    raw = {**review, "shared_anchor_ids": [short]}
    saved = deepcopy((payload, main, raw))
    with pytest.raises(DedupEvaluationError, match="CRITIC_SCOPE_ANCHOR"):
        coverage.apply_review(main, payload, raw)
    normalized, changes = subject.normalize_shared_anchor_ids(payload, raw)
    assert normalized == review
    assert changes == [{"index": 0, "original": short, "canonical": canonical}]
    assert coverage.apply_review(main, payload, normalized) == expected
    assert saved == (payload, main, raw)
    assert subject.normalize_shared_anchor_ids(payload, normalized) == (normalized, [])


@pytest.mark.parametrize(
    "anchors",
    [
        ["S999"],
        ["S99"],
        ["A1"],
        ["A001"],
        ["s1"],
        [" S1"],
        ["S1 "],
        ["S0001"],
        ["S+1"],
        ["S1.0"],
        ["S\uff11"],
        ["S\u0661"],
        ["1"],
        ["S"],
        [""],
        ["S1", "S1"],
        ["S1", "S001"],
        ["S1", "S01"],
        ["S1", "S999"],
        ["S1", "A001"],
        [None],
        [1],
        [True],
        [["S1"]],
        "S1",
        None,
    ],
)
def test_unknown_ambiguous_duplicate_wrong_kind_or_malformed_ids_stay_rejected(anchors):
    payload, main, review = fixture()
    review["shared_anchor_ids"] = anchors
    original = deepcopy(review)
    normalized, changes = subject.normalize_shared_anchor_ids(payload, review)
    assert normalized == original == review
    assert changes == []
    with pytest.raises(DedupEvaluationError):
        coverage.apply_review(main, payload, normalized)


@pytest.mark.parametrize("anchors", [[], ["S001"]])
def test_valid_anchor_lists_are_unchanged(anchors):
    payload, main, review = fixture()
    review["shared_anchor_ids"] = anchors
    assert subject.normalize_shared_anchor_ids(payload, review) == (review, [])
    normalized, _ = subject.normalize_shared_anchor_ids(payload, review)
    assert coverage.apply_review(main, payload, normalized) == coverage.apply_review(main, payload, review)


@pytest.mark.parametrize("damage", ["duplicate_inventory_id", "ambiguous_spelling", "wrong_kind", "misaligned_quote"])
def test_invalid_inventory_cannot_enable_repairs(damage):
    payload, _, review = fixture()
    review["shared_anchor_ids"] = ["S1"]
    spans = payload["semantic_diff_evidence"]["spans"]
    shared = next(s for s in spans if s["kind"] == "SHARED")
    if damage == "duplicate_inventory_id":
        spans.append(deepcopy(shared))
    elif damage == "ambiguous_spelling":
        spans.append({**shared, "span_id": "S01"})
    elif damage == "wrong_kind":
        shared.update(kind="A_ONLY", side="A")
    else:
        shared["a_start_char"] += 1
    with pytest.raises(DedupEvaluationError, match="FOLLOWUP_BOUNDARY_SELECTION_PACKET"):
        subject.normalize_shared_anchor_ids(payload, review)


@pytest.mark.parametrize(("field", "value"), [("a_context_span_id", "S1"), ("b_loss_span_id", "B1")])
def test_context_and_loss_references_are_outside_repair_scope(field, value):
    payload, main, review = fixture()
    review.update(shared_anchor_ids=["S1"], **{field: value})
    normalized, changes = subject.normalize_shared_anchor_ids(payload, review)
    assert len(changes) == 1
    assert normalized[field] == value
    with pytest.raises(DedupEvaluationError, match="SELECTION_REFERENCE_INVALID"):
        coverage.apply_review(main, payload, normalized)


def test_anchor_order_and_non_anchor_fields_are_preserved():
    payload, _, review = fixture()
    shared = next(s for s in payload["semantic_diff_evidence"]["spans"] if s["kind"] == "SHARED")
    payload["semantic_diff_evidence"]["spans"].append({**shared, "span_id": "S021"})
    review["shared_anchor_ids"] = ["S21", "S1"]
    normalized, changes = subject.normalize_shared_anchor_ids(payload, review)
    assert normalized["shared_anchor_ids"] == ["S021", "S001"]
    assert changes == [
        {"index": 0, "original": "S21", "canonical": "S021"},
        {"index": 1, "original": "S1", "canonical": "S001"},
    ]
    assert {k: v for k, v in normalized.items() if k != "shared_anchor_ids"} == {
        k: v for k, v in review.items() if k != "shared_anchor_ids"
    }
