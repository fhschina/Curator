# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import critic_anchor_ids_v2 as subject
from eval.dedup.judging import critic_retention_v4 as coverage
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_critic_retention_v3 import fixture


def test_explicit_shared_field_recovers_prefix_without_changing_other_fields():
    payload, main, review = fixture()
    review["shared_anchor_ids"] = ["001"]
    before = deepcopy((payload, main, review))
    fixed, changes = subject.normalize_shared_anchor_ids(payload, review)
    assert changes == [{"index": 0, "original": "001", "canonical": "S001"}]
    expected = {**review, "shared_anchor_ids": ["S001"]}
    assert fixed == expected
    assert coverage.apply_review(main, payload, fixed) == coverage.apply_review(main, payload, expected)
    assert before == (payload, main, review)
    assert subject.normalize_shared_anchor_ids(payload, fixed) == (fixed, [])


@pytest.mark.parametrize(
    "anchors",
    [
        ["1"],
        ["01"],
        ["0001"],
        ["\uff10\uff10\uff11"],
        [" 001"],
        ["999"],
        ["001", "S001"],
        ["001", "001"],
        ["001", "A001"],
        ["001", None],
    ],
)
def test_unknown_ambiguous_wrong_kind_and_loose_numbers_are_not_repaired(anchors):
    payload, main, review = fixture()
    review["shared_anchor_ids"] = anchors
    assert subject.normalize_shared_anchor_ids(payload, review) == (review, [])
    with pytest.raises(DedupEvaluationError):
        coverage.apply_review(main, payload, review)


@pytest.mark.parametrize("damage", ["duplicate", "alignment", "kind"])
def test_prefix_repair_still_requires_valid_bilateral_inventory(damage):
    payload, _, review = fixture()
    review["shared_anchor_ids"] = ["001"]
    spans = payload["semantic_diff_evidence"]["spans"]
    shared = next(s for s in spans if s["kind"] == "SHARED")
    if damage == "duplicate":
        spans.append(deepcopy(shared))
    elif damage == "alignment":
        shared["a_start_char"] += 1
    else:
        shared.update(kind="A_ONLY", side="A")
    with pytest.raises(DedupEvaluationError):
        subject.normalize_shared_anchor_ids(payload, review)


def test_old_zero_padding_and_context_loss_fields_are_unchanged():
    payload, _, review = fixture()
    review.update(shared_anchor_ids=["S1"], a_context_span_id="001")
    assert subject.normalize_shared_anchor_ids(payload, review) == subject.previous.normalize_shared_anchor_ids(
        payload, review
    )
    review["shared_anchor_ids"] = ["001"]
    fixed, _ = subject.normalize_shared_anchor_ids(payload, review)
    assert fixed["a_context_span_id"] == "001"
