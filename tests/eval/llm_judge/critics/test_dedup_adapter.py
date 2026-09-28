# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from nemo_curator.eval.llm_judge.critics.dedup_adapter import adapt_alignment, complete_text_equality, normalize_main


def test_core_evidence_is_separate_from_context(pair: dict[str, Any]) -> None:
    original = deepcopy(pair)
    packet = adapt_alignment(pair)
    shared, a, b = packet["spans"]
    assert shared == pair["semantic_diff"]["spans"][0]
    assert (a["text"], a["start_char"], a["end_char"], a["side"]) == ("apples", 8, 14, "A")
    assert (b["text"], b["start_char"], b["end_char"], b["side"]) == ("pears", 8, 13, "B")
    assert a["context_text"] == pair["text_a"]
    assert pair == original


@pytest.mark.parametrize("defect", ["missing_delta", "outside_delta", "wrong_text", "duplicate_id", "wrong_flag"])
def test_invalid_alignment_is_not_repaired(pair: dict[str, Any], defect: str) -> None:
    spans = pair["semantic_diff"]["spans"]
    if defect == "missing_delta":
        del spans[1]["delta_start_char"]
    elif defect == "outside_delta":
        spans[1]["delta_end_char"] = 100
    elif defect == "wrong_text":
        spans[0]["a_text"] = "invented"
    elif defect == "duplicate_id":
        spans[2]["span_id"] = "A001"
    else:
        pair["truncated"] = "false"
    with pytest.raises(ValueError, match="pair-1"):
        adapt_alignment(pair)


def test_complete_does_not_imply_untruncated_or_raw_equality(pair: dict[str, Any]) -> None:
    pair["text_b"] = pair["text_a"]
    packet = {"status": "COMPLETE", "truncated": True}
    assert not complete_text_equality(pair, packet)
    packet["truncated"] = False
    assert complete_text_equality(pair, packet)
    pair["text_b"] = pair["text_a"].upper()
    assert not complete_text_equality(pair, packet)


def test_main_is_unwrapped_without_rewriting_reasoning(pair: dict[str, Any]) -> None:
    original = deepcopy(pair)
    result = normalize_main(pair, "pair_semantic_judgment")
    assert result["a_can_replace_b"] == "yes"
    assert result["relation_type"] == "near_surface"
    assert pair == original
    pair["pair_semantic_judgment"]["a_can_replace_b"]["score"] = "no"
    with pytest.raises(ValueError, match=r"pair-1.*disagree"):
        normalize_main(pair, "pair_semantic_judgment")
