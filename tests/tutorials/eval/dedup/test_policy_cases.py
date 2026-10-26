# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from critics.spans import SpanPacket
from policy_cases import aligned, blind_row, controlled, mirrored, select_holdout, text_pair_key


def test_swapping_regenerates_evidence_and_preserves_direction():
    row = controlled()[-2]
    original, swapped = aligned([row, mirrored(row)])
    assert swapped["a_can_replace_b"] == original["b_can_replace_a"]
    assert swapped["b_can_replace_a"] == original["a_can_replace_b"]
    for value in (original, swapped):
        packet = SpanPacket.from_record(value)
        assert packet.complete
        for span in packet.spans.values():
            for side in ("A", "B"):
                if excerpt := span.excerpt(side):
                    assert value["text_" + side.lower()][excerpt.start_char : excerpt.end_char] == excerpt.text


def test_blind_payload_excludes_answers_and_metadata():
    row = {**aligned(controlled()[:1])[0], "keeper_direction": "a_can_replace_b", "source": "secret-metadata"}
    visible = blind_row(row)
    assert set(visible) == {"pair_id", "text_a", "text_b", "semantic_diff", "truncated"}


def test_truncated_evidence_is_explicit_and_never_complete():
    row = aligned([{"pair_id": "long", "text_a": "a " * 4000, "text_b": "a " * 4000}])[0]
    assert row["truncated"]
    assert not SpanPacket.from_record(row).complete


def test_text_pair_deduplication_ignores_order_and_surface_normalization():
    assert text_pair_key({"text_a": " A  b ", "text_b": "C"}) == text_pair_key({"text_a": "c", "text_b": "a b"})


def test_holdout_is_reproducible_and_excludes_shared_docs_groups_and_texts():
    candidates = []
    for i in range(140):
        candidates.append(
            {
                "pair_id": f"p{i:03d}",
                "pair_type": "keeper_removed" if i < 70 else "cross_group_sample",
                "doc_id_a": f"a{i}",
                "doc_id_b": f"b{i}",
                "group_id_a": f"g{i}",
                "group_id_b": None,
                "text_a": f"Article A {i}",
                "text_b": f"Article B {i}",
            }
        )
    excluded = [
        candidates[0],
        {"doc_id_a": "a1"},
        {"group_id_a": "g2"},
        {"text_a": "Article B 3", "text_b": "Article A 3"},
        {"text_a": "  ARTICLE A 4  "},
    ]
    selected = select_holdout(candidates, excluded)
    assert selected == select_holdout(candidates, excluded)
    assert len(selected) == 100
    assert not {"p000", "p001", "p002", "p003", "p004"} & {row["pair_id"] for row in selected}


def test_insufficient_disjoint_holdout_does_not_silently_shrink():
    with pytest.raises(ValueError, match="Insufficient disjoint"):
        select_holdout([], [])
