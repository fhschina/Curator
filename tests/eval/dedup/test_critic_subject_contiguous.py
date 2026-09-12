# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from eval.dedup.judging import critic_subject_contiguous as candidate
from eval.dedup.judging import critic_subject_scope as control
from eval.dedup.judging.payload import _semantic_diff_packet


def test_full_documents_and_original_locator_inventory_are_preserved():
    payload = {
        "document_a": {"text": "left source"},
        "document_b": {"text": "right source"},
        "semantic_diff_evidence": _semantic_diff_packet("left source", "right source", truncated=False),
    }
    expected = control.messages(payload, "unchanged system")
    actual = candidate.messages(payload, "unchanged system")
    assert actual[0] == expected[0]
    assert actual[1]["content"].endswith(expected[1]["content"])
    assert payload["document_a"]["text"] in actual[1]["content"]
    assert payload["document_b"]["text"] in actual[1]["content"]
    assert candidate.response_schema(payload) == control.response_schema(payload)
    assert candidate.apply_review is control.apply_review
    assert candidate.route is control.route
