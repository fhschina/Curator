from eval.dedup.judging.anchor_ids import normalize_shared_anchor_ids

PAYLOAD = {
    "document_a": {"text": "sameleft"},
    "document_b": {"text": "same"},
    "semantic_diff_evidence": {
        "spans": [
            {
                "span_id": "S001",
                "kind": "SHARED",
                "a_text": "same",
                "a_start_char": 0,
                "a_end_char": 4,
                "b_text": "same",
                "b_start_char": 0,
                "b_end_char": 4,
            },
            {"span_id": "A001", "kind": "A_ONLY", "side": "A", "text": "left", "start_char": 4, "end_char": 8},
        ]
    },
}


def test_normalizes_only_unambiguous_shared_anchor_spellings() -> None:
    value = {"shared_anchor_ids": ["001"]}
    normalized, corrections = normalize_shared_anchor_ids(PAYLOAD, value)

    assert normalized["shared_anchor_ids"] == ["S001"]
    assert corrections == [{"index": 0, "original": "001", "canonical": "S001"}]
    assert value == {"shared_anchor_ids": ["001"]}


def test_does_not_invent_or_deduplicate_anchor_ids() -> None:
    unknown = {"shared_anchor_ids": ["999"]}
    duplicate = {"shared_anchor_ids": ["S1", "S001"]}

    assert normalize_shared_anchor_ids(PAYLOAD, unknown) == (unknown, [])
    assert normalize_shared_anchor_ids(PAYLOAD, duplicate) == (duplicate, [])
