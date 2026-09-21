from copy import deepcopy

import pytest

from eval.dedup.core.validation import DedupEvaluationError
from eval.dedup.judging import coverage
from eval.dedup.judging.anchor_ids import normalize_shared_anchor_ids

PAYLOAD = {
    "document_a": {"text": "sameleft"},
    "document_b": {"text": "same"},
    "long_document_evidence": {"windows": []},
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


def test_does_not_invent_or_merge_differently_spelled_anchor_ids() -> None:
    unknown = {"shared_anchor_ids": ["999"]}
    duplicate = {"shared_anchor_ids": ["S1", "S001"]}

    assert normalize_shared_anchor_ids(PAYLOAD, unknown) == (unknown, [])
    assert normalize_shared_anchor_ids(PAYLOAD, duplicate) == (duplicate, [])


def test_removes_only_exact_duplicates_preserving_evidence_and_first_occurrence() -> None:
    payload = deepcopy(PAYLOAD)
    payload["semantic_diff_evidence"]["spans"].append(
        {**payload["semantic_diff_evidence"]["spans"][0], "span_id": "S002"}
    )
    value = {"shared_anchor_ids": ["S002", "S001", "S002", "S001", "S002"], "explanation": "Keep this evidence."}
    original = deepcopy(value)

    normalized, corrections = normalize_shared_anchor_ids(payload, value)

    assert normalized == {**value, "shared_anchor_ids": ["S002", "S001"]}
    assert set(normalized["shared_anchor_ids"]) == set(value["shared_anchor_ids"])
    assert corrections == [
        {"index": index, "original": sid, "canonical": sid, "action": "remove_exact_duplicate"}
        for index, sid in ((2, "S002"), (3, "S001"), (4, "S002"))
    ]
    assert value == original
    assert normalize_shared_anchor_ids(payload, normalized) == (normalized, [])


@pytest.mark.parametrize(
    "anchors",
    [
        ["S999", "S999"],
        ["A001", "A001"],
        ["001", "001"],
        ["S001", "S001", "S1"],
        ["S001", "S001", "S999"],
        ["S001", "S001", "A001"],
    ],
)
def test_duplicate_recovery_rejects_any_invalid_or_nonshared_reference(anchors: list[str]) -> None:
    value = {"shared_anchor_ids": anchors}

    assert normalize_shared_anchor_ids(PAYLOAD, value) == (value, [])


def test_exact_duplicate_recovery_satisfies_strict_coverage_validation() -> None:
    value = {
        "a_loss_span_id": "",
        "b_loss_span_id": "",
        "a_context_span_id": "S001",
        "b_context_span_id": "S001",
        "conflict": "NONE",
        "overlap_basis": "RETAINED_CONTENT",
        "shared_anchor_ids": ["S001", "S001"],
        "explanation": "The same shared evidence was listed twice.",
    }
    proof = coverage._proof(coverage._compile_review(value, PAYLOAD))
    with pytest.raises(DedupEvaluationError, match="CRITIC_SCOPE_ANCHOR"):
        coverage.veto.validate_review(proof, PAYLOAD)

    normalized, _ = normalize_shared_anchor_ids(PAYLOAD, value)
    proof = coverage._proof(coverage._compile_review(normalized, PAYLOAD))
    evidence = coverage.veto.validate_review(proof, PAYLOAD)

    assert {item["side"] for item in evidence} == {"A", "B"}
