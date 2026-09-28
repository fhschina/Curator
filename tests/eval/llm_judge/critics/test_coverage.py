# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from nemo_curator.eval.llm_judge.critics.coverage import CoverageCritic


@pytest.mark.parametrize(
    ("a_loss", "b_loss", "conflict", "overlap", "action", "directions"),
    [
        ("", "", "NONE", "RETAINED_CONTENT", "KEEP_MAIN", ("yes", "yes")),
        ("A001", "", "NONE", "RETAINED_CONTENT", "REJECT_B_REPLACES_A", ("yes", "no")),
        ("", "B001", "NONE", "RETAINED_CONTENT", "REJECT_A_REPLACES_B", ("no", "yes")),
        ("A001", "B001", "NONE", "RETAINED_CONTENT", "REJECT_BOTH", ("no", "no")),
        ("", "", "IDENTITY_CONFLICT", "RETAINED_CONTENT", "REJECT_BOTH", ("no", "no")),
        ("", "", "NONE", "UNCERTAIN", "ABSTAIN", ("unresolved", "unresolved")),
        ("A001", "", "NONE", "INTERFACE_ONLY", "KEEP_MAIN", ("yes", "yes")),
    ],
)
def test_review_applies_only_supported_directions(  # noqa: PLR0913
    pair: dict[str, Any],
    review: dict[str, Any],
    a_loss: str,
    b_loss: str,
    conflict: str,
    overlap: str,
    action: str,
    directions: tuple[str, str],
) -> None:
    critic = CoverageCritic("pair_semantic_judgment")
    original = deepcopy(pair)
    review.update(a_loss_span_id=a_loss, b_loss_span_id=b_loss, conflict=conflict, overlap_basis=overlap)
    output = critic.apply({**pair, **critic.prepare(pair), "coverage_review": review})
    assert output["coverage_action"] == action
    final = output["final_decision"]
    assert (final["a_can_replace_b"], final["b_can_replace_a"]) == directions
    if action.startswith("REJECT"):
        assert final["material_difference"] == "major"
        assert final["relation_type"] == ("containment" if "yes" in directions else "related_non_duplicate")
    for evidence in output["coverage_evidence"]:
        assert (
            pair[f"text_{evidence['side'].lower()}"][evidence["start_char"] : evidence["end_char"]]
            == evidence["quote"]
        )
    assert pair == original


@pytest.mark.parametrize(
    "defect", ["wrong_side", "shared_loss", "unknown_id", "truncated", "missing_anchor", "empty_result"]
)
def test_invalid_review_fails_instead_of_skipping(pair: dict[str, Any], review: dict[str, Any], defect: str) -> None:
    critic = CoverageCritic("pair_semantic_judgment")
    review["a_loss_span_id"] = "A001"
    if defect == "wrong_side":
        review["a_loss_span_id"] = "B001"
    elif defect == "shared_loss":
        review["a_loss_span_id"] = "S001"
    elif defect == "unknown_id":
        review["b_context_span_id"] = "B999"
    elif defect == "truncated":
        pair["truncated"] = True
        pair["semantic_diff"].update(truncated=True, truncated_a=True)
    elif defect == "missing_anchor":
        review.update(overlap_basis="INTERFACE_ONLY", shared_anchor_ids=[])
    prepared = critic.prepare(pair)
    with pytest.raises(ValueError, match="pair-1"):
        critic.apply({**pair, **prepared, "coverage_review": None if defect == "empty_result" else review})


def test_empty_anchor_only_vetoes_containment_and_does_not_reopen_no(
    pair: dict[str, Any],
    review: dict[str, Any],
) -> None:
    summary = pair["pair_semantic_judgment"]
    for key, value in {
        "b_can_replace_a": "no",
        "relation_type": "containment",
        "material_difference": "major",
        "primary_material_difference": "main_content_addition_deletion",
    }.items():
        summary[key]["score"] = value
    critic = CoverageCritic("pair_semantic_judgment")
    review.update(a_loss_span_id="A001")
    prepared = {**pair, **critic.prepare(pair), "coverage_review": review}
    output = critic.apply(prepared)
    assert output["coverage_action"] == "KEEP_MAIN"
    assert output["coverage_reason"] == "OBJECTION_ONLY_TO_ALREADY_UNSAFE_DIRECTION"
    review["overlap_basis"] = "INTERFACE_ONLY"
    output = critic.apply(prepared)
    assert output["coverage_action"] == "REJECT_EMPTY_ANCHOR"
    assert output["final_decision"]["a_can_replace_b"] == output["final_decision"]["b_can_replace_a"] == "no"


@pytest.mark.parametrize("relation", ["related_non_duplicate", "unresolved", "exact"])
def test_routed_rows_keep_main_without_review(pair: dict[str, Any], relation: str) -> None:
    summary = pair["pair_semantic_judgment"]
    if relation == "related_non_duplicate":
        values = {
            "a_can_replace_b": "no",
            "b_can_replace_a": "no",
            "relation_type": relation,
            "material_difference": "major",
            "primary_material_difference": "other_material",
        }
    elif relation == "unresolved":
        values = dict.fromkeys(summary, "unresolved")
        values.update(primary_risk_factor="extraction_or_payload_limit", confidence_tier="low")
    else:
        values = {"relation_type": "exact"}
        pair["text_b"] = pair["text_a"]
        pair["semantic_diff"]["spans"] = [pair["semantic_diff"]["spans"][0]]
        pair["semantic_diff"]["span_counts"].update(A_ONLY=0, B_ONLY=0)
    for key, value in values.items():
        summary[key]["score"] = value
    critic = CoverageCritic("pair_semantic_judgment")
    prepared = critic.prepare(pair)
    assert prepared["coverage_should_run"] is False
    result = critic.apply({**pair, **prepared, "coverage_review": None})
    assert result["coverage_action"] == "SKIP"
    assert result["final_decision"] == {key: value["score"] for key, value in summary.items()}


def test_native_structured_column_carries_skip() -> None:
    import data_designer.config as dd

    column = CoverageCritic("pair_semantic_judgment").build_column(
        model_alias="judge",
        prompt="{{ _coverage_payload }}",
        system_prompt="Audit evidence.",
    )
    assert isinstance(column, dd.LLMStructuredColumnConfig)
    assert column.skip.when == "{{ not coverage_should_run }}"
    assert column.skip.value is None
    assert column.output_format["properties"]["a_loss_span_id"]["type"] == "string"
