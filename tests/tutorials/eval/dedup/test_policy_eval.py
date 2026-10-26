# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path

import pytest
from policy_cases import aligned, controlled
from policy_eval import audit, exact_index, metrics, review_page, stage_effects, verify_files


def example():
    source = aligned(controlled()[:1])[0]
    value = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
    }
    row = {
        **source,
        "pair_semantic_judgment": {k: {"score": v, "reasoning": "Covered"} for k, v in value.items()},
        "coverage_decision": value.copy(),
        "final_decision": value.copy(),
        "coverage_action": "KEEP_MAIN",
        "subject_action": "KEEP_COVERAGE",
    }
    return source, row


def test_duplicate_or_missing_output_ids_are_fatal():
    with pytest.raises(ValueError, match="Duplicate"):
        exact_index([{"pair_id": "a"}, {"pair_id": "a"}])
    with pytest.raises(ValueError, match="mismatch"):
        exact_index([{"pair_id": "a"}], {"a", "b"})


def test_invalid_critic_cannot_count_as_a_passing_case():
    source, row = example()
    row["coverage_action"] = "UNVALIDATED_KEEP_MAIN"
    row["coverage_reason"] = "INVALID_CRITIC_OUTPUT: bad span"
    report = metrics([row], [source], [source])
    assert not report["gate_passed"]
    assert "COVERAGE_UNVALIDATED" in report["issues"][row["pair_id"]]


def test_critic_no_to_yes_is_detected():
    source, row = example()
    row["coverage_decision"].update(
        a_can_replace_b="no",
        b_can_replace_a="no",
        relation_type="related_non_duplicate",
        material_difference="major",
        primary_material_difference="other_material",
    )
    assert "CRITIC_PROMOTED_NEGATIVE:final" in audit(row, source)


def test_main_negative_requires_evidence_from_the_removed_side():
    source = aligned(controlled()[1:2])[0]
    _, row = example()
    row = {**row, **source}
    for field in ("a_can_replace_b", "b_can_replace_a"):
        row["pair_semantic_judgment"][field] = {"score": "no", "reasoning": "A001"}
    row["pair_semantic_judgment"]["relation_type"]["score"] = "related_non_duplicate"
    row["pair_semantic_judgment"]["material_difference"]["score"] = "major"
    row["pair_semantic_judgment"]["primary_material_difference"]["score"] = "other_material"
    assert "MISSING_MAIN_LOSS_EVIDENCE:a_can_replace_b" in audit(row, source)
    row["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = "B001"
    assert not any(issue.startswith("MISSING_MAIN_LOSS_EVIDENCE") for issue in audit(row, source))


def test_changed_or_truncated_input_cannot_pass_validation():
    source, row = example()
    changed = copy.deepcopy(source)
    changed["truncated"] = True
    changed["semantic_diff"].update(truncated=True, truncated_a=True)
    assert "CONCLUSIVE_INCOMPLETE_EVIDENCE" in audit(row, changed)


def test_provisional_and_unresolved_human_labels_are_not_gold():
    source, row = example()
    label = {**source, "label_status": "provisional"}
    report = metrics([row], [label], [source])
    assert report["layers"]["final"]["scored_pairs"] == 0
    assert not report["gate_passed"]


def test_agent_agreement_never_satisfies_independent_human_acceptance():
    source, row = example()
    inputs = [{**source, "pair_id": str(i)} for i in range(80)]
    outputs = [{**row, "pair_id": str(i)} for i in range(80)]
    labels = [{**r, "label_status": "agent_assisted_blind"} for r in inputs]
    report = metrics(outputs, labels, inputs)
    assert report["layers"]["final"]["direction_accuracy"] == 1
    assert report["label_sources"] == {"agent_assisted_blind": 80}
    assert not report["holdout_passed"]
    for label in labels:
        label["label_status"] = "human_independent"
    assert metrics(outputs, labels, inputs)["holdout_passed"]


def test_diagnostic_reference_agreement_preserves_provisional_status_and_excludes_truncation():
    source, row = example()
    labels = [{**source, "label_status": "provisional"}]
    effects = stage_effects([row], labels, [source])
    assert len(effects) == 3
    assert all(e["label_status"] == "provisional" and e["reference_agrees"] for e in effects)
    source["truncated"] = True
    assert all(e["reference_agrees"] is None for e in stage_effects([row], labels, [source]))


def test_frozen_artifact_edit_is_rejected(tmp_path: Path):
    (tmp_path / "input").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_files(tmp_path, {"input": "wrong"})


def test_blind_review_contains_no_model_answers_and_escapes_documents():
    page = review_page(
        [
            {
                "pair_id": "H001",
                "text_a": "<script>attack()</script>",
                "text_b": "B",
                "a_can_replace_b": "no",
                "historical_reason": "HIDDEN_MODEL_REASON",
            }
        ]
    )
    assert "&lt;script&gt;attack()&lt;/script&gt;" in page
    assert "HIDDEN_MODEL_REASON" not in page
