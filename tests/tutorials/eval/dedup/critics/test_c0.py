# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pyarrow as pa
import pytest
from critics.c0 import C0AuditStage, main_errors
from critics.models import Decision

from nemo_curator.tasks import DocumentBatch


def test_covered_service_slots_remain_valid_without_rewriting(c0_record: dict) -> None:
    before = copy.deepcopy(c0_record)
    assert main_errors(c0_record, "pair_semantic_judgment", 240) == []
    assert c0_record == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("span_b_delta", "same_record_content_extension"),
        ("span_a_delta", "material_non_main_message_change"),
        ("span_hard_conflict", "legal_context"),
        ("span_shared_basis", "none"),
    ],
)
def test_literal_ledger_contradictions_are_invalid(c0_record: dict, field: str, value: str) -> None:
    c0_record["pair_semantic_judgment"][field]["score"] = value
    assert main_errors(c0_record, "pair_semantic_judgment", 240)


def _negative(record: dict) -> None:
    raw = record["pair_semantic_judgment"]
    updates = {
        "a_can_replace_b": "no",
        "b_can_replace_a": "no",
        "relation_type": "related_non_duplicate",
        "material_difference": "major",
        "primary_material_difference": "other_material",
        "span_translation_status": "complete_faithful",
    }
    for key, value in updates.items():
        raw[key]["score"] = value
    raw["a_can_replace_b"]["reasoning"] = 'B001: "City Beta" is the removed-side phrase.'
    raw["b_can_replace_a"]["reasoning"] = 'A001: "City Alpha" is the removed-side phrase.'


def test_translation_negatives_are_not_inferred_harmless_from_coverage(c0_record: dict) -> None:
    _negative(c0_record)
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)
    c0_record["pair_semantic_judgment"]["span_translation_status"]["score"] = "not_translation"
    assert any("covered removed-side" in issue for issue in main_errors(c0_record, "pair_semantic_judgment", 240))


@pytest.mark.parametrize(
    "reasoning",
    [
        'A001: "City Alpha" is missing.',
        'B999: "City Beta" is missing.',
        'B001: "a fabricated obligation" is missing.',
        'B001: "Service hours" is missing.',
        'B001: "   " is missing.',
        'S001: "Service hours" is missing.',
    ],
)
def test_no_requires_existing_removed_side_citation_and_literal_quote(c0_record: dict, reasoning: str) -> None:
    _negative(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = reasoning
    assert main_errors(c0_record, "pair_semantic_judgment", 240)


def test_complete_shared_quote_must_identify_removed_side(c0_record: dict) -> None:
    _negative(c0_record)
    c0_record["pair_semantic_judgment"]["a_can_replace_b"]["reasoning"] = 'S001 (B): "Service hours" is lost.'
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)


def test_incomplete_packet_requires_fully_unresolved_summary(c0_record: dict) -> None:
    c0_record["truncated"] = True
    c0_record["semantic_diff"].update(truncated=True, truncated_a=True, status="INCOMPLETE_LIMIT")
    assert main_errors(c0_record, "pair_semantic_judgment", 240)
    raw = c0_record["pair_semantic_judgment"]
    for field in Decision.model_fields:
        raw[field]["score"] = "unresolved"
    raw["confidence_tier"]["score"] = "low"
    raw["primary_risk_factor"]["score"] = "extraction_or_payload_limit"
    assert not main_errors(c0_record, "pair_semantic_judgment", 240)


def test_missing_ledger_and_cross_field_enums_are_invalid(c0_record: dict) -> None:
    raw = c0_record["pair_semantic_judgment"]
    del raw["span_shared_basis"]
    raw["primary_material_difference"]["score"] = "state_or_version"
    errors = main_errors(c0_record, "pair_semantic_judgment", 240)
    assert any("span_shared_basis" in error for error in errors)
    assert any("primary_material_difference" in error for error in errors)


def test_audit_retains_rows_raw_answers_and_all_layer_validity(c0_record: dict) -> None:
    decision = Decision.from_judge(c0_record["pair_semantic_judgment"]).model_dump()
    c0_record.update(coverage_decision=decision, final_decision=decision, coverage_action="KEEP_MAIN")
    bad = copy.deepcopy(c0_record)
    bad["pair_id"] = "invalid"
    bad["pair_semantic_judgment"]["span_b_delta"]["score"] = "same_record_content_extension"
    missing = copy.deepcopy(c0_record)
    missing.update(pair_id="missing", pair_semantic_judgment=None, coverage_decision=None, final_decision=None)
    table = pa.Table.from_pylist([c0_record, bad, missing])
    output = C0AuditStage().process(DocumentBatch(dataset_name="audit", data=table)).to_pyarrow()
    assert output.select(table.column_names).to_pylist() == table.to_pylist()
    assert [row["c0_final_status"] for row in output.to_pylist()] == ["valid", "invalid", "invalid"]
    assert all(
        output.column(f"c0_{layer}_status").to_pylist() == ["valid", "invalid", "invalid"]
        for layer in ("main", "coverage", "final")
    )


def test_unanswered_critic_is_invalid_without_erasing_main(c0_record: dict) -> None:
    decision = Decision.from_judge(c0_record["pair_semantic_judgment"]).model_dump()
    c0_record.update(coverage_decision=decision, final_decision=decision, coverage_action="UNVALIDATED_KEEP_MAIN")
    result = C0AuditStage()._audit(c0_record)
    assert result["c0_main_status"] == "valid"
    assert result["c0_coverage_status"] == result["c0_final_status"] == "invalid"
