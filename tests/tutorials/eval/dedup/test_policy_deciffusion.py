# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path

import pytest
from policy_cases import aligned, controlled
from policy_deciffusion import (
    FIELDS,
    model_answers,
    parse_response,
    request_body,
    require_qwen_acceptance,
    validated_choices,
)
from policy_eval import write_json


def response():
    return {"answers": {field: {"probabilities": {"YES": 0.5, "NO": 0.5, "UNRESOLVED": 0.0}} for field in FIELDS}}


def test_native_tied_probability_is_unresolved():
    assert set(parse_response(response()).values()) == {"unresolved"}


def test_native_conclusions_with_truncated_evidence_cannot_allow_deletion():
    raw = {"status": "valid", "a_can_replace_b": "yes", "b_can_replace_a": "no"}
    valid = validated_choices(raw, {"truncated": True})
    assert valid["validation_status"] == "invalid"
    assert valid["validation_errors"] == ["CONCLUSIVE_INCOMPLETE_EVIDENCE"]
    assert all(valid[field] == "invalid" for field in FIELDS)
    assert raw["a_can_replace_b"] == "yes"


def test_model_comparison_uses_distinct_native_main_and_final_decisions():
    row = {
        "validated": {"a_can_replace_b": "yes"},
        "qwen_main": {"a_can_replace_b": "yes"},
        "qwen_final": {"a_can_replace_b": "no"},
    }
    assert model_answers(row, "deciffusion")["a_can_replace_b"] == "yes"
    assert model_answers(row, "qwen_main")["a_can_replace_b"] == "yes"
    assert model_answers(row, "qwen_final")["a_can_replace_b"] == "no"


def test_stale_acceptance_flag_cannot_bypass_qwen_validation(tmp_path: Path):
    write_json(tmp_path / "manifest.json", {})
    write_json(tmp_path / "acceptance.json", {"passed": True})
    with pytest.raises(ValueError, match="Qwen human acceptance"):
        require_qwen_acceptance(tmp_path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, True, 1.1])
def test_invalid_native_probabilities_are_not_decisions(bad: object):
    value = copy.deepcopy(response())
    value["answers"][FIELDS[0]]["probabilities"]["YES"] = bad
    with pytest.raises(ValueError, match="finite"):
        parse_response(value)


def test_choice_input_has_only_evidence_not_qwen_protocol_or_labels():
    row = aligned(controlled()[:1])[0]
    body = request_body(
        row,
        "Retention policy\nSPAN PACKET AND LEDGER\nHIDDEN_QWEN_PROTOCOL",
        '<semantic_diff status="COMPLETE">{{ semantic_diff }}</semantic_diff>',
    )
    assert "HIDDEN_QWEN_PROTOCOL" not in str(body)
    assert "a_can_replace_b" not in body["state"]["visible_material"]
    assert body["questions"][FIELDS[0]]["type"] == "choice"
