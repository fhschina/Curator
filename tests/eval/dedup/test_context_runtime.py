# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import context_runtime as subject
from eval.dedup.judging.composite_format_runtime import build_format_builder, format_blocks
from eval.dedup.judging.context_coverage import context_schema
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_context_experiment import fixture


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_same_model_and_payload_with_consistent_new_system_pair_and_yaml_rules(stage):
    old, old_providers = build_format_builder(stage=stage, endpoint="http://127.0.0.1:1/v1")
    new, new_providers = subject.build_context_builder(stage=stage, endpoint="http://127.0.0.1:1/v1")
    assert new.model_configs == old.model_configs
    assert new_providers == old_providers
    column = new.get_column_config(COLUMN)
    assert column.output_format == context_schema()
    assert subject.OLD_WITNESS_RULE not in column.system_prompt
    assert subject.VERSION in column.system_prompt
    assert subject.BASE_VERSION not in column.system_prompt
    cfg = subject.load_context_config()
    for key in ("context_witness_rule", "translation_rule"):
        assert cfg[key] in column.system_prompt
        assert key in column.prompt
    for packet in fixture(stage)[0]:
        if stage == "critic" and "proposed_main_text" not in packet:
            continue
        snapshot = deepcopy(packet)
        record = subject.context_record(packet, stage)
        messages = subject.context_renderer(stage=stage)(packet)
        assert record["payload"] == packet["payload"]
        assert "<context_contract_rubric>" in messages[1]["content"]
        assert packet == snapshot


def test_old_main_claim_and_prediction_leaks_cannot_enter_new_critic():
    packet = deepcopy(fixture("critic")[0][0])
    packet["proposed_main_text"] = packet["proposed_main_text"].replace("Validated V5", "Validated V4")
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_MAIN_MISSING"):
        subject.context_record(packet, "critic")
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_MAIN_LEAK"):
        subject.context_record(packet, "main")


def test_branch_encoding_stays_exact_and_has_no_inactive_field_defaults():
    block, _ = format_blocks()
    for branch, definition in context_schema()["$defs"].items():
        status = branch.removesuffix("Side").upper()
        assert f"{status}: exactly these {len(definition['required'])} keys:" in block
        assert definition["additionalProperties"] is False
