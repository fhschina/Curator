# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import re
from copy import deepcopy

import pytest

from eval.dedup.judging import directional_runtime as subject
from eval.dedup.judging.context_coverage import context_schema
from eval.dedup.judging.context_runtime import build_context_builder, context_renderer
from eval.dedup.judging.coverage_selection import COLUMN
from tests.eval.dedup.test_context_experiment import fixture


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_same_schema_model_semantics_payload_with_mechanical_owned_id_guide(stage):
    from data_designer.engine.validation import validate_prompt_templates

    old, old_providers = build_context_builder(stage=stage, endpoint="http://127.0.0.1:1/v1")
    new, providers = subject.build_directional_builder(stage=stage, endpoint="http://127.0.0.1:1/v1")
    assert old.model_configs == new.model_configs
    assert old_providers == providers
    assert not validate_prompt_templates(
        columns=[new.get_column_config(COLUMN)],
        allowed_references=["payload", "repair_feedback", "proposed_main_text"],
    )
    assert (
        old.get_column_config(COLUMN).output_format == new.get_column_config(COLUMN).output_format == context_schema()
    )
    config = subject.load_directional_config()
    old_render, render = context_renderer(stage=stage), subject.directional_renderer(stage=stage)
    for packet in fixture(stage)[0]:
        if stage == "critic" and "proposed_main_text" not in packet:
            continue
        original = deepcopy(packet)
        before, after = old_render(packet), render(packet)
        assert after[0]["content"].startswith(before[0]["content"].replace(subject.BASE_VERSION, subject.VERSION))
        restored = re.sub(
            r"<directional_binding_rubric>.*?</document_owned_evidence>\n\n",
            "",
            after[1]["content"],
            count=1,
            flags=re.DOTALL,
        )
        assert restored == before[1]["content"]
        assert config["binding_rule"] in after[0]["content"]
        assert "<document_owned_evidence>" in after[1]["content"]
        if packet["canonical_pair_id"] == "context_ba":
            block = after[1]["content"].split("<document_owned_evidence>")[1]
            assert "A_ONLY source IDs: A001" in block
            assert "B_ONLY source IDs: \n" in block
        assert packet == original


def test_arbitrary_seed_fields_do_not_control_the_id_guide_or_leak_labels():
    packet = fixture("main")[0][-1]
    normal = subject.directional_renderer(stage="main")(packet)
    malicious = packet | {"owned_ids": "SECRET_GOLD", "binding_rule": "SECRET_GOLD", "human_label": "SECRET_GOLD"}
    assert subject.directional_renderer(stage="main")(malicious) == normal
