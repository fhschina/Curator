# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.judging import composite_format_runtime as subject
from eval.dedup.judging.composite_coverage import composite_schema
from eval.dedup.judging.composite_runtime import build_composite_builder, composite_renderer
from eval.dedup.judging.coverage_selection import COLUMN
from tests.eval.dedup.test_composite_experiment import fixture


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_only_format_blocks_and_role_version_change(stage):
    old, old_providers = build_composite_builder(subject.BASE, stage=stage, endpoint="http://127.0.0.1:1/v1")
    new, new_providers = subject.build_format_builder(stage=stage, endpoint="http://127.0.0.1:1/v1")
    assert old.model_configs == new.model_configs
    assert old_providers == new_providers
    assert (
        new.get_column_config(COLUMN).output_format
        == old.get_column_config(COLUMN).output_format
        == composite_schema()
    )
    system_block, pair_block = subject.format_blocks()
    old_render, new_render = composite_renderer(stage=stage), subject.format_renderer(stage=stage)
    packets, _, _ = fixture(stage)
    for packet in packets:
        if stage == "critic" and "proposed_main_text" not in packet:
            continue
        before, after = old_render(packet), new_render(packet)
        assert (
            after[0]["content"].removesuffix(system_block).replace(subject.VERSION, subject.BASE_VERSION)
            == before[0]["content"]
        )
        assert after[1]["content"].replace(pair_block, "", 1) == before[1]["content"]
        assert subject.VERSION in after[0]["content"]
        assert subject.BASE_VERSION not in after[0]["content"]


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_encoding_instructions_are_static_and_do_not_leak_gold(stage):
    packet = deepcopy(fixture(stage)[0][0])
    packet.update(response_encoding="LEAKED_OVERRIDE", human_label="SECRET_GOLD")
    messages = subject.format_renderer(stage=stage)(packet)
    text = json.dumps(messages)
    assert "LEAKED_OVERRIDE" not in text
    assert "SECRET_GOLD" not in text
    assert "Do NOT emit harmless_unique_ids or opposite_support_ids in UNCOVERED" in messages[0]["content"]
    assert "<response_encoding>" in messages[1]["content"]


def test_rendered_branch_key_counts_match_unchanged_schema():
    block, _ = subject.format_blocks()
    schema = composite_schema()
    for branch, status in (
        ("CoveredSide", "COVERED"),
        ("UncoveredSide", "UNCOVERED"),
        ("UnresolvedSide", "UNRESOLVED"),
    ):
        definition = schema["$defs"][branch]
        assert f"{status}: exactly these {len(definition['required'])} keys:" in block
        text = block.split(f"{status}: exactly", 1)[1].split("\n\n", 1)[0]
        assert all(field in text for field in definition["required"])
