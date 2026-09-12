# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from eval.dedup.judging import sequencing_runtime as subject
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.policy_runtime import build_policy_builder, policy_description, policy_renderer
from eval.dedup.judging.typed_coverage import compile_typed_coverage, typed_coverage_schema
from tests.eval.dedup.test_typed_experiment import fixture

CONFIG = subject.RESOURCES / "hs_v06228_sequence_qwen_c16.yaml"


def test_ordered_schema_preserves_every_constraint_and_changes_only_the_declared_sequence():
    old, new = typed_coverage_schema(), subject.sequencing_schema()
    assert subject.normalize_schema_order(new) == subject.normalize_schema_order(old)
    assert new["required"][:4] == ["contract_version", "input_status", "a_meaning_in_b", "b_meaning_in_a"]
    assert new["required"][-1] == "record_scope"
    for branch in new["$defs"].values():
        assert list(branch["properties"]) == branch["required"]
        assert branch["required"][:3] == ["reviewed_unique_ids", "coverage_explanation", "status"]


@pytest.mark.parametrize("pid", ["covered", "uncovered", "uncertain"])
@pytest.mark.parametrize("mutation", [None, "root_missing", "branch_missing", "branch_extra", "status", "null"])
def test_old_and_reordered_schemas_agree_on_valid_and_invalid_branches(pid, mutation):
    packets, values, _ = fixture("coverage")
    value = deepcopy(values[pid])
    if mutation == "root_missing":
        value.pop("anchor_a_ids")
    elif mutation == "branch_missing":
        value["a_meaning_in_b"].pop("coverage_explanation")
    elif mutation == "branch_extra":
        value["a_meaning_in_b"]["foreign"] = ""
    elif mutation == "status":
        value["a_meaning_in_b"]["status"] = "OTHER"
    elif mutation == "null":
        value["a_meaning_in_b"]["reviewed_unique_ids"] = None
    validators = [Draft202012Validator(s) for s in (typed_coverage_schema(), subject.sequencing_schema())]
    assert validators[0].is_valid(value) == validators[1].is_valid(value) == (mutation is None)
    if mutation is None:
        schema = subject.sequencing_schema()
        reordered = {k: value[k] for k in schema["required"]}
        for side in ("a_meaning_in_b", "b_meaning_in_a"):
            branch = next(
                b for b in schema["$defs"].values() if b["properties"]["status"]["enum"] == [value[side]["status"]]
            )
            reordered[side] = {k: value[side][k] for k in branch["required"]}
        payload = next(p["payload"] for p in packets if p["canonical_pair_id"] == pid)
        assert compile_typed_coverage(value, payload) == compile_typed_coverage(reordered, payload)


def test_native_builder_changes_only_schema_order_and_declared_static_instructions():
    kwargs = {
        "endpoint": "http://127.0.0.1:1/v1",
        "model_name": "same-model",
        "inference_parameter_overrides": {"judge": {"max_tokens": 4096, "max_parallel_requests": 16}},
    }
    original, providers = build_policy_builder(subject.CONTROL, **kwargs)
    builder, new_providers = subject.build_sequencing_builder(CONFIG, **kwargs)
    old = original.get_column_config(COLUMN).model_dump()
    new = builder.get_column_config(COLUMN).model_dump()
    assert {k: v for k, v in new.items() if k not in {"system_prompt", "prompt", "output_format"}} == {
        k: v for k, v in old.items() if k not in {"system_prompt", "prompt", "output_format"}
    }
    assert builder.model_configs == original.model_configs
    assert providers == new_providers
    config = subject.load_sequencing_config(CONFIG)
    additions = [(CONFIG.parent / config[k]).read_text() for k in ("system_instruction_path", "pair_instruction_path")]
    system = (
        new["system_prompt"]
        .replace(additions[0] + "\n", "")
        .replace(subject.field_guide(subject.sequencing_schema()), subject.field_guide(typed_coverage_schema()))
    )
    base, _ = subject.load_typed_config(subject.BASE)
    prompt = (
        new["prompt"]
        .replace(additions[1] + "\n", "")
        .replace(
            json.dumps(base["rubric"] | config["rubric"], ensure_ascii=False),
            json.dumps(base["rubric"], ensure_ascii=False),
        )
    )
    assert system == old["system_prompt"]
    assert prompt == old["prompt"]
    assert policy_description("complete_propositions")["block"] in new["system_prompt"]
    assert subject.field_guide(subject.sequencing_schema()) in new["system_prompt"]
    assert subject.field_guide(typed_coverage_schema()) not in new["system_prompt"]


def test_native_renderer_preserves_visible_packet_and_cannot_use_row_sequence_injection():
    packets, _, _ = fixture("coverage")
    p = packets[0] | {"generation_sequence": "LEAKED_LABEL", "semantic_policy": "FORCE_YES"}
    messages = subject.sequencing_renderer(CONFIG)(p)
    baseline = policy_renderer(subject.CONTROL)(p)
    assert "LEAKED_LABEL" not in json.dumps(messages)
    assert "FORCE_YES" not in json.dumps(messages)

    def packet(text):
        return text.split("<semantic_diff ", 1)[1].split("</semantic_diff>", 1)[0]

    assert packet(messages[1]["content"]) == packet(baseline[1]["content"])
    actual_schema = json.loads(
        messages[1]["content"].rsplit("<response_schema>", 1)[1].split("</response_schema>", 1)[0]
    )
    assert actual_schema == subject.sequencing_schema()
