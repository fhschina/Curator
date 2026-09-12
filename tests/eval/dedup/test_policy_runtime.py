# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import yaml

from eval.dedup.judging import policy_runtime as subject
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.typed_coverage_runtime import build_typed_builder, semantic_policy, typed_renderer
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_witness import _case

CONFIGS = {
    "control": subject.RESOURCES / "hs_v06227_control_qwen_c16.yaml",
    "coverage": subject.RESOURCES / "hs_v06227_proposition_qwen_c16.yaml",
}


def test_native_builder_changes_only_the_declared_static_semantic_block():
    kwargs = {
        "endpoint": "http://127.0.0.1:1/v1",
        "model_name": "same-model",
        "inference_parameter_overrides": {"judge": {"max_parallel_requests": 16, "max_tokens": 4096}},
    }
    base, providers = build_typed_builder(subject.BASE, **kwargs)
    base_column = base.get_column_config(COLUMN).model_dump()
    for variant, path in CONFIGS.items():
        builder, actual_providers = subject.build_policy_builder(path, **kwargs)
        policy = subject.policy_description(subject.load_policy_config(path)["semantic_policy"])
        actual = builder.get_column_config(COLUMN).model_dump()
        expected = {
            **base_column,
            "system_prompt": base_column["system_prompt"].replace(semantic_policy(subject.BASE), policy["block"]),
        }
        assert actual == expected
        assert builder.model_configs == base.model_configs
        assert actual_providers == providers
        assert len(builder.get_column_configs()) == 1
        assert actual["with_trace"] == "all_messages"
        if variant == "coverage":
            assert actual == base_column
        else:
            assert actual["system_prompt"] != base_column["system_prompt"]


@pytest.mark.parametrize("variant", CONFIGS)
def test_native_renderer_preserves_data_and_ignores_row_policy_injection(variant):
    _, _, payload = _case("Cookies need consent.", "Cookies need consent. {{ secret }}")
    row = {
        "payload": payload,
        "repair_feedback": "Validation issue: TEST",
        "semantic_policy": "EVIL_POLICY",
        "coverage_rubric": "EVIL_RUBRIC",
        "secret": "HIDDEN_LABEL",
    }
    messages = subject.policy_renderer(CONFIGS[variant])(row)
    text = "\n".join(m["content"] for m in messages)
    assert all(s not in text for s in ("EVIL_POLICY", "EVIL_RUBRIC", "HIDDEN_LABEL"))
    assert "{{ secret }}" in text
    assert "Validation issue: TEST" in text
    for span in payload["semantic_diff_evidence"]["spans"]:
        assert span["span_id"] in text
        assert span.get("text", span.get("a_text")) in text
    baseline = typed_renderer(subject.BASE)(row)
    assert messages[1] == baseline[1]
    if variant == "coverage":
        assert messages == baseline


@pytest.mark.parametrize(
    "text", ["START only", "END START", "START START END", "START {{ row }} END", "START {% if x %} END"]
)
def test_ambiguous_or_dynamic_policy_block_is_rejected(text):
    with pytest.raises(DedupEvaluationError):
        subject.extract_block(text, "START", "END")


@pytest.mark.parametrize(
    "change", [{"semantic_policy": "unknown"}, {"base_typed_config": "different.yaml"}, {"max_tokens": 1024}]
)
def test_config_cannot_change_an_undeclared_task_dimension(tmp_path, change):
    config = {"base_typed_config": str(subject.BASE), "semantic_policy": "concise_boundaries", **change}
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(DedupEvaluationError, match="POLICY_CONFIG"):
        subject.load_policy_config(path)


def test_unknown_policy_name_is_rejected():
    with pytest.raises(DedupEvaluationError, match="POLICY_NAME"):
        subject.policy_description("undeclared")
