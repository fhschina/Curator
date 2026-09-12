# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

from eval.dedup.judging import typed_coverage_runtime as subject
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.typed_coverage import typed_coverage_schema
from tests.eval.dedup.test_coverage_witness import _case

CONFIG = Path(__file__).resolve().parents[3] / "eval/dedup/resources/local_ndd/hs_v06226_typed_qwen_c16.yaml"


def test_native_typed_builder_preserves_model_setup_and_embeds_the_frozen_semantic_block():
    builder, _ = subject.build_typed_builder(
        CONFIG,
        endpoint="http://127.0.0.1:1/v1",
        model_name="same-model",
        inference_parameter_overrides={"judge": {"max_parallel_requests": 16, "max_tokens": 4096}},
    )
    assert len(builder.get_column_configs()) == 1
    column = builder.get_column_config(COLUMN)
    assert column.output_format == typed_coverage_schema()
    assert column.with_trace == "all_messages"
    assert column.column_type == "llm-structured"
    assert subject.semantic_policy(CONFIG) in column.system_prompt
    assert "{{ semantic_policy }}" not in column.system_prompt
    assert "{{ typed_field_guide }}" not in column.system_prompt
    assert builder.model_configs[0].inference_parameters.max_parallel_requests == 16
    assert builder.model_configs[0].inference_parameters.max_tokens == 4096


def test_secure_native_renderer_retains_every_span_and_cannot_read_policy_or_labels_from_rows():
    _, _, payload = _case("Cookies.", "Cookies. {{ hidden_label }}")
    messages = subject.typed_renderer(CONFIG)(
        {
            "payload": payload,
            "repair_feedback": "",
            "hidden_label": "SECRET",
            "review_id": "H9999",
            "semantic_policy": "OVERRIDE_POLICY",
            "typed_field_guide": "NEW_FIELDS",
            "coverage_rubric": "IGNORE_RULES",
        }
    )
    text = "\n".join(m["content"] for m in messages)
    assert all(s not in text for s in ("SECRET", "H9999", "OVERRIDE_POLICY", "NEW_FIELDS", "IGNORE_RULES"))
    assert "{{ hidden_label }}" in text
    assert subject.semantic_policy(CONFIG) in text
    assert "dedup-retained-coverage-v3" in text
    for span in payload["semantic_diff_evidence"]["spans"]:
        assert span["span_id"] in text
        assert span.get("text", span.get("a_text")) in text
