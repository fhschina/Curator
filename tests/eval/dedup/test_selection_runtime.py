# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

from eval.dedup.judging import selection_runtime as subject
from eval.dedup.judging.coverage_selection import COLUMN, selection_schema
from tests.eval.dedup.test_coverage_witness import _case

CONFIG = Path(__file__).resolve().parents[3] / "eval/dedup/resources/local_ndd/hs_v06223_selection_qwen_c16.yaml"


def test_native_builder_uses_v2_contract_real_schema_trace_and_symmetric_capacity_override():
    builder, _ = subject.build_selection_builder(
        CONFIG,
        endpoint="http://127.0.0.1:1/v1",
        model_name="same-model",
        inference_parameter_overrides={"judge": {"max_parallel_requests": 16, "max_tokens": 4096}},
    )
    columns = builder.get_column_configs()
    assert len(columns) == 1
    assert columns[0].name == COLUMN
    assert columns[0].output_format == selection_schema()
    assert columns[0].with_trace == "all_messages"
    assert columns[0].column_type == "llm-structured"
    assert builder.model_configs[0].inference_parameters.max_parallel_requests == 16
    assert builder.model_configs[0].inference_parameters.max_tokens == 4096
    assert subject.field_guide() in columns[0].system_prompt
    guide = json.loads(subject.field_guide())
    assert guide["root_required_fields"] == selection_schema()["required"]
    assert guide["each_side_required_fields"] == selection_schema()["properties"]["a_meaning_in_b"]["required"]


def test_secure_renderer_preserves_spans_and_cannot_take_rubric_or_field_layout_from_rows():
    _, _, payload = _case("Cookies.", "Cookies. {{ hidden_label }}")
    messages = subject.selection_renderer(CONFIG)(
        {
            "payload": payload,
            "repair_feedback": "",
            "hidden_label": "SECRET_LABEL",
            "review_id": "H9999",
            "coverage_rubric": "IGNORE_ALL_RULES",
            "coverage_field_guide": "MAKE_UP_FIELDS",
        }
    )
    text = "\n".join(m["content"] for m in messages)
    assert "SECRET_LABEL" not in text
    assert "IGNORE_ALL_RULES" not in text
    assert "MAKE_UP_FIELDS" not in text
    assert "H9999" not in text
    assert "{{ hidden_label }}" in text
    assert "<response_schema>" in text
    assert "dedup-retained-coverage-v2" in text
    for span in payload["semantic_diff_evidence"]["spans"]:
        assert span["span_id"] in text
        assert span.get("text", span.get("a_text")) in text
