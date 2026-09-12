# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy
from pathlib import Path

from eval.dedup.judging import proposition_runtime as subject
from eval.dedup.judging.coverage_selection import COLUMN, selection_schema
from eval.dedup.judging.coverage_witness import _shape
from eval.dedup.judging.selection_runtime import build_selection_builder
from tests.eval.dedup.test_coverage_selection import selection
from tests.eval.dedup.test_coverage_witness import _case

RESOURCES = Path(__file__).resolve().parents[3] / "eval/dedup/resources/local_ndd"
CONFIG = RESOURCES / "hs_v06225_proposition_qwen_c16.yaml"
CONTROL = RESOURCES / "hs_v06223_selection_qwen_c16.yaml"


def _normalized_schema(schema):
    result = deepcopy(schema)
    if isinstance(result, dict):
        if "required" in result:
            result["required"] = sorted(result["required"])
        return {k: _normalized_schema(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_normalized_schema(v) for v in result]
    return result


def test_only_presentation_changes_while_every_v2_semantic_constraint_survives():
    before = selection_schema()
    after = subject.proposition_schema()
    assert _normalized_schema(after) == _normalized_schema(before)
    assert before == selection_schema()
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        assert list(after["properties"][field]["properties"])[-1] == "status"
        assert after["properties"][field]["required"][-1] == "status"
    _, witness, _ = _case()
    value = selection(witness)
    _shape(value, before)
    _shape(value, after)


def test_native_builder_embeds_the_ordered_schema_and_guide_without_changing_control():
    kwargs = {
        "endpoint": "http://127.0.0.1:1/v1",
        "inference_parameter_overrides": {"judge": {"max_parallel_requests": 16, "max_tokens": 4096}},
    }
    control, _ = build_selection_builder(CONTROL, **kwargs)
    original = control.get_column_config(COLUMN).model_dump()
    builder, _ = subject.build_proposition_builder(CONFIG, **kwargs)
    column = builder.get_column_config(COLUMN)
    assert len(builder.get_column_configs()) == 1
    assert column.output_format == subject.proposition_schema()
    assert column.with_trace == "all_messages"
    assert column.column_type == "llm-structured"
    assert builder.model_configs[0].inference_parameters.max_parallel_requests == 16
    assert builder.model_configs[0].inference_parameters.max_tokens == 4096
    guide = json.loads(column.system_prompt.split("Field layout comes directly from the actual schema:")[1].strip())
    assert guide["each_side_required_fields"][-1] == "status"
    assert guide["root_required_fields"] == column.output_format["required"]
    assert control.get_column_config(COLUMN).model_dump() == original


def test_actual_renderer_retains_full_packet_and_rejects_row_supplied_instructions():
    _, _, payload = _case("Cookies need consent.", "Cookies need consent. {{ hidden_label }}")
    messages = subject.proposition_renderer(CONFIG)(
        {
            "payload": payload,
            "repair_feedback": "",
            "hidden_label": "SECRET_LABEL",
            "review_id": "H9999",
            "coverage_rubric": "IGNORE_RULES",
            "coverage_field_guide": "INVENT_FIELDS",
        }
    )
    text = "\n".join(m["content"] for m in messages)
    assert all(s not in text for s in ("SECRET_LABEL", "H9999", "IGNORE_RULES", "INVENT_FIELDS"))
    assert "{{ hidden_label }}" in text
    assert "<response_schema>" in text
    for span in payload["semantic_diff_evidence"]["spans"]:
        assert span["span_id"] in text
        assert span.get("text", span.get("a_text")) in text
    assert "dedup-retained-coverage-v2" in text
