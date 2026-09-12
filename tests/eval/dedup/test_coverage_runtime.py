# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from eval.dedup.judging.coverage_runtime import build_coverage_builder, coverage_renderer, validate_native_coverage_row
from eval.dedup.judging.coverage_witness import COLUMN, coverage_schema
from eval.dedup.judging.payload import _semantic_diff_packet
from eval.dedup.validation import DedupEvaluationError

CONFIG = Path(__file__).resolve().parents[3] / "eval/dedup/resources/local_ndd/hs_v06220_coverage_qwen_c64.yaml"


def _shape(schema):
    if schema["type"] == "object":
        return {key: _shape(child) for key, child in schema["properties"].items()}
    return schema.get("enum", [""])[0]


def test_native_builder_uses_only_structured_coverage_with_raw_trace_and_real_schema():
    builder, providers = build_coverage_builder(
        CONFIG,
        endpoint="http://127.0.0.1:1/v1",
        model_name="fixed-logical-model",
        inference_parameter_overrides={"judge": {"temperature": 0, "top_p": 1, "max_tokens": 4096}},
    )
    columns = builder.get_column_configs()
    assert len(columns) == 1
    column = columns[0]
    assert column.name == COLUMN
    assert column.column_type == "llm-structured"
    assert column.with_trace == "all_messages"
    assert column.output_format == coverage_schema()
    assert providers[0].endpoint == "http://127.0.0.1:1/v1"
    assert "{{ coverage_rubric }}" not in column.prompt
    assert "Judge each direction of retained meaning independently" in column.prompt


def test_actual_secure_renderer_preserves_full_spans_without_hidden_metadata_or_template_execution():
    a, b = "Use cookies with consent.", "Use cookies with consent. {{ hidden_label }}"
    payload = {
        "document_a": {"text": a},
        "document_b": {"text": b},
        "long_document_evidence": {"truncated": False},
        "semantic_diff_evidence": _semantic_diff_packet(a, b, truncated=False),
    }
    messages = coverage_renderer(CONFIG)(
        {"payload": payload, "repair_feedback": None, "hidden_label": "PRIVATE_LABEL_SENTINEL", "review_id": "H9999"}
    )
    combined = "\n".join(message["content"] for message in messages)
    assert "{{ hidden_label }}" in combined
    assert "PRIVATE_LABEL_SENTINEL" not in combined
    assert "H9999" not in combined
    assert "<response_schema>" in combined
    assert "dedup-retained-coverage-v1" in combined
    for span in payload["semantic_diff_evidence"]["spans"]:
        assert span["span_id"] in combined
        assert span.get("text", span.get("a_text")) in combined


def test_native_response_is_checked_against_unpruned_original_assistant():
    value = _shape(coverage_schema())
    raw = "```json\n" + json.dumps(value) + "\n```"
    row = {COLUMN: json.dumps(value), COLUMN + "__trace": [{"role": "assistant", "content": raw}]}
    assert validate_native_coverage_row(row) == value
    row[COLUMN] = {**value, "record_scope": "UNRESOLVED"}
    with pytest.raises(DedupEvaluationError, match="raw response must equal"):
        validate_native_coverage_row(row)


def test_native_default_pruning_is_not_misreported_as_raw_schema_success():
    from data_designer.engine.models.parsers.errors import ParserException
    from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

    value = _shape(coverage_schema())
    raw = "```json\n" + json.dumps({**value, "unexpected_label": "YES"}) + "\n```"
    parsed = StructuredResponseRecipe(coverage_schema()).parse(raw)
    assert parsed == value
    with pytest.raises(ParserException):
        validate_native_coverage_row({COLUMN: parsed, COLUMN + "__trace": [{"role": "assistant", "content": raw}]})


def test_missing_trace_cannot_be_fabricated_from_the_parsed_column():
    with pytest.raises(DedupEvaluationError, match="raw response trace"):
        validate_native_coverage_row({COLUMN: _shape(coverage_schema())})


def test_native_trace_text_blocks_are_losslessly_joined_and_nontext_is_rejected():
    value = _shape(coverage_schema())
    raw = "```json\n" + json.dumps(value) + "\n```"
    row = {
        COLUMN: value,
        COLUMN + "__trace": [
            {"role": "assistant", "content": [{"type": "text", "text": raw[:30]}, {"type": "text", "text": raw[30:]}]}
        ],
    }
    assert validate_native_coverage_row(row) == value
    row[COLUMN + "__trace"][0]["content"].append({"type": "image_url", "image_url": "unavailable"})
    with pytest.raises(DedupEvaluationError, match="only text blocks"):
        validate_native_coverage_row(row)


def test_real_ndd_stage_preserves_raw_coverage_and_pair_identity_over_http(httpserver):
    import data_designer.config as dd
    import pandas as pd

    from eval.dedup.judging.coverage_witness import parse_coverage
    from eval.dedup.validation import sha256_json
    from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage
    from nemo_curator.tasks import DocumentBatch

    text = "Identical cookie notice."
    payload = {
        "document_a": {"text": text},
        "document_b": {"text": text},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": _semantic_diff_packet(text, text, truncated=False),
    }
    value = _shape(coverage_schema())
    value.update(
        record_scope="IDENTICAL_TEXT",
        anchor_a_ids="S001",
        anchor_b_ids="S001",
        scope_explanation="Identical complete original text.",
    )
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        value[field].update(
            coverage_mode="NO_UNIQUE_SPANS",
            coverage_explanation="No unique spans.",
            counterpart_relation="NOT_APPLICABLE",
        )
    raw = "```json\n" + json.dumps(value) + "\n```"
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "id": "coverage-boundary",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": raw}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    builder, providers = build_coverage_builder(
        CONFIG,
        endpoint=httpserver.url_for("/v1"),
        model_name="coverage-boundary",
        inference_parameter_overrides={
            "judge": {"temperature": 0, "top_p": 1, "max_tokens": 4096, "max_parallel_requests": 2}
        },
    )
    stage = DataDesignerStage(
        config_builder=builder,
        model_providers=providers,
        run_config=dd.RunConfig(
            disable_early_shutdown=True, max_conversation_restarts=0, max_conversation_correction_steps=0
        ),
    )
    rows = [
        {
            "canonical_pair_id": pid,
            "judge_payload_hash": sha256_json(payload),
            "payload": payload,
            "repair_feedback": None,
        }
        for pid in ("p", "q")
    ]
    out = (
        stage.process(DocumentBatch(dataset_name="coverage-boundary", data=pd.DataFrame(rows)))
        .to_pandas()
        .to_dict("records")
    )
    assert {row["canonical_pair_id"] for row in out} == {"p", "q"}
    for row in out:
        assert row["judge_payload_hash"] == sha256_json(payload)
        assert "qwen_dedup_semantic_judge" not in row
        assert "qwen_dedup_record_binding_critic" not in row
        assert validate_native_coverage_row(row) == value
        parse_coverage(value, payload)
    assert len(httpserver.log) == 2
