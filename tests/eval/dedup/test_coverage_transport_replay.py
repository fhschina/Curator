# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pytest

from eval.dedup.analysis.coverage_diagnostic import CONFIG, bind_coverage
from eval.dedup.analysis.coverage_transport_replay import bind_coverage_transport
from eval.dedup.analysis.critic_diagnostic import _write_jsonl, blind_rows
from eval.dedup.judging.coverage_runtime import CoverageWitnessRuntime, coverage_renderer
from eval.dedup.judging.coverage_witness import COLUMN
from eval.dedup.judging.local_ndd import _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError, sha256_json
from tests.eval.dedup.test_coverage_witness import _case


def _mixed_case():
    main, value, payload = _case("Cookies need consent.", "Cookies need consent. Accept Settings", non_main=True)
    value["b_meaning_in_a"].update(
        coverage_mode="HARMLESS_ONLY",
        coverage_counterpart_ids="",
        coverage_explanation="Optional controls assert no new retained state or permission.",
    )
    packets = blind_rows([{"canonical_pair_id": "p", "payload": payload}], repeat=1)
    raw = "```json\n" + json.dumps(value) + "\n```"
    output = {
        **packets[0],
        COLUMN: value,
        COLUMN + "__trace": [{"role": "assistant", "content": [{"type": "text", "text": raw}]}],
    }
    return packets, output, {"p": main}


def test_transport_only_binding_keeps_original_response_trace_and_public_evidence():
    packets, output, mains = _mixed_case()
    raw = json.loads(pa.Table.from_pylist([output]).to_pandas().to_json(orient="records"))[0]
    original = deepcopy((packets, raw, mains))
    with pytest.raises(DedupEvaluationError, match="COVERAGE_PAYLOAD_CHANGED"):
        bind_coverage(packets, [raw], mains)
    actual = bind_coverage_transport(packets, [raw], mains)[0]
    expected = bind_coverage(packets, [output], mains)[0]
    assert all(actual[key] == value for key, value in expected.items())
    assert actual["payload_transport"]["raw_echo_payload_sha256"] == sha256_json(raw["payload"])
    assert (packets, raw, mains) == original


@pytest.mark.parametrize("failure", ["missing", "extra_pair", "text", "main", "trace", "citation"])
def test_transport_binding_does_not_relax_model_or_input_contracts(failure):
    packets, output, mains = _mixed_case()
    outputs = [deepcopy(output)]
    if failure == "missing":
        outputs = []
    elif failure == "extra_pair":
        outputs.append({**output, "canonical_pair_id": "foreign"})
    elif failure == "text":
        outputs[0]["payload"]["document_a"]["text"] = "Changed text"
    elif failure == "main":
        outputs[0]["qwen_dedup_semantic_judge"] = mains["p"]
    elif failure == "trace":
        outputs[0].pop(COLUMN + "__trace")
    else:
        outputs[0][COLUMN]["anchor_a_ids"] = "A999"
        outputs[0][COLUMN + "__trace"][0]["content"][0]["text"] = (
            "```json\n" + json.dumps(outputs[0][COLUMN]) + "\n```"
        )
    with pytest.raises(DedupEvaluationError):
        bind_coverage_transport(packets, outputs, mains)


@pytest.mark.parametrize("feedback", [None, {"code": "QUOTE_INVALID", "message": "Retry", "details": {}}])
def test_full_ray_pipeline_mixed_spans_with_local_http_only(httpserver, tmp_path: Path, feedback):
    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray address required for full executor integration")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    packets, output, mains = _mixed_case()
    packets[0]["repair_feedback"] = encode_repair_feedback(feedback)
    packets.append({**packets[0], "canonical_pair_id": "q"})
    mains["q"] = mains["p"]
    response = "```json\n" + json.dumps(output[COLUMN]) + "\n```"
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "id": "coverage-transport-boundary",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    _write_jsonl(tmp_path / "input.jsonl", packets)
    with CoverageWitnessRuntime(
        CONFIG,
        endpoint=httpserver.url_for("/v1"),
        model_name="coverage-transport-boundary",
        ray_temp_dir="/raid/hfang/ihb/r20diag",
    ) as runtime:
        runtime.run(
            input_path=str(tmp_path / "input.jsonl"),
            input_format="jsonl",
            output_path=str(tmp_path / "output"),
            checkpoint_path=str(tmp_path / "checkpoints"),
            files_per_partition=1,
            inference_parameter_overrides={
                "judge": {"temperature": 0, "top_p": 1, "max_tokens": 4096, "max_parallel_requests": 2}
            },
        )
    rows = _read_output_rows(tmp_path / "output")
    assert len(rows) == len(httpserver.log) == 2
    predictions = bind_coverage_transport(packets, rows, mains)
    assert {p["canonical_pair_id"] for p in predictions} == {"p", "q"}
    assert all(p["a_can_replace_b"] == p["b_can_replace_a"] == "YES" for p in predictions)
    assert all(p["payload_transport"]["representation_changes"]["added_null_keys"] > 0 for p in predictions)
    expected_messages = coverage_renderer(CONFIG)(packets[0])
    for request, _ in httpserver.log:
        messages = json.loads(request.get_data())["messages"]
        for actual, expected in zip(messages, expected_messages, strict=True):
            content = actual["content"]
            if isinstance(content, list):
                content = "".join(block["text"] for block in content)
            assert actual["role"] == expected["role"]
            assert content == expected["content"]
