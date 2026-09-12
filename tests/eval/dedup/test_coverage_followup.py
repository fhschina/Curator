# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis import coverage_followup as subject
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl, blind_rows
from eval.dedup.judging.coverage_witness import COLUMN
from eval.dedup.judging.local_ndd import RECORD_BINDING_CRITIC_COLUMN, _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_witness import _case


def _fixture(variant, feedback=None):
    main, coverage, payload = _case("Cookies need consent.", "Cookies need consent. Accept Settings", non_main=True)
    coverage["b_meaning_in_a"].update(coverage_mode="HARMLESS_ONLY", coverage_counterpart_ids="")
    control = {
        key: {"score": score, "reasoning": "S001 B001"}
        for key, score in (
            ("record_scope", "equivalent_complete_message"),
            ("record_binding_verdict", "benign_non_record_delta"),
            ("retained_conflict", "none"),
        )
    }
    packets = blind_rows([{"canonical_pair_id": "p", "payload": payload}], repeat=1)
    packets[0]["repair_feedback"] = encode_repair_feedback(feedback)
    value = coverage if variant == "coverage" else control
    return packets, value, {"p": main}


def _output(packet, value, variant):
    column = COLUMN if variant == "coverage" else RECORD_BINDING_CRITIC_COLUMN
    trace = [
        *subject.renderer(variant)(packet),
        {"role": "assistant", "content": "```json\n" + json.dumps(value) + "\n```"},
    ]
    return {**packet, column: value, column + "__trace": trace}


def _settings():
    return {
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 4096,
        "timeout_seconds": 30,
        "max_parallel_requests": 2,
        "max_retries": 2,
    }


@pytest.mark.parametrize("variant", subject.VARIANTS)
def test_native_control_and_coverage_bind_with_their_actual_recipes(variant):
    subject.validate_control()
    packets, value, mains = _fixture(variant)
    raw = _output(packets[0], value, variant)
    saved = deepcopy((packets, raw, mains))
    result = subject.bind(packets, [raw], mains, variant)[0]
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "YES"
    assert result["native_corrections"] == 0
    assert result["diagnostic_version"] == "v0.6.2.21"
    assert "offline_transport_diagnostic_only" not in result
    assert (packets, raw, mains) == saved


@pytest.mark.parametrize("variant", subject.VARIANTS)
@pytest.mark.parametrize("failure", ["payload", "feedback", "messages", "trace", "extra_response_field"])
def test_native_request_and_original_response_cannot_be_repaired_silently(variant, failure):
    from data_designer.engine.models.parsers.errors import ParserException

    packets, value, mains = _fixture(variant)
    raw = deepcopy(_output(packets[0], value, variant))
    column = COLUMN if variant == "coverage" else RECORD_BINDING_CRITIC_COLUMN
    if failure == "payload":
        raw["payload"]["document_a"]["text"] = "Changed"
    elif failure == "feedback":
        raw["repair_feedback"] = None
    elif failure == "messages":
        raw[column + "__trace"][1]["content"] += " changed prompt"
    elif failure == "trace":
        raw.pop(column + "__trace")
    else:
        raw[column + "__trace"][-1]["content"] = "```json\n" + json.dumps({**value, "extra": "hidden"}) + "\n```"
    with pytest.raises((DedupEvaluationError, ParserException)):
        subject.bind(packets, [raw], mains, variant)


class _Relay:
    def set_context(self, context):
        self.context = context


def test_only_model_contract_failure_retries_with_stable_feedback_and_exact_raw_replay(tmp_path):
    packets, value, mains = _fixture("coverage")
    packets.append({**packets[0], "canonical_pair_id": "q"})
    mains["q"] = mains["p"]
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            rows = []
            for packet in batch:
                v = deepcopy(value)
                if packet["canonical_pair_id"] == "q" and len(batches) == 1:
                    v["b_meaning_in_a"]["coverage_counterpart_ids"] = "S001"
                rows.append(_output(packet, v, "coverage"))
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    name = "preflight_coverage"
    complete = subject.run_cell(tmp_path / name, packets, mains, Runtime(), _Relay(), _settings(), "coverage")
    assert [[r["canonical_pair_id"] for r in b] for b in batches] == [["p", "q"], ["q"]]
    assert batches[1][0]["repair_feedback"].startswith("Validation issue: ")
    assert complete["valid"] == 2
    assert complete["errors"] == 0
    assert complete["judge_retried_pairs"] == 1
    assert not complete["fatal_boundary"]
    replayed, _ = subject.replay_cell(tmp_path, name, "coverage", packets, mains)
    assert len(replayed) == 2
    with pytest.raises(DedupEvaluationError, match="reuse"):
        subject.run_cell(tmp_path / name, packets, mains, Runtime(), _Relay(), _settings(), "coverage")


def test_deterministic_transport_failure_stops_without_model_retry_and_preserves_terminal_ids(tmp_path):
    packets, value, mains = _fixture("coverage")
    calls = []

    class Runtime:
        def run(self, **kwargs):
            calls.append(kwargs)
            raw = deepcopy(_output(packets[0], value, "coverage"))
            raw["payload"]["document_a"]["text"] = "Wrong input"
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", [raw])

    complete = subject.run_cell(
        tmp_path / "preflight_coverage", packets, mains, Runtime(), _Relay(), _settings(), "coverage"
    )
    assert len(calls) == 1
    assert complete["fatal_boundary"]
    assert complete["errors"] == 1
    assert complete["retried"] == 0
    assert json.loads((tmp_path / "preflight_coverage/terminal_pair_ids.json").read_text()) == ["p"]
    assert subject.replay_cell(tmp_path, "preflight_coverage", "coverage", packets, mains)[0] == []


@pytest.mark.parametrize("variant", subject.VARIANTS)
@pytest.mark.parametrize("feedback", [None, {"code": "QUOTE_INVALID", "message": "Retry", "details": {}}])
def test_full_ray_http_boundary_for_both_arms_and_feedback_states(httpserver, tmp_path, variant, feedback):
    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray address required for full pipeline integration")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    packets, value, mains = _fixture(variant, feedback)
    packets.append({**packets[0], "canonical_pair_id": "q"})
    mains["q"] = mains["p"]
    response = "```json\n" + json.dumps(value) + "\n```"
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "id": "coverage-followup-boundary",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    settings = {
        **_settings(),
        "logical_model": "coverage-followup-boundary",
        "ray_temp_dir": "/raid/hfang/ihb/r21diag",
    }
    _write_jsonl(tmp_path / "input.jsonl", packets)
    with subject.runtime_for(variant, httpserver.url_for("/v1"), settings) as runtime:
        runtime.run(
            input_path=str(tmp_path / "input.jsonl"),
            input_format="jsonl",
            output_path=str(tmp_path / "output"),
            output_format="jsonl",
            checkpoint_path=str(tmp_path / "checkpoints"),
            files_per_partition=1,
            inference_parameter_overrides={
                "judge": {"temperature": 0, "top_p": 1, "max_tokens": 4096, "max_parallel_requests": 2}
            },
        )
    raw = _read_output_rows(tmp_path / "output")
    assert len(raw) == len(httpserver.log) == 2
    predictions = subject.bind(packets, raw, mains, variant)
    assert all(p["a_can_replace_b"] == p["b_can_replace_a"] == "YES" for p in predictions)
    assert all(p["payload_transport"]["representation_changes"]["added_null_keys"] > 0 for p in predictions)
    expected = subject.renderer(variant)(packets[0])
    for request, _ in httpserver.log:
        messages = json.loads(request.get_data())["messages"]
        assert [{"role": m["role"], "content": subject.trace_text(m["content"])} for m in messages] == expected
