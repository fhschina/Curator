# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis import proposition_experiment as subject
from eval.dedup.analysis.coverage_followup import trace_text
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.local_ndd import _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError, sha256_json
from tests.eval.dedup.test_coverage_experiment import _mixed
from tests.eval.dedup.test_coverage_followup import _Relay, _settings
from tests.eval.dedup.test_coverage_selection import selection
from tests.eval.dedup.test_coverage_witness import _case, _uncover


@pytest.fixture
def experiment():
    return subject.PropositionExperiment(subject.common.ANALYSIS / "v06225_experiment.json")


def fixture(*, extension=False):
    packets, value, mains = _mixed("coverage")
    if extension:
        main, value, payload = _case("FAQ. กฎหมายแพ่งและพาณิชย์ว่าด้วยหุ้นส่วนบริษัท", "FAQ.")
        _uncover(value, payload, "A")
    else:
        main, value, payload = _case("Cookies need consent.", "Cookie 需要获得同意。", non_main=True)
        value["record_scope"] = "SAME_SUBSTANTIVE_RECORD"
    packets[0].update(payload=payload, judge_payload_hash=sha256_json(payload))
    mains["p"] = main
    return packets, selection(value), mains


def output(experiment, packet, value, variant):
    return {
        **packet,
        COLUMN: value,
        COLUMN + "__trace": [
            *experiment.renderers[variant](packet),
            {"role": "assistant", "content": "```json\n" + json.dumps(value) + "\n```"},
        ],
    }


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize("native_correction", [False, True])
def test_both_arms_use_identical_owned_routes_scope_policy_and_real_correction_denominator(
    experiment, tmp_path, variant, native_correction
):
    packets, value, mains = fixture()
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            raw = [output(experiment, p, value, variant) for p in batch]
            if native_correction:
                raw[0][COLUMN + "__trace"][2:2] = [
                    {"role": "assistant", "content": "Not JSON"},
                    {"role": "user", "content": "Correct the schema"},
                ]
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", raw)

    ops = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), variant)
    assert [[p["canonical_pair_id"] for p in batch] for batch in batches] == [["p"]]
    assert ops["called_pairs"] == 1
    assert ops["not_requested_owned_pairs"] == 2
    assert ops["valid"] == ops["requested"] == 3
    assert ops["called_pair_retry_rate"] == float(native_correction)
    assert ops["native_corrected_pairs"] == int(native_correction)
    rows, _ = experiment.replay_cell(tmp_path, "cell", variant, packets, mains)
    called = next(r for r in rows if r["canonical_pair_id"] == "p")
    assert called["same_duplicate_group"] == "YES"
    assert called["scope_postprocessor"] == subject.POLICY
    assert called["scope_postprocessor_audit"]["action"] == "BILATERAL_COVERAGE_INDEPENDENT_OF_TAXONOMY"
    for row in rows:
        if row["attempts"] == 0:
            assert "scope_postprocessor" not in row
            assert "raw_output_sha256" not in row


@pytest.mark.parametrize("variant", ["control", "coverage"])
def test_model_contract_retry_preserves_feedback_and_only_calls_the_original_active_pair(
    experiment, tmp_path, variant
):
    packets, value, mains = fixture()
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            new = deepcopy(value)
            if len(batches) == 1:
                new["b_meaning_in_a"]["source_span_id"] = "B999"
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl", [output(experiment, p, new, variant) for p in batch]
            )

    ops = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), variant)
    assert len(batches) == 2
    assert batches[1][0]["repair_feedback"].startswith("Validation issue: ")
    assert ops["called_pair_retry_rate"] == 1.0
    assert len(experiment.replay_cell(tmp_path, "cell", variant, packets, mains)[0]) == 3


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize("failure", ["payload", "feedback", "prompt", "raw_extra", "main_leak"])
def test_original_boundary_checks_apply_to_both_selection_arms(experiment, variant, failure):
    from data_designer.engine.models.parsers.errors import ParserException

    packets, value, mains = fixture()
    called, _ = experiment.layout(packets, mains, variant)
    row = deepcopy(output(experiment, called[0], value, variant))
    if failure == "payload":
        row["payload"]["document_a"]["text"] += "changed"
    elif failure == "feedback":
        row["repair_feedback"] = None
    elif failure == "prompt":
        row[COLUMN + "__trace"][1]["content"] += "changed"
    elif failure == "main_leak":
        row["qwen_dedup_semantic_judge"] = mains["p"]
    else:
        row[COLUMN + "__trace"][-1]["content"] = "```json\n" + json.dumps({**value, "guess": "YES"}) + "\n```"
    with pytest.raises((DedupEvaluationError, ParserException)):
        experiment.bind(called, [row], mains, variant)


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize(
    "feedback", [None, {"code": "SELECTION_REFERENCE_INVALID", "message": "Retry", "details": {}}]
)
def test_full_ray_http_first_and_retry_preserve_selection_evidence_and_messages(
    experiment, httpserver, tmp_path, variant, feedback
):
    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray address required for full pipeline integration")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    packets, value, mains = fixture(extension=True)
    called, owned = experiment.layout(packets, mains, variant)
    for packet in called:
        packet["repair_feedback"] = encode_repair_feedback(feedback)
    response = "```json\n" + json.dumps(value) + "\n```"
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "id": "proposition-boundary",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    settings = {**_settings(), "logical_model": "proposition-boundary", "ray_temp_dir": "/raid/hfang/ihb/r25diag"}
    _write_jsonl(tmp_path / "input.jsonl", called)
    with experiment.runtime(variant, httpserver.url_for("/v1"), settings) as runtime:
        runtime.run(
            input_path=str(tmp_path / "input.jsonl"),
            input_format="jsonl",
            output_path=str(tmp_path / "output"),
            output_format="jsonl",
            checkpoint_path=str(tmp_path / "checkpoints"),
            files_per_partition=1,
            inference_parameter_overrides={
                "judge": {"temperature": 0, "top_p": 1, "max_tokens": 4096, "max_parallel_requests": 16}
            },
        )
    raw = _read_output_rows(tmp_path / "output")
    assert len(raw) == len(httpserver.log) == len(called) == 1
    predictions = experiment.bind(called, raw, mains, variant)
    assert len(predictions) + len(owned) == 3
    assert predictions[0]["relation_type"] == "CONTAINMENT"
    assert predictions[0]["a_can_replace_b"] == "YES"
    assert predictions[0]["b_can_replace_a"] == "NO"
    body = json.loads(httpserver.log[0][0].get_data())
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0
    assert body["top_p"] == 1
    assert [
        {"role": m["role"], "content": trace_text(m["content"])} for m in body["messages"]
    ] == experiment.renderers[variant](called[0])
