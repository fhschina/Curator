# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis import selection_experiment as subject
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.local_ndd import _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_experiment import _mixed, _output
from tests.eval.dedup.test_coverage_followup import _Relay, _settings
from tests.eval.dedup.test_coverage_selection import selection


@pytest.fixture
def experiment():
    return subject.SelectionExperiment(subject.common.ANALYSIS / "v06223_experiment.json")


def fixture(variant):
    packets, value, mains = _mixed(variant)
    return packets, selection(value) if variant == "coverage" else value, mains


@pytest.mark.parametrize("variant", ["control", "coverage"])
def test_shared_engine_routes_and_replays_new_binding_without_changing_old_contract(experiment, tmp_path, variant):
    packets, value, mains = fixture(variant)
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl", [_output(experiment, p, value, variant) for p in batch]
            )

    ops = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), variant)
    assert ops["valid"] == ops["requested"] == 3
    assert ops["called_pairs"] == (1 if variant == "coverage" else 3)
    assert ops["judge_retried_pairs"] == 0
    rows, _ = experiment.replay_cell(tmp_path, "cell", variant, packets, mains)
    assert len(rows) == 3
    requested = next(r for r in rows if r["canonical_pair_id"] == "p")
    assert requested["a_can_replace_b"] == requested["b_can_replace_a"] == "YES"
    if variant == "coverage":
        assert requested["coverage_contract"] == subject.CONTRACT
        assert "selection_response_sha256" in requested
        assert "compiled_coverage_sha256" in requested
    for r in rows:
        if r["attempts"] == 0:
            assert "selection_response_sha256" not in r
            assert "compiled_coverage_sha256" not in r


def test_wrong_selection_retries_only_called_pair_and_preserves_actual_feedback(experiment, tmp_path):
    packets, value, mains = fixture("coverage")
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            v = deepcopy(value)
            if len(batches) == 1:
                v["b_meaning_in_a"]["counterpart_span_id"] = "B999"
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl", [_output(experiment, p, v, "coverage") for p in batch]
            )

    ops = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), "coverage")
    assert [[p["canonical_pair_id"] for p in b] for b in batches] == [["p"], ["p"]]
    assert batches[1][0]["repair_feedback"].startswith("Validation issue: ")
    assert ops["called_pair_retry_rate"] == 1.0
    assert ops["valid"] == 3
    assert len(experiment.replay_cell(tmp_path, "cell", "coverage", packets, mains)[0]) == 3


@pytest.mark.parametrize("failure", ["payload", "feedback", "prompt", "trace", "old_v1", "extra_field"])
def test_original_transport_prompt_and_raw_contract_are_not_silently_repaired(experiment, failure):
    from data_designer.engine.models.parsers.errors import ParserException

    packets, value, mains = fixture("coverage")
    called, _ = experiment.layout(packets, mains, "coverage")
    row = deepcopy(_output(experiment, called[0], value, "coverage"))
    if failure == "payload":
        row["payload"]["document_a"]["text"] += "changed"
    elif failure == "feedback":
        row["repair_feedback"] = None
    elif failure == "prompt":
        row[COLUMN + "__trace"][1]["content"] += "changed"
    elif failure == "trace":
        row.pop(COLUMN + "__trace")
    else:
        bad = deepcopy(value)
        if failure == "old_v1":
            bad["contract_version"] = "dedup-retained-coverage-v1"
        else:
            bad["source_quote"] = "unrequested quote"
        row[COLUMN + "__trace"][-1]["content"] = "```json\n" + json.dumps(bad) + "\n```"
    with pytest.raises((DedupEvaluationError, ParserException)):
        experiment.bind(called, [row], mains, "coverage")


def test_inventory_failure_stops_before_any_request(experiment, tmp_path):
    from eval.dedup.validation import sha256_json

    packets, _, mains = fixture("coverage")
    packets = deepcopy(packets)
    p = packets[0]
    p["payload"]["semantic_diff_evidence"]["spans"][-1]["text"] += "broken"
    p["judge_payload_hash"] = sha256_json(p["payload"])
    with pytest.raises(DedupEvaluationError):
        experiment.run_cell(tmp_path / "cell", packets, mains, None, _Relay(), _settings(), "coverage")
    assert not list(tmp_path.rglob("input.jsonl"))


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize(
    "feedback", [None, {"code": "SELECTION_REFERENCE_INVALID", "message": "Retry", "details": {}}]
)
def test_full_ray_http_boundary_for_both_arms_and_feedback_states(experiment, httpserver, tmp_path, variant, feedback):
    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray address required for full pipeline integration")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    packets, value, mains = fixture(variant)
    if variant == "coverage":
        from eval.dedup.validation import sha256_json
        from tests.eval.dedup.test_coverage_witness import _case, _uncover

        main, witness, payload = _case("FAQ Model Z. กฎหมายแพ่งและพาณิชย์ว่าด้วยหุ้นส่วนบริษัท", "FAQ Model Z.")
        _uncover(witness, payload, "A")
        value = selection(witness)
        packets[0].update(payload=payload, judge_payload_hash=sha256_json(payload))
        mains["p"] = main
    called, owned = experiment.layout(packets, mains, variant)
    for p in called:
        p["repair_feedback"] = encode_repair_feedback(feedback)
    response = "```json\n" + json.dumps(value) + "\n```"
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "id": "selection-boundary",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    settings = {**_settings(), "logical_model": "selection-boundary", "ray_temp_dir": "/raid/hfang/ihb/r23diag"}
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
    assert len(raw) == len(httpserver.log) == len(called)
    predictions = experiment.bind(called, raw, mains, variant)
    assert len(predictions) + len(owned) == 3
    if variant == "coverage":
        assert predictions[0]["relation_type"] == "CONTAINMENT"
        assert predictions[0]["a_can_replace_b"] == "YES"
        assert predictions[0]["b_can_replace_a"] == "NO"
    expected = [experiment.renderers[variant](p) for p in called]
    actual = []
    for request, _ in httpserver.log:
        body = json.loads(request.get_data())
        assert body["max_tokens"] == 4096
        assert body["temperature"] == 0
        assert body["top_p"] == 1
        actual.append(
            [{"role": m["role"], "content": subject.native.trace_text(m["content"])} for m in body["messages"]]
        )
    assert sorted(map(json.dumps, actual)) == sorted(map(json.dumps, expected))
