# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis import coverage_experiment as subject
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl, blind_rows
from eval.dedup.judging.coverage_witness import COLUMN, adapt_coverage_witness, parse_coverage
from eval.dedup.judging.local_ndd import RECORD_BINDING_CRITIC_COLUMN, _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError, sha256_file, write_json_atomic
from tests.eval.dedup.test_coverage_followup import _fixture, _Relay, _settings
from tests.eval.dedup.test_coverage_witness import _case, _uncover


@pytest.fixture
def experiment():
    return subject.CoverageExperiment(subject.common.ANALYSIS / "v06222_experiment.json")


def _output(experiment, packet, value, variant):
    column = COLUMN if variant == "coverage" else RECORD_BINDING_CRITIC_COLUMN
    return {
        **packet,
        column: value,
        column + "__trace": [
            *experiment.renderers[variant](packet),
            {"role": "assistant", "content": "```json\n" + json.dumps(value) + "\n```"},
        ],
    }


def _mixed(variant):
    packets, value, mains = _fixture(variant)
    negative = deepcopy(mains["p"])
    negative["span_shared_basis"]["score"] = "none"
    exact, _, payload = _case("Same notice.", "Same notice.", non_main=True)
    packets += blind_rows(
        [
            {"canonical_pair_id": "negative", "payload": packets[0]["payload"]},
            {"canonical_pair_id": "exact", "payload": payload},
        ],
        repeat=1,
    )
    for p in packets:
        p["repair_feedback"] = ""
    return packets, value, {**mains, "negative": negative, "exact": exact}


@pytest.mark.parametrize("variant", subject.VARIANTS)
def test_routing_retains_all_outputs_and_preserves_full_control_calls(experiment, tmp_path, variant):
    packets, value, mains = _mixed(variant)
    called, owned = experiment.layout(packets, mains, variant)
    assert len(called) == (1 if variant == "coverage" else 3)
    assert len(called) + len(owned) == 3
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl", [_output(experiment, p, value, variant) for p in batch]
            )

    complete = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), variant)
    assert complete["requested"] == complete["valid"] == 3
    assert complete["called_pairs"] == len(called)
    assert complete["called_final_contract_completion"] == 1
    assert complete["called_pair_retry_rate"] == 0
    rows, _ = experiment.replay_cell(tmp_path, "cell", variant, packets, mains)
    assert {r["canonical_pair_id"] for r in rows} == {p["canonical_pair_id"] for p in packets}
    assert len(batches) == 1
    for row in rows:
        if row["critic_request_status"] == subject.NOT_REQUESTED:
            assert row["attempts"] == 0
            assert not (
                {"coverage_response_sha256", "native_corrections", "native_trace_sha256", "raw_output_sha256"}
                & row.keys()
            )


def test_owned_only_cell_never_calls_runtime_or_claims_schema_success(experiment, tmp_path):
    packets, _, mains = _mixed("coverage")
    packets = [p for p in packets if p["canonical_pair_id"] != "p"]

    class Runtime:
        def run(self, **kwargs):
            pytest.fail("owned branch called a model")

    complete = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), "coverage")
    assert complete["requested"] == complete["valid"] == 2
    assert complete["called_pairs"] == 0
    assert complete["called_final_contract_completion"] is None
    assert complete["called_pair_retry_rate"] is None
    assert len(experiment.replay_cell(tmp_path, "cell", "coverage", packets, mains)[0]) == 2


def test_actual_call_denominator_and_exact_retry_response_are_preserved(experiment, tmp_path):
    packets, value, mains = _mixed("coverage")
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            v = deepcopy(value)
            if len(batches) == 1:
                v["b_meaning_in_a"].pop("coverage_counterpart_ids")
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl", [_output(experiment, p, v, "coverage") for p in batch]
            )

    complete = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), "coverage")
    assert [[p["canonical_pair_id"] for p in b] for b in batches] == [["p"], ["p"]]
    assert batches[1][0]["repair_feedback"].startswith("Validation issue: ")
    assert complete["valid"] == 3
    assert complete["retried"] == complete["judge_retried_pairs"] == 1
    assert complete["called_pair_retry_rate"] == 1.0
    rows, _ = experiment.replay_cell(tmp_path, "cell", "coverage", packets, mains)
    assert next(r for r in rows if r["canonical_pair_id"] == "p")["attempts"] == 2
    with pytest.raises(DedupEvaluationError, match="reuse"):
        experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), "coverage")


@pytest.mark.parametrize("failure", ["payload", "feedback", "messages", "trace"])
def test_deterministic_boundary_failure_stops_without_extra_calls(experiment, tmp_path, failure):
    packets, value, mains = _mixed("coverage")
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            row = deepcopy(_output(experiment, batch[0], value, "coverage"))
            if failure == "payload":
                row["payload"]["document_a"]["text"] += " changed"
            elif failure == "feedback":
                row["repair_feedback"] = None
            elif failure == "messages":
                row[COLUMN + "__trace"][1]["content"] += " changed"
            else:
                row.pop(COLUMN + "__trace")
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", [row])

    complete = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), "coverage")
    assert len(batches) == 1
    assert complete["fatal_boundary"]
    assert complete["errors"] == 1
    assert complete["retried"] == 0
    assert complete["valid"] == 2
    assert len(experiment.replay_cell(tmp_path, "cell", "coverage", packets, mains)[0]) == 2


@pytest.mark.parametrize("variant", subject.VARIANTS)
def test_pruned_extra_native_field_is_rejected(experiment, variant):
    from data_designer.engine.models.parsers.errors import ParserException

    packets, value, mains = _fixture(variant)
    row = _output(experiment, packets[0], value, variant)
    column = COLUMN if variant == "coverage" else RECORD_BINDING_CRITIC_COLUMN
    row[column + "__trace"][-1]["content"] = "```json\n" + json.dumps({**value, "hidden_extra": "invalid"}) + "\n```"
    with pytest.raises((DedupEvaluationError, ParserException)):
        experiment.bind(packets, [row], mains, variant)


def test_owned_replay_rejects_fabricated_proof_even_with_updated_artifact_digest(experiment, tmp_path):
    packets, _, mains = _mixed("coverage")
    packets = [p for p in packets if p["canonical_pair_id"] != "p"]
    complete = experiment.run_cell(tmp_path / "cell", packets, mains, None, _Relay(), _settings(), "coverage")
    path = tmp_path / "cell/predictions.jsonl"
    rows = _jsonl(path)
    rows[0]["raw_output_sha256"] = "fabricated"
    path.unlink()
    _write_jsonl(path, rows)
    complete["artifacts"][str(path)] = sha256_file(path)
    (tmp_path / "cell/complete.json").unlink()
    write_json_atomic(tmp_path / "cell/complete.json", complete)
    with pytest.raises(DedupEvaluationError, match="cannot claim"):
        experiment.replay_cell(tmp_path, "cell", "coverage", packets, mains)


def test_call_set_and_preflight_are_based_on_original_order_not_labels(experiment):
    packets, _, mains = _mixed("coverage")
    for i in range(7):
        packets.append({**packets[0], "canonical_pair_id": f"q{i}"})
        mains[f"q{i}"] = mains["p"]
    inputs = {1: packets, 2: list(reversed(packets))}
    calls = experiment.call_sets(inputs, mains)
    assert calls["preflight_pair_ids"]["coverage"] == ["p", *[f"q{i}" for i in range(7)]]
    assert calls["repeat_2_coverage"]["called_pair_ids"] == list(
        reversed(calls["repeat_1_coverage"]["called_pair_ids"])
    )
    assert len(calls["repeat_1_coverage"]["all_pair_ids"]) == 10


def test_two_separated_unique_spans_cannot_be_assembled_into_a_quote(experiment):
    from data_designer.engine.models.parsers.errors import ParserException

    a = "FAQ Model Z. Enable logging. Shared instructions for the original device. Keep backups."
    b = "FAQ Model Z. Shared instructions for the original device."
    main, value, payload = _case(a, b)
    own = [s for s in payload["semantic_diff_evidence"]["spans"] if s.get("side") == "A"]
    assert len(own) == 2
    _uncover(value, payload, "A")
    side = value["a_meaning_in_b"]
    side["source_ids"] = " ".join(s["span_id"] for s in own)
    side["source_quote"] = "\n".join(s["text"] for s in own)
    with pytest.raises(DedupEvaluationError, match="COVERAGE_QUOTE_INVALID"):
        parse_coverage(value, payload)
    side["source_ids"], side["source_quote"] = own[0]["span_id"], own[0]["text"]
    parse_coverage(value, payload)
    result = adapt_coverage_witness(main, value, payload)
    assert result["relation_type"] == "CONTAINMENT"
    assert result["a_can_replace_b"] == "YES"
    assert result["b_can_replace_a"] == "NO"
    packets = blind_rows([{"canonical_pair_id": "p", "payload": payload}], repeat=1)
    packets[0]["repair_feedback"] = ""
    bound = experiment.bind(packets, [_output(experiment, packets[0], value, "coverage")], {"p": main}, "coverage")[0]
    assert bound["relation_type"] == "CONTAINMENT"
    del side["coverage_counterpart_ids"]
    with pytest.raises((DedupEvaluationError, ParserException), match="coverage_counterpart_ids"):
        experiment.bind(packets, [_output(experiment, packets[0], value, "coverage")], {"p": main}, "coverage")


def test_control_native_retry_failure_is_not_hidden_by_candidate_semantic_success():
    cell = {
        "local_gates": {"passed": True},
        "targets": {"passed": True},
        "metrics": {"weighted": {"primary_decision_exact": 0.9}, "over_group": 1},
        "operations": {"called_pair_retry_rate": 0.0},
    }
    assert all(subject.paired_checks(cell, cell).values())
    control = {**cell, "operations": {"called_pair_retry_rate": 0.02}}
    checks = subject.paired_checks(cell, control)
    assert checks["local_gates"]
    assert not checks["control_called_retry_rate_at_most_one_percent"]


@pytest.mark.parametrize("variant", subject.VARIANTS)
@pytest.mark.parametrize("feedback", [None, {"code": "QUOTE_INVALID", "message": "Retry", "details": {}}])
def test_full_ray_http_boundary_for_both_arms_and_feedback_states(experiment, httpserver, tmp_path, variant, feedback):
    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray address required for full pipeline integration")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    packets, value, mains = _mixed(variant)
    called, owned = experiment.layout(packets, mains, variant)
    for p in called:
        p["repair_feedback"] = encode_repair_feedback(feedback)
    response = "```json\n" + json.dumps(value) + "\n```"
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "id": "routed-coverage-boundary",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    settings = {**_settings(), "logical_model": "routed-coverage-boundary", "ray_temp_dir": "/raid/hfang/ihb/r22diag"}
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
                "judge": {"temperature": 0, "top_p": 1, "max_tokens": 4096, "max_parallel_requests": 2}
            },
        )
    raw = _read_output_rows(tmp_path / "output")
    assert len(raw) == len(httpserver.log) == len(called)
    predictions = experiment.bind(called, raw, mains, variant)
    assert len(predictions) + len(owned) == 3
    assert all(p["payload_transport"]["representation_changes"]["added_null_keys"] > 0 for p in predictions)
    expected = [experiment.renderers[variant](p) for p in called]
    actual = []
    for request, _ in httpserver.log:
        messages = json.loads(request.get_data())["messages"]
        actual.append([{"role": m["role"], "content": subject.native.trace_text(m["content"])} for m in messages])
    assert sorted(map(json.dumps, actual)) == sorted(map(json.dumps, expected))
