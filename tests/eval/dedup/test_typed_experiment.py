# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis import typed_experiment as subject
from eval.dedup.analysis.coverage_followup import trace_text
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl, blind_rows
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.local_ndd import _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.judging.typed_coverage import typed_coverage_schema
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_followup import _Relay, _settings
from tests.eval.dedup.test_coverage_selection import selection
from tests.eval.dedup.test_coverage_witness import _case, _uncover
from tests.eval.dedup.test_typed_coverage import typed


@pytest.fixture
def experiment():
    return subject.TypedExperiment(subject.common.ANALYSIS / "v06226_experiment.json")


def fixture(variant):
    inputs, mains, values = [], {}, {}
    for pid, a, b in (
        ("covered", "Covered probe. Cookies need consent.", "Covered probe. Cookie 需要获得同意。"),
        ("uncovered", "Uncovered probe. FAQ. กฎหมายแพ่งและพาณิชย์ว่าด้วยหุ้นส่วนบริษัท", "Uncovered probe. FAQ."),
        ("uncertain", "Uncertain probe. Red.", "Uncertain probe. Blue."),
        ("exact", "Exact probe.", "Exact probe."),
        ("negative", "Negative probe. Left.", "Negative probe. Right."),
    ):
        main, value, payload = _case(a, b)
        if pid == "uncovered":
            _uncover(value, payload, "A")
        elif pid == "uncertain":
            value["b_meaning_in_a"].update(
                status="UNRESOLVED", coverage_mode="NOT_COVERED", coverage_counterpart_ids=""
            )
        elif pid == "negative":
            main["span_shared_basis"]["score"] = "none"
        flat = selection(value)
        values[pid] = typed(flat) if variant == "coverage" else flat
        inputs.append({"canonical_pair_id": pid, "payload": payload})
        mains[pid] = main
    packets = [{**p, "repair_feedback": ""} for p in blind_rows(inputs, repeat=1)]
    return packets, values, mains


def output(experiment, packet, value, variant, *, padding=False):
    column = deepcopy(value)
    if padding:
        fields = set().union(*(d["properties"] for d in typed_coverage_schema()["$defs"].values()))
        for key in ("a_meaning_in_b", "b_meaning_in_a"):
            column[key].update(dict.fromkeys(fields - column[key].keys()))
    return {
        **packet,
        COLUMN: column,
        COLUMN + "__trace": [
            *experiment.renderers[variant](packet),
            {"role": "assistant", "content": "```json\n" + json.dumps(value) + "\n```"},
        ],
    }


@pytest.mark.parametrize("variant", ["control", "coverage"])
def test_all_typed_branches_and_owned_outputs_survive_shared_engine_replay(experiment, tmp_path, variant):
    packets, values, mains = fixture(variant)

    class Runtime:
        def run(self, **kwargs):
            rows = [
                output(experiment, p, values[p["canonical_pair_id"]], variant, padding=variant == "coverage")
                for p in _jsonl(Path(kwargs["input_path"]))
            ]
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    ops = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), variant)
    assert ops["called_pairs"] == 3
    assert ops["not_requested_owned_pairs"] == 2
    assert ops["valid"] == ops["requested"] == 5
    assert ops["called_pair_retry_rate"] == 0
    rows, _ = experiment.replay_cell(tmp_path, "cell", variant, packets, mains)
    by_id = {r["canonical_pair_id"]: r for r in rows}
    assert by_id["covered"]["a_can_replace_b"] == by_id["covered"]["b_can_replace_a"] == "YES"
    assert by_id["uncovered"]["a_can_replace_b"] == "YES"
    assert by_id["uncovered"]["b_can_replace_a"] == "NO"
    assert by_id["uncertain"]["same_duplicate_group"] == "UNRESOLVED"
    assert by_id["uncertain"]["confidence_tier"] == "LOW"
    assert by_id["negative"]["same_duplicate_group"] == "NO"
    assert by_id["exact"]["relation_type"] == "EXACT"
    assert all("raw_output_sha256" not in by_id[k] for k in ("negative", "exact"))
    if variant == "coverage":
        assert by_id["covered"]["response_transport"]["representation_changes"]["added_null_keys"] > 0


@pytest.mark.parametrize("variant", ["control", "coverage"])
def test_contract_retry_only_resubmits_invalid_pair_with_bound_feedback(experiment, tmp_path, variant):
    packets, values, mains = fixture(variant)
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            rows = []
            for p in batch:
                value = deepcopy(values[p["canonical_pair_id"]])
                if len(batches) == 1 and p["canonical_pair_id"] == "uncovered":
                    value["a_meaning_in_b"]["source_span_id"] = "A999"
                rows.append(output(experiment, p, value, variant))
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    ops = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), variant)
    assert len(batches) == 2
    assert [r["canonical_pair_id"] for r in batches[1]] == ["uncovered"]
    assert batches[1][0]["repair_feedback"].startswith("Validation issue: ")
    assert ops["called_pair_retry_rate"] == 1 / 3
    assert len(experiment.replay_cell(tmp_path, "cell", variant, packets, mains)[0]) == 5


@pytest.mark.parametrize(
    "failure", ["raw_null", "changed_echo", "unknown_null", "payload", "feedback", "prompt", "main_leak"]
)
def test_raw_contract_and_transport_cannot_be_silently_repaired(experiment, failure):
    from data_designer.engine.models.parsers.errors import ParserException

    packets, values, mains = fixture("coverage")
    packet = next(p for p in packets if p["canonical_pair_id"] == "covered")
    row = output(experiment, packet, values["covered"], "coverage", padding=True)
    row = deepcopy(row)
    if failure == "raw_null":
        bad = deepcopy(values["covered"])
        bad["a_meaning_in_b"]["source_span_id"] = None
        row[COLUMN + "__trace"][-1]["content"] = "```json\n" + json.dumps(bad) + "\n```"
    elif failure == "changed_echo":
        row[COLUMN]["a_meaning_in_b"]["status"] = "UNRESOLVED"
    elif failure == "unknown_null":
        row[COLUMN]["a_meaning_in_b"]["invented"] = None
    elif failure == "payload":
        row["payload"]["document_a"]["text"] += "changed"
    elif failure == "feedback":
        row["repair_feedback"] = None
    elif failure == "prompt":
        row[COLUMN + "__trace"][1]["content"] += "changed"
    else:
        row["qwen_dedup_semantic_judge"] = mains["covered"]
    with pytest.raises((DedupEvaluationError, ParserException)):
        experiment.bind([packet], [row], mains, "coverage")


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize(
    "feedback", [None, {"code": "SELECTION_REFERENCE_INVALID", "message": "Retry", "details": {}}]
)
def test_full_mixed_branch_native_ray_http_first_and_retry(experiment, httpserver, tmp_path, variant, feedback):
    from werkzeug.wrappers import Response

    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray address required for full pipeline integration")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    packets, values, mains = fixture(variant)
    called, owned = experiment.layout(packets, mains, variant)
    for p in called:
        p["repair_feedback"] = encode_repair_feedback(feedback)

    def respond(request):
        body = json.loads(request.get_data())
        text = "\n".join(trace_text(m["content"]) for m in body["messages"])
        pid = next(pid for pid in ("covered", "uncovered", "uncertain") if pid.capitalize() + " probe." in text)
        value = values[pid]
        return Response(
            json.dumps(
                {
                    "id": "typed-boundary",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "```json\n" + json.dumps(value) + "\n```"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ),
            mimetype="application/json",
        )

    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(respond)
    settings = {**_settings(), "logical_model": "typed-boundary", "ray_temp_dir": "/raid/hfang/ihb/r26diag"}
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
    assert len(raw) == len(httpserver.log) == len(called) == 3
    rows = experiment.bind(called, raw, mains, variant)
    by_id = {r["canonical_pair_id"]: r for r in rows}
    assert len(rows) + len(owned) == 5
    assert by_id["covered"]["same_duplicate_group"] == "YES"
    assert by_id["uncovered"]["relation_type"] == "CONTAINMENT"
    assert by_id["uncovered"]["a_can_replace_b"] == "YES"
    assert by_id["uncovered"]["b_can_replace_a"] == "NO"
    assert by_id["uncertain"]["confidence_tier"] == "LOW"
    actual = []
    for request, _ in httpserver.log:
        body = json.loads(request.get_data())
        assert body["max_tokens"] == 4096
        assert body["temperature"] == 0
        assert body["top_p"] == 1
        actual.append([{"role": m["role"], "content": trace_text(m["content"])} for m in body["messages"]])
    expected = [experiment.renderers[variant](p) for p in called]
    assert sorted(map(json.dumps, actual)) == sorted(map(json.dumps, expected))
    if variant == "coverage":
        assert all(r[COLUMN] == values[r["canonical_pair_id"]] for r in raw)
        assert all(r["response_transport"]["representation_changes"] == {} for r in rows)


def test_unknown_response_padding_is_a_terminal_boundary_not_a_model_retry(experiment, tmp_path):
    packets, values, mains = fixture("coverage")
    calls = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            calls.append(batch)
            rows = [output(experiment, p, values[p["canonical_pair_id"]], "coverage") for p in batch]
            next(r for r in rows if r["canonical_pair_id"] == "covered")[COLUMN]["a_meaning_in_b"]["invented"] = None
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    ops = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), "coverage")
    assert len(calls) == 1
    assert ops["fatal_boundary"]
    assert ops["errors"] == 1
    assert ops["retried"] == 0
