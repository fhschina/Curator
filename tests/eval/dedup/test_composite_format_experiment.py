# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis import composite_experiment as old
from eval.dedup.analysis.composite_format_experiment import (
    FormatArm,
    compare,
    critic_inputs,
    preflight_passes,
    prompt_contrast,
    run_components,
    validate,
)
from eval.dedup.analysis.coverage_followup import trace_text
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl
from eval.dedup.judging.composite_format_runtime import VERSION
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.local_ndd import _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic
from tests.eval.dedup.test_composite_experiment import fixture, output
from tests.eval.dedup.test_coverage_followup import _Relay, _settings


def test_fresh_main_critic_replay_version_and_counterfactual(tmp_path):
    inputs, values, _ = fixture("main")
    seen = []

    class Runtime:
        def __init__(self, arm):
            self.arm = arm

        def run(self, **kwargs):
            packets = _jsonl(Path(kwargs["input_path"]))
            seen.extend((self.arm.stage, p) for p in packets)
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl",
                [output(self.arm, p, values[p["canonical_pair_id"]]) for p in packets],
            )

    root = tmp_path / "cell"
    ops = run_components(root, inputs, lambda arm: nullcontext(Runtime(arm)), _Relay(), _settings())
    assert ops["complete"]
    assert ops["main"]["called_pairs"] == 4
    assert ops["critic"]["called_pairs"] == 2
    mains, _ = FormatArm("main").replay_cell(root, "main", "coverage", inputs, {})
    assert all(r["diagnostic_version"] == VERSION for r in mains)
    packets, certificates = critic_inputs(inputs, mains)
    finals, _ = FormatArm("critic").replay_cell(root, "critic", "coverage", packets, certificates)
    assert all(r["diagnostic_version"] == VERSION for r in finals)
    assert all(p["repair_feedback"] == "" for stage, p in seen if stage == "critic")
    labels = [
        {
            "canonical_pair_id": r["canonical_pair_id"],
            "review_id": r["canonical_pair_id"],
            "stratum_population_n": 1,
            "stratum_sample_n": 1,
            **{
                f"human_{k}": r[k]
                for k in (
                    "same_duplicate_group",
                    "a_can_replace_b",
                    "b_can_replace_a",
                    "relation_type",
                    "material_difference",
                )
            },
        }
        for r in mains
    ]
    assert set(compare(root, inputs, labels)["transitions"]) == {"CORRECT -> CORRECT"}
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_ROOT_EXISTS"):
        run_components(root, inputs, None, None, {})
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_MAIN_PROVENANCE"):
        critic_inputs(inputs, [{**r, "diagnostic_version": old.VERSION} for r in mains])


@pytest.mark.parametrize("stage", ["main", "critic"])
@pytest.mark.parametrize("field", ["harmless_unique_ids", "opposite_support_ids"])
@pytest.mark.parametrize("value", ["", None])
def test_original_cross_branch_fields_remain_invalid_even_if_writer_prunes(stage, field, value):
    from data_designer.engine.models.parsers.errors import ParserException

    arm = FormatArm(stage)
    packets, values, mains = fixture(stage)
    packet = next(p for p in packets if p["canonical_pair_id"] == "uncovered")
    response = values["uncovered"]
    row = output(arm, packet, response)
    bad = deepcopy(response)
    bad["b_meaning_in_a"][field] = value
    row[COLUMN + "__trace"][-1]["content"] = "```json\n" + json.dumps(bad) + "\n```"
    with pytest.raises(ParserException):
        arm.bind([packet], [row], mains, "coverage")


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_old_prompt_cannot_be_relabelled_as_current_response(stage):
    packet, values, mains = fixture(stage)
    row = output(old.CompositeArm(stage), packet[0], values[packet[0]["canonical_pair_id"]])
    with pytest.raises(DedupEvaluationError, match="FOLLOWUP_BOUNDARY_PROMPT"):
        FormatArm(stage).bind([packet[0]], [row], mains, "coverage")


def test_prompt_contrast_checks_all_frozen_input_content():
    packets, _, _ = fixture("main")
    assert len(prompt_contrast(packets)) == len(packets)


def test_gate_rejects_retry_semantic_regression_and_incomplete_cells():
    stage = {"judge_retried_pairs": 0, "http_statuses": {"200": 8}, "called_pairs": 8}
    ops = {"complete": True, "main": deepcopy(stage), "critic": deepcopy(stage)}
    comparison = {"transitions": {"CORRECT -> CORRECT": {"count": 8}}}
    assert preflight_passes(ops, comparison)
    ops["main"]["judge_retried_pairs"] = 1
    assert not preflight_passes(ops, comparison)
    ops["main"]["judge_retried_pairs"] = 0
    assert not preflight_passes(ops, {"transitions": {"CORRECT -> WRONG": {"count": 8}}})
    assert not preflight_passes(ops | {"complete": False}, comparison)


def test_frozen_artifacts_and_manifest_cannot_change(tmp_path):
    from eval.dedup.rejudge_comparison import _source_digest

    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"immutable": True})
    m = {
        "version": VERSION,
        "source_implementation_sha256": _source_digest(),
        "frozen_files": {str(path): sha256_file(path)},
        "input_files": {},
    }
    m["contract_digest"] = sha256_json(m)
    write_json_atomic(tmp_path / "manifest.json", m)
    assert validate(tmp_path) == m
    path.write_text(json.dumps({"immutable": False}))
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_FORMAT_FREEZE"):
        validate(tmp_path)


@pytest.mark.parametrize("fail_preflight", [False, True])
def test_schedule_writes_immutable_snapshots_and_stops_after_failed_preflight(tmp_path, monkeypatch, fail_preflight):
    import transformers

    from eval.dedup.analysis import composite_format_experiment as subject
    from eval.dedup.judging import request_relay

    packets, _, _ = fixture("main")
    for repeat in (1, 2):
        _write_jsonl(tmp_path / f"input_repeat_{repeat}.jsonl", packets)
    write_json_atomic(tmp_path / "labels_private.json", {"human_development": [], "synthetic_policy": []})
    settings = {
        "api_key_env": "FORMAT_TEST_KEY",
        "logical_model": "test",
        "hub_base_url": "http://127.0.0.1:1/v1",
        "hub_model": "test",
        "timeout_seconds": 1,
    }
    m = {
        "settings": settings,
        "schedule": ["preflight", "repeat_1", "repeat_2"],
        "preflight_ids": [p["canonical_pair_id"] for p in packets],
        "contract_digest": "test-digest",
    }
    monkeypatch.setenv("FORMAT_TEST_KEY", "local-test-only")
    monkeypatch.setenv("RAY_ADDRESS", "test-only-no-connection")
    monkeypatch.setattr(subject, "validate", lambda _root: m)

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return range(1)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    monkeypatch.setattr(request_relay, "RequestRelay", lambda **_kwargs: nullcontext(_Relay()))
    observed = []

    def component(root, *_args, **_kwargs):
        observed.append(root.name)
        stage = {"called_pairs": 8, "http_statuses": {"200": 8}, "judge_retried_pairs": int(fail_preflight)}
        return {"complete": True, "main": stage, "critic": stage}

    monkeypatch.setattr(subject, "run_components", component)
    monkeypatch.setattr(subject, "compare", lambda *_args: {"transitions": {"CORRECT -> CORRECT": {"count": 8}}})
    if fail_preflight:
        with pytest.raises(DedupEvaluationError, match="COMPOSITE_PREFLIGHT_FAILED"):
            subject.run(tmp_path)
        assert observed == ["preflight"]
        assert (tmp_path / "stopped.json").exists()
        assert not (tmp_path / "assessment.json").exists()
    else:
        report = subject.run(tmp_path)
        assert observed == m["schedule"]
        assert list(report["cells"]) == m["schedule"]
        assert len(list(tmp_path.glob("schedule_after_*.json"))) == 3
        assert (tmp_path / "assessment.json").exists()
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_STARTED"):
        subject.run(tmp_path)


@pytest.mark.parametrize("stage", ["main", "critic"])
@pytest.mark.parametrize("retry", [False, True])
def test_native_ray_http_writer_binder_first_and_retry(stage, retry, httpserver, tmp_path):
    from werkzeug.wrappers import Response

    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray cluster required")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    arm = FormatArm(stage)
    inputs, values, mains = fixture(stage)
    called, owned = arm.layout(inputs, mains, "coverage")
    for p in called:
        p["repair_feedback"] = encode_repair_feedback(
            {"code": "COMPOSITE_CONFLICT_MISSING", "message": "Retry", "details": {}} if retry else None
        )

    def respond(request):
        body = json.loads(request.get_data())
        text = "\n".join(trace_text(m["content"]) for m in body["messages"])
        pid = next(p["canonical_pair_id"] for p in called if p["canonical_pair_id"].capitalize() + " probe." in text)
        return Response(
            json.dumps(
                {
                    "id": "format-boundary",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "```json\n" + json.dumps(values[pid]) + "\n```",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ),
            mimetype="application/json",
        )

    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(respond)
    _write_jsonl(tmp_path / "input.jsonl", called)
    settings = _settings() | {"logical_model": "format-boundary", "ray_temp_dir": "/raid/hfang/ihb/r30diag"}
    with arm.runtime("coverage", httpserver.url_for("/v1"), settings) as runtime:
        runtime.run(
            input_path=str(tmp_path / "input.jsonl"),
            input_format="jsonl",
            output_path=str(tmp_path / "output"),
            checkpoint_path=str(tmp_path / "checkpoints"),
            files_per_partition=1,
        )
    raw = _read_output_rows(tmp_path / "output")
    bound = arm.bind(called, raw, mains, "coverage")
    assert len(bound) + len(owned) == 5
    assert len(raw) == len(httpserver.log) == len(called)
    assert all(r["diagnostic_version"] == VERSION for r in bound)
    actual = []
    for request, _ in httpserver.log:
        body = json.loads(request.get_data())
        assert body["temperature"] == 0
        assert body["top_p"] == 1
        assert body["max_tokens"] == 4096
        actual.append([{"role": m["role"], "content": trace_text(m["content"])} for m in body["messages"]])
    assert sorted(map(json.dumps, actual)) == sorted(json.dumps(arm.renderers["coverage"](p)) for p in called)
