# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from contextlib import nullcontext
from pathlib import Path

import pytest
from werkzeug.wrappers import Response

from eval.dedup.analysis import paced_development as subject
from eval.dedup.analysis.coverage_followup import trace_text
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl
from eval.dedup.analysis.judge_calibration import evaluate_predictions
from eval.dedup.judging.paced_relay import PacedRelay, TransportProfile
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic
from tests.eval.dedup.test_composite_experiment import output
from tests.eval.dedup.test_context_experiment import fixture
from tests.eval.dedup.test_coverage_followup import _Relay, _settings


def test_larger_client_guard_preserves_payload_prompt_and_rejects_over_budget():
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return range(45346)

    packets, _, _ = fixture("main")
    old, new = subject.semantic.DirectionalArm("main"), subject.PacedArm("main", tokenizer=Tokenizer())
    assert old.layout(packets, {}, "coverage") == new.layout(packets, {}, "coverage")
    assert old.renderers["coverage"](packets[0]) == new.renderers["coverage"](packets[0])

    class Oversized:
        def apply_chat_template(self, *_args, **_kwargs):
            return range(65536)

    with pytest.raises(DedupEvaluationError, match="PACED_TOKEN_BUDGET"):
        subject.PacedArm("main", tokenizer=Oversized()).layout(packets, {}, "coverage")


def test_terminal_missing_output_remains_in_recall_and_primary_denominators():
    inputs = [{"canonical_pair_id": str(i)} for i in range(2)]
    accepted = [inputs[0] | dict.fromkeys(subject.PRIMARY_FIELDS, "YES")]
    scoring = subject.accounted_predictions(inputs, accepted, ["1"])
    labels = [p | {"human_" + k: "YES" for k in subject.PRIMARY_FIELDS} for p in inputs]
    metrics = evaluate_predictions(labels, scoring)["weighted"]
    assert metrics["rows"] == 2
    assert metrics["duplicate_precision"] == 1
    assert metrics["duplicate_recall"] == metrics["primary_decision_exact"] == 0.5
    assert scoring[1]["metric_only_missing_output"]
    assert "metric_only_missing_output" not in accepted[0]
    with pytest.raises(DedupEvaluationError, match="PACED_DENOMINATOR"):
        subject.accounted_predictions(inputs, accepted, [])


def test_terminal_main_is_not_given_to_critic_and_is_not_forged_valid(tmp_path):
    packets, values, _ = fixture("main")
    failing = packets[-1]["canonical_pair_id"]
    seen = []

    class Runtime:
        def __init__(self, arm):
            self.arm = arm

        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            seen.extend((self.arm.stage, p["canonical_pair_id"]) for p in batch)
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl",
                [
                    output(self.arm, p, values[p["canonical_pair_id"]])
                    for p in batch
                    if p["canonical_pair_id"] != failing
                ],
            )

    root = tmp_path / "cell"
    ops = subject.run_components(root, packets, lambda arm: nullcontext(Runtime(arm)), _Relay(), _settings())
    assert not ops["complete"]
    assert ops["final_terminal_errors"] == 1
    assert ops["final_valid"] == len(packets) - 1
    assert ("critic", failing) not in seen
    assert all(p["canonical_pair_id"] != failing for p in _jsonl(root / "critic/predictions.jsonl"))
    assert len(_jsonl(root / "final_scoring_only.jsonl")) == len(packets)


def test_transport_summary_distinguishes_recovered_rate_limits_from_failure(tmp_path):
    events = [
        {
            "external_request": True,
            "sequence": i + 1,
            "logical_request_sequence": 1,
            "upstream_attempt": i + 1,
            "upstream_http_status": status,
            "admission_wait_seconds": 0,
        }
        for i, status in enumerate((429, 200))
    ]
    _write_jsonl(tmp_path / "events.jsonl", events)
    summary = subject.transport_receipts(tmp_path)
    assert summary["external_attempts"] == 2
    assert summary["transport_retry_attempts"] == 1
    assert summary["unrecovered_logical_requests"] == 0
    _write_jsonl(tmp_path / "failure/events.jsonl", [events[0] | {"sequence": 3, "logical_request_sequence": 2}])
    assert subject.transport_receipts(tmp_path)["unrecovered_logical_requests"] == 1


def test_freeze_rejects_changes_before_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(subject.semantic, "validate", lambda _root: {})
    artifact = tmp_path / "input.json"
    write_json_atomic(artifact, {"original": True})
    m = {
        "execution_contract": subject.EXECUTION_CONTRACT,
        "version": subject.VERSION,
        "input_files": {str(artifact): sha256_file(artifact)},
        "frozen_files": {},
    }
    m["contract_digest"] = sha256_json(m)
    write_json_atomic(tmp_path / "manifest.json", m)
    assert subject.validate(tmp_path) == m
    artifact.write_text("{}")
    with pytest.raises(DedupEvaluationError, match="PACED_FREEZE"):
        subject.validate(tmp_path)


@pytest.mark.parametrize("technical_failure", [False, True])
def test_frozen_schedule_reaches_full_only_after_technical_preflight(tmp_path, monkeypatch, technical_failure):
    import transformers

    from eval.dedup import cli

    packet = {"canonical_pair_id": "probe"}
    for name in ("input_repeat_1", "input_repeat_2", "input_full"):
        _write_jsonl(tmp_path / f"{name}.jsonl", [packet])
    write_json_atomic(tmp_path / "labels_local_private.json", {"synthetic": [packet]})
    write_json_atomic(tmp_path / "labels_full_private.json", [packet])
    manifest = {
        "settings": {
            "api_key_env": "PACED_TEST_KEY",
            "logical_model": "test",
            "hub_model": "test",
            "hub_base_url": "http://127.0.0.1:1/v1",
            "timeout_seconds": 1,
        },
        "transport_profile": {},
        "contract_digest": "test",
        "preflight_ids": ["probe"],
        "long_input_id": "probe",
        "schedule": ["preflight", "long_input", "repeat_1", "repeat_2", "full"],
    }
    monkeypatch.setenv("PACED_TEST_KEY", "local-only")
    monkeypatch.setenv("RAY_ADDRESS", "test-only")
    monkeypatch.setattr(cli, "_load_repository_env", lambda _path: None)
    monkeypatch.setattr(subject, "validate", lambda _root: manifest)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "PacedRelay", lambda **_kwargs: nullcontext(_Relay()))
    seen = []

    def component(root, *_args, **_kwargs):
        seen.append(root.name)
        return {"complete": not technical_failure, "final_valid": int(not technical_failure), "transport": {}}

    monkeypatch.setattr(subject, "run_components", component)
    monkeypatch.setattr(subject, "compare", lambda *_args: {"deliberate_semantic_failure": True})
    if technical_failure:
        with pytest.raises(DedupEvaluationError, match="PACED_TECHNICAL_PREFLIGHT"):
            subject.run(tmp_path, env_file=tmp_path / "unused")
        assert seen == ["preflight"]
        assert (tmp_path / "stopped.json").exists()
        assert not (tmp_path / "assessment.json").exists()
    else:
        report = subject.run(tmp_path, env_file=tmp_path / "unused")
        assert seen == manifest["schedule"]
        assert list(report["cells"]) == manifest["schedule"]
        assert report["eligible_for_release"] is False
    with pytest.raises(DedupEvaluationError, match="PACED_STARTED"):
        subject.run(tmp_path, env_file=tmp_path / "unused")


@pytest.mark.parametrize("stage", ["main", "critic"])
@pytest.mark.parametrize("transport_retry", [False, True])
def test_native_paced_ray_http_writer_binding(stage, transport_retry, httpserver, tmp_path):
    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray cluster required")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    arm = subject.PacedArm(stage)
    packets, values, mains = fixture(stage)
    called, _ = arm.layout(packets, mains, "coverage")
    requests = []

    def respond(request):
        body = json.loads(request.get_data())
        text = "\n".join(trace_text(m["content"]) for m in body["messages"])
        pid = next(p["canonical_pair_id"] for p in called if p["canonical_pair_id"].capitalize() + " probe." in text)
        requests.append((pid, body))
        if transport_retry and len(requests) == 1:
            return Response("limited", status=429, headers={"Retry-After": "0.01"})
        return Response(
            json.dumps(
                {
                    "id": "paced-native",
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
    settings = _settings() | {"logical_model": "paced-native", "ray_temp_dir": "/raid/hfang/ihb/r32paced"}
    with (
        PacedRelay(
            profile=TransportProfile(
                min_interval_seconds=0.01, max_in_flight=2, retry_base_seconds=0.01, retry_cap_seconds=0.02
            ),
            logical_model=settings["logical_model"],
            upstream_model="local-native",
            upstream_base_url=httpserver.url_for("/v1"),
            upstream_api_key="private-local-test-key",
            timeout_seconds=600,
            expected_generation_parameters={
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ) as relay,
        arm.runtime("coverage", relay.endpoint, settings) as runtime,
    ):
        ops = arm.run_cell(tmp_path / "cell", packets, mains, runtime, relay, settings, "coverage")
    rows, _ = arm.replay_cell(tmp_path, "cell", "coverage", packets, mains)
    assert len(rows) == ops["valid"] == len(packets)
    assert ops["judge_retried_pairs"] == ops["native_corrections"] == 0
    assert len(requests) == len(httpserver.log) == len(called) + int(transport_retry)
    receipts = subject.transport_receipts(tmp_path / "cell")
    assert receipts["external_attempts"] == len(requests)
    assert receipts["transport_retry_attempts"] == int(transport_retry)
    assert receipts["unrecovered_logical_requests"] == receipts["local_rejections"] == 0
    expected = {p["canonical_pair_id"]: arm.renderers["coverage"](p) for p in called}
    for pid, body in requests:
        assert body["model"] == "local-native"
        assert [{"role": m["role"], "content": trace_text(m["content"])} for m in body["messages"]] == expected[pid]
    assert "private-local-test-key" not in "".join(p.read_text() for p in (tmp_path / "cell").rglob("*.jsonl"))
