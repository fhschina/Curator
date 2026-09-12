# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis import directional_experiment as subject
from eval.dedup.analysis.coverage_followup import trace_text
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl
from eval.dedup.judging.context_coverage import CONTRACT
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.directional_runtime import VERSION
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic
from tests.eval.dedup.test_composite_experiment import output
from tests.eval.dedup.test_context_experiment import fixture
from tests.eval.dedup.test_coverage_followup import _Relay, _settings


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_component_runs_and_replays_context_witnesses_with_original_raw_trace(stage, tmp_path):
    arm = subject.DirectionalArm(stage)
    packets, values, mains = fixture(stage)

    class Runtime:
        def run(self, **kwargs):
            rows = [output(arm, p, values[p["canonical_pair_id"]]) for p in _jsonl(Path(kwargs["input_path"]))]
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    ops = arm.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), "coverage")
    assert ops["valid"] == ops["requested"] == 7
    assert ops["called_pairs"] == (6 if stage == "main" else 5)
    assert ops["called_pair_retry_rate"] == 0
    rows, _ = arm.replay_cell(tmp_path, "cell", "coverage", packets, mains)
    assert all(r["diagnostic_version"] == VERSION for r in rows)
    for row in rows:
        if row["canonical_pair_id"].startswith("context_"):
            assert row["same_duplicate_group"] == "NO"
            assert row["coverage_response"] == values[row["canonical_pair_id"]]


def test_fresh_main_critic_attribution_never_reuses_old_predictions(tmp_path):
    packets, values, _ = fixture("main")
    seen = []

    class Runtime:
        def __init__(self, arm):
            self.arm = arm

        def run(self, **kwargs):
            rows = _jsonl(Path(kwargs["input_path"]))
            seen.extend((self.arm.stage, p) for p in rows)
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl",
                [output(self.arm, p, values[p["canonical_pair_id"]]) for p in rows],
            )

    root = tmp_path / "fresh"
    ops = subject.run_components(root, packets, lambda arm: nullcontext(Runtime(arm)), _Relay(), _settings())
    assert ops["complete"]
    assert ops["main"]["called_pairs"] == 6
    assert ops["critic"]["called_pairs"] == 2
    assert all(p["repair_feedback"] == "" for s, p in seen if s == "critic")
    rows, _ = subject.DirectionalArm("main").replay_cell(root, "main", "coverage", packets, {})
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
        for r in rows
    ]
    assert set(subject.compare(root, packets, labels)["transitions"]) == {"CORRECT -> CORRECT"}
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_ROOT_EXISTS"):
        subject.run_components(root, packets, None, None, {})
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_MAIN_PROVENANCE"):
        subject.critic_inputs(packets, [r | {"diagnostic_version": "dedup-judge-hs-v0.6.2.30"} for r in rows])


@pytest.mark.parametrize(
    "failure", ["pruned_raw", "changed_main", "changed_payload", "wrong_prompt", "old_contract", "legacy_leak"]
)
def test_rejects_trace_and_payload_mismatches(failure):
    from data_designer.engine.models.parsers.errors import ParserException

    arm = subject.DirectionalArm("critic")
    packets, values, mains = fixture("critic")
    packet = packets[-1]
    row = deepcopy(output(arm, packet, values[packet["canonical_pair_id"]]))
    if failure in {"pruned_raw", "old_contract"}:
        value = deepcopy(row[COLUMN])
        if failure == "pruned_raw":
            value["a_meaning_in_b"]["harmless_unique_ids"] = None
        else:
            value["contract_version"] = "dedup-retained-coverage-v4"
        row[COLUMN + "__trace"][-1]["content"] = "```json\n" + json.dumps(value) + "\n```"
    elif failure == "changed_main":
        row["proposed_main_text"] += " changed"
    elif failure == "changed_payload":
        row["payload"]["document_a"]["text"] += " changed"
    elif failure == "wrong_prompt":
        row[COLUMN + "__trace"][0]["content"] += " changed"
    else:
        row["qwen_dedup_semantic_judge"] = {}
    with pytest.raises((DedupEvaluationError, ParserException)):
        arm.bind([packet], [row], mains, "coverage")


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_previous_same_schema_trace_cannot_be_relabelled_as_new_candidate(stage):
    from eval.dedup.analysis.context_experiment import ContextArm

    packets, values, mains = fixture(stage)
    packet = packets[0]
    row = output(ContextArm(stage), packet, values[packet["canonical_pair_id"]])
    with pytest.raises(DedupEvaluationError, match="FOLLOWUP_BOUNDARY_PROMPT"):
        subject.DirectionalArm(stage).bind([packet], [row], mains, "coverage")


def test_only_failed_pair_retries_and_critic_feedback_resets(tmp_path):
    arm = subject.DirectionalArm("main")
    packets, values, _ = fixture("main")
    packets = packets[-2:]
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            rows = []
            for packet in batch:
                value = deepcopy(values[packet["canonical_pair_id"]])
                if len(batches) == 1 and packet == packets[0]:
                    value["a_meaning_in_b"]["source_span_id"] = "S999"
                rows.append(output(arm, packet, value))
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    ops = arm.run_cell(tmp_path / "cell", packets, {}, Runtime(), _Relay(), _settings(), "coverage")
    assert ops["valid"] == 2
    assert ops["called_pair_retry_rate"] == 0.5
    assert len(batches[1]) == 1
    assert batches[1][0]["repair_feedback"].startswith("Validation issue: ")
    rows, _ = arm.replay_cell(tmp_path, "cell", "coverage", packets, {})
    prepared, _ = subject.critic_inputs(packets, rows)
    assert all(p["repair_feedback"] == "" for p in prepared)


def test_frozen_manifest_and_budget_guard(tmp_path):
    from eval.dedup.rejudge_comparison import _source_digest

    artifact = tmp_path / "artifact.json"
    write_json_atomic(artifact, {"immutable": True})
    manifest = {
        "version": VERSION,
        "response_contract": CONTRACT,
        "source_implementation_sha256": _source_digest(),
        "frozen_files": {str(artifact): sha256_file(artifact)},
        "input_files": {},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(tmp_path / "manifest.json", manifest)
    assert subject.validate(tmp_path) == manifest
    artifact.write_text("{}")
    with pytest.raises(DedupEvaluationError, match="DIRECTIONAL_FREEZE"):
        subject.validate(tmp_path)

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return range(32768)

    with pytest.raises(DedupEvaluationError, match="COMPOSITE_TOKEN_BUDGET"):
        subject.DirectionalArm("main", tokenizer=Tokenizer()).layout(fixture("main")[0], {}, "coverage")
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_MAIN_REUSE"):
        subject.DirectionalArm("main").layout(fixture("main")[0], {"old": {}}, "coverage")


@pytest.mark.parametrize("fail_preflight", [False, True])
def test_eleven_case_preflight_gate_and_immutable_schedule(tmp_path, monkeypatch, fail_preflight):
    import transformers

    from eval.dedup.judging import request_relay

    packets, _, _ = fixture("main")
    packets += [packets[0] | {"canonical_pair_id": f"extra_{i}"} for i in range(4)]
    for repeat in (1, 2):
        _write_jsonl(tmp_path / f"input_repeat_{repeat}.jsonl", packets)
    write_json_atomic(tmp_path / "labels_private.json", {"human_development": packets, "synthetic_policy": []})
    manifest = {
        "settings": {
            "api_key_env": "CONTEXT_TEST_KEY",
            "logical_model": "test",
            "hub_model": "test",
            "hub_base_url": "http://127.0.0.1:1/v1",
            "timeout_seconds": 1,
        },
        "schedule": ["preflight", "repeat_1", "repeat_2"],
        "contract_digest": "test-digest",
        "preflight_ids": [p["canonical_pair_id"] for p in packets],
    }
    monkeypatch.setenv("CONTEXT_TEST_KEY", "local-test-only")
    monkeypatch.setenv("RAY_ADDRESS", "test-only-no-connection")
    monkeypatch.setattr(subject, "validate", lambda _root: manifest)

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return range(1)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    monkeypatch.setattr(request_relay, "RequestRelay", lambda **_kwargs: nullcontext(_Relay()))
    observed = []

    def component(root, *_args, **_kwargs):
        observed.append(root.name)
        stage = {"called_pairs": 11, "http_statuses": {"200": 11}, "judge_retried_pairs": int(fail_preflight)}
        return {"complete": True, "main": stage, "critic": stage}

    monkeypatch.setattr(subject, "run_components", component)
    monkeypatch.setattr(subject, "compare", lambda *_args: {"transitions": {"CORRECT -> CORRECT": {"count": 11}}})
    if fail_preflight:
        with pytest.raises(DedupEvaluationError, match="COMPOSITE_PREFLIGHT_FAILED"):
            subject.run(tmp_path)
        assert observed == ["preflight"]
        assert (tmp_path / "stopped.json").exists()
        assert not (tmp_path / "assessment.json").exists()
    else:
        report = subject.run(tmp_path)
        assert observed == manifest["schedule"]
        assert list(report["cells"]) == manifest["schedule"]
        assert len(list(tmp_path.glob("schedule_after_*.json"))) == 3
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_STARTED"):
        subject.run(tmp_path)


@pytest.mark.parametrize("stage", ["main", "critic"])
@pytest.mark.parametrize("retry", [False, True])
def test_native_ray_http_full_binding_retry(stage, retry, httpserver, tmp_path):
    from werkzeug.wrappers import Response

    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray cluster required")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    arm = subject.DirectionalArm(stage)
    packets, values, mains = fixture(stage)
    called, _owned = arm.layout(packets, mains, "coverage")
    requests = []

    def respond(request):
        body = json.loads(request.get_data())
        text = "\n".join(trace_text(m["content"]) for m in body["messages"])
        pid = next(p["canonical_pair_id"] for p in called if p["canonical_pair_id"].capitalize() + " probe." in text)
        value = deepcopy(values[pid])
        repaired = "DIRECTIONAL_WITNESS_BINDING" in text
        if retry and pid == "context_ba" and not repaired:
            value["a_meaning_in_b"].update(source_span_id="S001", counterpart_span_id="A001")
            value["hard_conflict"].update(a_span_id="S001", b_span_id="S001")
        requests.append({"pair": pid, "repair": repaired, "body": body})
        return Response(
            json.dumps(
                {
                    "id": "directional-boundary",
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
    settings = _settings() | {"logical_model": "directional-boundary", "ray_temp_dir": "/raid/hfang/ihb/r32diag"}
    with arm.runtime("coverage", httpserver.url_for("/v1"), settings) as runtime:
        ops = arm.run_cell(tmp_path / "cell", packets, mains, runtime, _Relay(), settings, "coverage")
    bound, _ = arm.replay_cell(tmp_path, "cell", "coverage", packets, mains)
    assert len(bound) == ops["valid"] == 7
    assert ops["judge_retried_pairs"] == int(retry)
    assert len(requests) == len(httpserver.log) == len(called) + int(retry)
    assert sum(r["repair"] for r in requests) == int(retry)
    assert all(r["pair"] == "context_ba" for r in requests if r["repair"])
    expected_messages = []
    for path in sorted((tmp_path / "cell").glob("attempt_*/input.jsonl")):
        expected_messages.extend(arm.renderers["coverage"](p) for p in _jsonl(path))
    actual = []
    for r in requests:
        body = r["body"]
        assert body["temperature"] == 0
        assert body["top_p"] == 1
        assert body["max_tokens"] == 4096
        actual.append([{"role": m["role"], "content": trace_text(m["content"])} for m in body["messages"]])
    assert sorted(map(json.dumps, actual)) == sorted(map(json.dumps, expected_messages))
