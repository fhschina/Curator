# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis.composite_experiment import CompositeArm, critic_inputs, run_components, validate_diagnostic
from eval.dedup.analysis.coverage_followup import trace_text
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl
from eval.dedup.judging.composite_runtime import VERSION, encode_main_review
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.local_ndd import _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic
from tests.eval.dedup.test_composite_coverage import certificate
from tests.eval.dedup.test_coverage_followup import _Relay, _settings


def fixture(stage):
    packets, values, mains = [], {}, {}
    for pid in ("covered", "uncovered", "uncertain", "negative", "exact"):
        if pid == "covered":
            value, payload, _, _ = certificate(
                "Covered probe. Use cookies.", "Covered probe. 使用 Cookie。", loss_sides=()
            )
            value["translation_status"] = "COMPLETE_FAITHFUL"
        elif pid == "exact":
            value, payload, _, _ = certificate("Exact probe.", "Exact probe.", loss_sides=())
        else:
            value, payload, _, _ = certificate(
                f"{pid.capitalize()} probe. Library closes Monday.",
                f"{pid.capitalize()} probe. Library closes Monday. Astronomy papers.",
            )
        mains[pid] = deepcopy(value)
        if pid == "uncertain":
            value["b_meaning_in_a"] = {
                "status": "UNRESOLVED",
                "reviewed_unique_ids": value["b_meaning_in_a"]["reviewed_unique_ids"],
                "coverage_explanation": "Cannot establish the retained meaning.",
                "checked_opposite_ids": "S001",
            }
        if pid == "negative":
            value["shared_basis"] = "NO_SHARED_SUBSTANTIVE_CONTENT"
            mains[pid] = deepcopy(value)
        packet = {
            "canonical_pair_id": pid,
            "judge_payload_hash": sha256_json(payload),
            "payload": payload,
            "repair_feedback": "",
        }
        if stage == "critic" and pid not in {"negative", "exact"}:
            packet["proposed_main_text"] = encode_main_review(mains[pid])
        packets.append(packet)
        values[pid] = value
    return packets, values, mains if stage == "critic" else {}


def output(arm, packet, value):
    return {
        **packet,
        COLUMN: value,
        COLUMN + "__trace": [
            *arm.renderers["coverage"](packet),
            {"role": "assistant", "content": "```json\n" + json.dumps(value) + "\n```"},
        ],
    }


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_actual_component_routing_complete_capture_and_raw_replay(stage, tmp_path):
    arm = CompositeArm(stage)
    inputs, values, mains = fixture(stage)

    class Runtime:
        def run(self, **kwargs):
            rows = [output(arm, p, values[p["canonical_pair_id"]]) for p in _jsonl(Path(kwargs["input_path"]))]
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    ops = arm.run_cell(tmp_path / "cell", inputs, mains, Runtime(), _Relay(), _settings(), "coverage")
    assert ops["valid"] == ops["requested"] == 5
    assert ops["called_pairs"] == (4 if stage == "main" else 3)
    assert ops["called_pair_retry_rate"] == 0
    rows, _ = arm.replay_cell(tmp_path, "cell", "coverage", inputs, mains)
    by_id = {r["canonical_pair_id"]: r for r in rows}
    assert by_id["uncovered"]["b_can_replace_a"] == "YES"
    assert by_id["uncertain"]["confidence_tier"] == "LOW"
    assert by_id["exact"]["attempts"] == 0
    assert "coverage_response" not in by_id["exact"]
    if stage == "critic":
        assert "coverage_response" not in by_id["negative"]


def test_fresh_main_then_critic_share_exact_certificate_but_not_retry_feedback(tmp_path):
    inputs, values, _ = fixture("main")
    observed = []

    class Runtime:
        def __init__(self, arm):
            self.arm = arm

        def run(self, **kwargs):
            rows = _jsonl(Path(kwargs["input_path"]))
            observed.extend((self.arm.stage, p) for p in rows)
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl",
                [output(self.arm, p, values[p["canonical_pair_id"]]) for p in rows],
            )

    ops = run_components(tmp_path / "fresh", inputs, lambda arm: nullcontext(Runtime(arm)), _Relay(), _settings())
    assert ops["complete"]
    assert {p["canonical_pair_id"] for stage, p in observed if stage == "main"} == {
        "covered",
        "uncovered",
        "uncertain",
        "negative",
    }
    assert {p["canonical_pair_id"] for stage, p in observed if stage == "critic"} == {"covered", "uncovered"}
    for stage, packet in observed:
        if stage == "critic":
            assert packet["repair_feedback"] == ""
            assert packet["proposed_main_text"] == encode_main_review(values[packet["canonical_pair_id"]])
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_ROOT_EXISTS"):
        run_components(tmp_path / "fresh", inputs, None, None, {})


@pytest.mark.parametrize("failure", ["pruned_raw", "changed_main", "changed_payload", "wrong_prompt", "legacy_leak"])
def test_untrusted_boundary_changes_are_rejected(failure):
    arm = CompositeArm("critic")
    inputs, values, mains = fixture("critic")
    packet = inputs[0]
    row = deepcopy(output(arm, packet, values[packet["canonical_pair_id"]]))
    if failure == "pruned_raw":
        bad = deepcopy(values[packet["canonical_pair_id"]])
        bad["a_meaning_in_b"]["source_span_id"] = None
        row[COLUMN + "__trace"][-1]["content"] = "```json\n" + json.dumps(bad) + "\n```"
    elif failure == "changed_main":
        row["proposed_main_text"] += " changed"
    elif failure == "changed_payload":
        row["payload"]["document_a"]["text"] += " changed"
    elif failure == "wrong_prompt":
        row[COLUMN + "__trace"][0]["content"] += " changed"
    else:
        row["qwen_dedup_semantic_judge"] = {}
    from data_designer.engine.models.parsers.errors import ParserException

    with pytest.raises((DedupEvaluationError, ParserException)):
        arm.bind([packet], [row], mains, "coverage")


def test_missing_main_cannot_be_replaced_with_a_previous_run_or_partial_results():
    packets, _, _ = fixture("main")
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_MAIN_COMPLETION"):
        critic_inputs(packets, [])
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_MAIN_REUSE"):
        CompositeArm("main").layout(packets, {"legacy": {}}, "coverage")


def test_only_invalid_main_pair_is_retried_and_replays_with_bound_feedback(tmp_path):
    arm = CompositeArm("main")
    packets, values, _ = fixture("main")
    packets = packets[:2]
    batches = []

    class Runtime:
        def run(self, **kwargs):
            batch = _jsonl(Path(kwargs["input_path"]))
            batches.append(batch)
            rows = []
            for packet in batch:
                value = deepcopy(values[packet["canonical_pair_id"]])
                if len(batches) == 1 and packet["canonical_pair_id"] == "uncovered":
                    value["b_meaning_in_a"]["source_span_id"] = "B999"
                rows.append(output(arm, packet, value))
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    ops = arm.run_cell(tmp_path / "cell", packets, {}, Runtime(), _Relay(), _settings(), "coverage")
    assert ops["valid"] == 2
    assert ops["called_pair_retry_rate"] == 0.5
    assert [p["canonical_pair_id"] for p in batches[1]] == ["uncovered"]
    assert batches[1][0]["repair_feedback"].startswith("Validation issue: ")
    rows, _ = arm.replay_cell(tmp_path, "cell", "coverage", packets, {})
    prepared, _ = critic_inputs(packets, rows)
    assert all(p["repair_feedback"] == "" for p in prepared)


def test_actual_prompt_budget_blocks_submission_instead_of_truncating():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["role"] == "system"
            return range(32768)

    packets, _, _ = fixture("main")
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_TOKEN_BUDGET"):
        CompositeArm("main", tokenizer=Tokenizer()).layout(packets, {}, "coverage")


def test_candidate_digest_covers_labels_inputs_and_frozen_source(tmp_path):
    from eval.dedup.rejudge_comparison import _source_digest

    source = tmp_path / "source.txt"
    source.write_text("frozen")
    manifest = {
        "version": VERSION,
        "source_implementation_sha256": _source_digest(),
        "frozen_files": {str(source): sha256_file(source)},
        "input_files": {},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(tmp_path / "manifest.json", manifest)
    assert validate_diagnostic(tmp_path) == manifest
    source.write_text("changed")
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_FREEZE"):
        validate_diagnostic(tmp_path)


@pytest.mark.parametrize("stage", ["main", "critic"])
@pytest.mark.parametrize("retry", [False, True])
def test_native_ray_http_writer_binder_first_and_retry(stage, retry, httpserver, tmp_path):
    from werkzeug.wrappers import Response

    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray cluster required")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    arm = CompositeArm(stage)
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
                    "id": "composite-boundary",
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
    settings = _settings() | {"logical_model": "composite-boundary", "ray_temp_dir": "/raid/hfang/ihb/r29diag"}
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
    actual = []
    for request, _ in httpserver.log:
        body = json.loads(request.get_data())
        assert body["temperature"] == 0
        assert body["top_p"] == 1
        assert body["max_tokens"] == 4096
        actual.append([{"role": m["role"], "content": trace_text(m["content"])} for m in body["messages"]])
    assert sorted(map(json.dumps, actual)) == sorted(json.dumps(arm.renderers["coverage"](p)) for p in called)
    assert all(r["response_transport"]["representation_changes"] == {} for r in bound)
