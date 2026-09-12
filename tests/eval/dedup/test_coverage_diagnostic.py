# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis.coverage_diagnostic import bind_coverage, replay_cell, run_coverage_cell, technical_passed
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl, blind_rows
from eval.dedup.judging.coverage_witness import COLUMN
from eval.dedup.validation import DedupEvaluationError, sha256_json
from tests.eval.dedup.test_coverage_witness import _case


def _packets():
    main, value, payload = _case("Identical cookie notice.", "Identical cookie notice.", non_main=True)
    inputs = blind_rows([{"canonical_pair_id": "p", "payload": payload}], repeat=1)
    return inputs, value, {"p": main}


def _output(packet, value):
    return {
        **packet,
        COLUMN: value,
        COLUMN + "__trace": [
            {"role": "assistant", "content": [{"type": "text", "text": "```json\n" + json.dumps(value) + "\n```"}]}
        ],
    }


def test_bind_requires_actual_new_response_and_trace_without_mutating_inputs():
    inputs, value, mains = _packets()
    outputs = [_output(inputs[0], value)]
    snapshot = deepcopy((inputs, outputs, mains))
    row = bind_coverage(inputs, outputs, mains)[0]
    assert row["same_duplicate_group"] == "YES"
    assert row["relation_type"] == "EXACT"
    assert row["coverage_response_sha256"] == sha256_json(value)
    assert row["coverage_trace_sha256"] == sha256_json(outputs[0][COLUMN + "__trace"])
    assert row["native_corrections"] == 0
    assert (inputs, outputs, mains) == snapshot


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "unknown", "hash", "payload", "main", "old_critic", "trace"]
)
def test_no_missing_duplicate_foreign_or_leaked_response_can_be_scored(failure):
    inputs, value, mains = _packets()
    outputs = [_output(inputs[0], value)]
    if failure == "missing":
        outputs.clear()
    elif failure == "duplicate":
        outputs *= 2
    elif failure == "unknown":
        outputs[0]["canonical_pair_id"] = "other"
    elif failure == "hash":
        outputs[0]["judge_payload_hash"] = "other"
    elif failure == "payload":
        outputs[0]["payload"] = {}
    elif failure == "main":
        outputs[0]["qwen_dedup_semantic_judge"] = mains["p"]
    elif failure == "old_critic":
        outputs[0]["qwen_dedup_record_binding_critic"] = {}
    else:
        outputs[0].pop(COLUMN + "__trace")
    with pytest.raises(DedupEvaluationError):
        bind_coverage(inputs, outputs, mains)


def test_native_corrections_are_retained_and_fail_technical_preflight():
    inputs, value, mains = _packets()
    output = _output(inputs[0], value)
    output[COLUMN + "__trace"].insert(0, {"role": "assistant", "content": "previous invalid result"})
    assert bind_coverage(inputs, [output], mains)[0]["native_corrections"] == 1
    operations = {
        "requested": 8,
        "valid": 8,
        "errors": 0,
        "retried": 0,
        "native_corrections": 0,
        "http_statuses": {"200": 8},
    }
    assert technical_passed(operations)
    assert not technical_passed({**operations, "native_corrections": 1})
    assert not technical_passed({**operations, "http_statuses": {"200": 9}})
    assert not technical_passed({**operations, "retried": 1})


class _Relay:
    def set_context(self, context):
        self.context = context


def _settings():
    return {
        "max_retries": 2,
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 4096,
        "timeout_seconds": 30,
        "max_parallel_requests": 2,
    }


def test_only_invalid_pair_retries_and_saved_raw_digest_replays(tmp_path: Path):
    inputs, value, mains = _packets()
    inputs.append({**inputs[0], "canonical_pair_id": "q"})
    mains["q"] = mains["p"]
    batches = []

    class Runtime:
        def run(self, **kwargs):
            packets = _jsonl(Path(kwargs["input_path"]))
            batches.append([p["canonical_pair_id"] for p in packets])
            rows = []
            for p in packets:
                output = _output(p, value)
                if p["canonical_pair_id"] == "q" and len(batches) == 1:
                    bad = {**value, "anchor_a_ids": "A999"}
                    output = _output(p, bad)
                rows.append(output)
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    name = "repeat_1_coverage"
    complete = run_coverage_cell(tmp_path / name, inputs, mains, Runtime(), _Relay(), _settings())
    assert batches == [["p", "q"], ["q"]]
    assert complete["valid"] == 2
    assert complete["errors"] == 0
    assert complete["retried"] == 1
    rows, _ = replay_cell(tmp_path, name, inputs, mains)
    assert len(rows) == 2
    assert {r["canonical_pair_id"]: r["attempts"] for r in rows} == {"p": 1, "q": 2}
    with pytest.raises(DedupEvaluationError, match="observed cell"):
        run_coverage_cell(tmp_path / name, inputs, mains, Runtime(), _Relay(), _settings())
    changed = [{**r, "a_can_replace_b": "NO"} for r in rows]
    with pytest.raises(DedupEvaluationError, match="IMMUTABLE_ARTIFACT_COLLISION"):
        _write_jsonl(tmp_path / name / "predictions.jsonl", changed)
    (tmp_path / name / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in changed))
    with pytest.raises(DedupEvaluationError, match="artifacts changed"):
        replay_cell(tmp_path, name, inputs, mains)


def test_terminal_failures_remain_explicit_and_are_not_smaller_scored_denominators(tmp_path: Path):
    inputs, _, mains = _packets()

    class Runtime:
        def run(self, **kwargs):
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", [])

    name = "preflight_coverage"
    complete = run_coverage_cell(tmp_path / name, inputs, mains, Runtime(), _Relay(), _settings())
    assert complete["requested"] == 1
    assert complete["valid"] == 0
    assert complete["errors"] == 1
    assert json.loads((tmp_path / name / "terminal_pair_ids.json").read_text()) == ["p"]
    assert replay_cell(tmp_path, name, inputs, mains)[0] == []
    assert not technical_passed(complete)
