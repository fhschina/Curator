# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from eval.dedup.analysis import policy_experiment as subject
from eval.dedup.analysis.coverage_followup import trace_text
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.local_ndd import _read_output_rows
from eval.dedup.judging.payload_transport import encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_followup import _Relay, _settings
from tests.eval.dedup.test_typed_experiment import fixture, output


@pytest.fixture
def experiment():
    return subject.PolicyExperiment(subject.common.ANALYSIS / "v06227_experiment.json")


def test_contrast_proves_single_block_difference_and_prior_candidate_identity(experiment):
    packets, _, mains = fixture("coverage")
    audit = experiment.contrast(packets, mains)
    assert len(audit["called_inputs"]) == 3
    assert all(r["request_sha256"]["control"] != r["request_sha256"]["coverage"] for r in audit["called_inputs"])
    assert audit["policies"]["control"]["block"].startswith("NONEMPTY RECORD ANCHORS")
    assert audit["policies"]["coverage"]["block"].startswith("ESTABLISH THE ACTUAL RETAINED SUBJECT")


def test_contrast_rejects_an_extra_pair_prompt_change(experiment):
    packets, _, mains = fixture("coverage")
    render = experiment.renderers["control"]

    def changed(packet):
        messages = deepcopy(render(packet))
        messages[1]["content"] += " Extra policy."
        return messages

    experiment.renderers["control"] = changed
    with pytest.raises(DedupEvaluationError, match="POLICY_CONTRAST_CHANGED"):
        experiment.contrast(packets, mains)


@pytest.mark.parametrize("variant", ["control", "coverage"])
def test_both_arms_use_typed_replay_and_count_the_actual_native_column(experiment, tmp_path, variant):
    packets, values, mains = fixture("coverage")

    class Runtime:
        def run(self, **kwargs):
            rows = []
            for p in _jsonl(Path(kwargs["input_path"])):
                row = output(experiment, p, values[p["canonical_pair_id"]], variant, padding=True)
                if p["canonical_pair_id"] == "covered":
                    row[COLUMN + "__trace"][2:2] = [
                        {"role": "assistant", "content": "invalid json"},
                        {"role": "user", "content": "Return a valid object."},
                    ]
                rows.append(row)
            _write_jsonl(Path(kwargs["output_path"]) / "part.jsonl", rows)

    ops = experiment.run_cell(tmp_path / "cell", packets, mains, Runtime(), _Relay(), _settings(), variant)
    assert ops["called_pairs"] == 3
    assert ops["not_requested_owned_pairs"] == 2
    assert ops["valid"] == ops["requested"] == 5
    assert ops["retried"] == 0
    assert ops["judge_retried_pairs"] == 1
    assert ops["native_corrected_pairs"] == ops["native_corrections"] == 1
    assert ops["called_pair_retry_rate"] == 1 / 3
    rows, _ = experiment.replay_cell(tmp_path, "cell", variant, packets, mains)
    by_id = {r["canonical_pair_id"]: r for r in rows}
    assert by_id["covered"]["same_duplicate_group"] == "YES"
    assert by_id["uncovered"]["a_can_replace_b"] == "YES"
    assert by_id["uncovered"]["b_can_replace_a"] == "NO"
    assert by_id["uncertain"]["confidence_tier"] == "LOW"
    assert by_id["negative"]["same_duplicate_group"] == "NO"
    assert by_id["exact"]["relation_type"] == "EXACT"
    assert all("raw_output_sha256" not in by_id[k] for k in ("negative", "exact"))
    assert by_id["covered"]["coverage_contract"] == "dedup-retained-coverage-v3"
    assert by_id["covered"]["response_transport"]["representation_changes"]["added_null_keys"] > 0


@pytest.mark.parametrize("variant", ["control", "coverage"])
def test_invalid_loss_only_retries_its_pair_with_exact_feedback(experiment, tmp_path, variant):
    packets, values, mains = fixture("coverage")
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
    assert [p["canonical_pair_id"] for p in batches[1]] == ["uncovered"]
    assert batches[1][0]["repair_feedback"].startswith("Validation issue: ")
    assert ops["called_pair_retry_rate"] == 1 / 3
    assert len(experiment.replay_cell(tmp_path, "cell", variant, packets, mains)[0]) == 5


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize("failure", ["raw_null", "wrong_branch", "unknown_echo", "payload", "prompt", "main_leak"])
def test_both_typed_arms_keep_strict_raw_and_blind_boundaries(experiment, variant, failure):
    from data_designer.engine.models.parsers.errors import ParserException

    packets, values, mains = fixture("coverage")
    p = next(p for p in packets if p["canonical_pair_id"] == "covered")
    row = deepcopy(output(experiment, p, values["covered"], variant))
    if failure in {"raw_null", "wrong_branch"}:
        bad = deepcopy(values["covered"])
        bad["a_meaning_in_b"]["source_span_id"] = None if failure == "raw_null" else "A001"
        row[COLUMN + "__trace"][-1]["content"] = "```json\n" + json.dumps(bad) + "\n```"
    elif failure == "unknown_echo":
        row[COLUMN]["a_meaning_in_b"]["invented"] = None
    elif failure == "payload":
        row["payload"]["document_a"]["text"] += " changed"
    elif failure == "prompt":
        row[COLUMN + "__trace"][1]["content"] += " changed"
    else:
        row["qwen_dedup_semantic_judge"] = mains["covered"]
    with pytest.raises((DedupEvaluationError, ParserException)):
        experiment.bind([p], [row], mains, variant)


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize(
    "feedback", [None, {"code": "SELECTION_REFERENCE_INVALID", "message": "Retry", "details": {}}]
)
def test_full_native_typed_policy_first_and_retry(experiment, httpserver, tmp_path, variant, feedback):
    from werkzeug.wrappers import Response

    if not os.environ.get("COVERAGE_TEST_RAY_ADDRESS"):
        pytest.skip("explicit owned Ray address required for full pipeline integration")
    assert os.environ.get("RAY_ADDRESS") == os.environ["COVERAGE_TEST_RAY_ADDRESS"]
    packets, values, mains = fixture("coverage")
    called, owned = experiment.layout(packets, mains, variant)
    for p in called:
        p["repair_feedback"] = encode_repair_feedback(feedback)

    def respond(request):
        body = json.loads(request.get_data())
        text = "\n".join(trace_text(m["content"]) for m in body["messages"])
        pid = next(pid for pid in ("covered", "uncovered", "uncertain") if pid.capitalize() + " probe." in text)
        return Response(
            json.dumps(
                {
                    "id": "policy-boundary",
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
    settings = {**_settings(), "logical_model": "policy-boundary", "ray_temp_dir": "/raid/hfang/ihb/r27diag"}
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
    assert all(r[COLUMN] == values[r["canonical_pair_id"]] for r in raw)
    assert all(r["response_transport"]["representation_changes"] == {} for r in rows)
