# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from pathlib import Path

import pytest

from eval.dedup.analysis import paced_result_audit as subject
from eval.dedup.analysis.critic_diagnostic import _jsonl, _write_jsonl
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_composite_experiment import output
from tests.eval.dedup.test_context_experiment import fixture
from tests.eval.dedup.test_coverage_followup import _Relay, _settings


def test_missing_terminal_and_successes_are_replayed_without_denominator_loss(tmp_path):
    packets, values, _ = fixture("main")
    failing = packets[-1]["canonical_pair_id"]

    class Runtime:
        def __init__(self, arm):
            self.arm = arm

        def run(self, **kwargs):
            rows = _jsonl(Path(kwargs["input_path"]))
            _write_jsonl(
                Path(kwargs["output_path"]) / "part.jsonl",
                [
                    output(self.arm, p, values[p["canonical_pair_id"]])
                    for p in rows
                    if p["canonical_pair_id"] != failing
                ],
            )

    subject.experiment.run_components(
        tmp_path / "cell", packets, lambda a: nullcontext(Runtime(a)), _Relay(), _settings()
    )
    replay = subject.replay(tmp_path, "cell", packets)
    assert len(replay["scoring"]["final"]) == len(packets)
    assert replay["final_valid"] == len(packets) - 1
    assert replay["terminal_ids"] == [failing]
    assert replay["operations"]["corrected_pair_ids"] == [failing]
    assert replay["operations"]["all_successes_have_saved_assistant_count"] is False
    (tmp_path / "cell/final_scoring_only.jsonl").write_text("")
    with pytest.raises(DedupEvaluationError, match="PACED_SCORING_CHANGED"):
        subject.replay(tmp_path, "cell", packets)


def test_transport_retries_do_not_increment_model_corrections(tmp_path):
    for stage in ("main", "critic"):
        folder = tmp_path / stage / "attempt_01"
        packet = {"canonical_pair_id": "same-pair"}
        _write_jsonl(folder / "input.jsonl", [packet])
        _write_jsonl(folder / "output/part.jsonl", [packet | {subject.COLUMN + "__trace": [{"role": "assistant"}]}])
        _write_jsonl(
            folder / "events.jsonl",
            [
                {
                    "external_request": True,
                    "sequence": i + 1,
                    "logical_request_sequence": 1,
                    "upstream_attempt": i + 1,
                    "upstream_http_status": status,
                    "admission_wait_seconds": 0,
                }
                for i, status in enumerate((429, 200))
            ],
        )
    result = subject.operation_audit(tmp_path)
    assert result["unique_called_pairs"] == 1
    assert result["unique_model_corrected_pairs"] == 0
    assert result["all_successes_have_saved_assistant_count"]


def test_weighted_errors_preserve_precision_and_recall_causes_separately():
    result = subject.weighted_error_distribution(
        {
            "errors": [
                {"error": "OVER_GROUP", "reason": "identity_slot", "weight": 20},
                {"error": "UNDER_GROUP_RESOLVED", "reason": "translation", "weight": 3},
                {"error": "OVER_GROUP", "reason": "identity_slot", "weight": 10},
            ]
        }
    )
    assert result["by_error"] == {"OVER_GROUP": 30, "UNDER_GROUP_RESOLVED": 3}
    assert result["by_reference_reason"]["identity_slot"] == {"OVER_GROUP": 30}


def test_incomplete_diagnostic_cannot_publish_full_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(subject.experiment, "validate", lambda _root: {})
    with pytest.raises(DedupEvaluationError, match="PACED_AUDIT_INCOMPLETE"):
        subject.audit(tmp_path)


def test_error_axes_keep_model_overlap_claim_separate_from_reference():
    label = {
        "canonical_pair_id": "p",
        "sample_weight": 10,
        "human_reason_code": "boilerplate_only",
        "human_same_duplicate_group": "NO",
        "human_a_can_replace_b": "NO",
        "human_b_can_replace_a": "NO",
    }
    prediction = {
        "canonical_pair_id": "p",
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "NO",
        "relation_type": "CONTAINMENT",
        "dominant_overlap_source": "COOKIE_CONSENT",
        "coverage_response": {"shared_basis": "SHARED_SUBSTANTIVE_CONTENT"},
    }
    result = subject.error_axes([label], [prediction])
    expected = {"OVER_GROUP": {"count": 1, "weight": 10}}
    assert result["reference_reason"]["boilerplate_only"] == expected
    assert result["predicted_basis"]["SHARED_SUBSTANTIVE_CONTENT"] == expected
    assert result["predicted_overlap"]["COOKIE_CONSENT"] == expected
    with pytest.raises(DedupEvaluationError, match="PACED_ERROR_MEMBERSHIP"):
        subject.error_axes([label], [])
