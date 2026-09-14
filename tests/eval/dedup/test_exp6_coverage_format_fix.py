# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import checkpoint_preflight as fixtures
from eval.dedup.analysis import exp6_coverage_format_fix as subject
from eval.dedup.analysis import exp6_repetition_fix2 as baseline
from tests.eval.dedup import test_exp1_reproduction_runtime as native
from tests.eval.dedup.test_coverage_format_recovery import response
from tests.eval.dedup.test_exp1_conflict_evidence_fix import conflict_fixture, run_case


@pytest.fixture(autouse=True)
def response_envelopes(monkeypatch):
    original = native.response
    monkeypatch.setattr(
        native, "response", lambda value: value if isinstance(value, dict) and "choices" in value else original(value)
    )


@pytest.mark.parametrize("kind", ["normal", "zero_padding", "conflict_repair", "exact", "negative"])
def test_nontrigger_request_and_complete_result_identity(tmp_path, kind):
    payload, main, bad, good = conflict_fixture()
    if kind == "zero_padding":
        good["shared_anchor_ids"] = ["S1"]
    answers = [main, json.dumps(good)] if kind != "conflict_repair" else [main, json.dumps(bad), json.dumps(good)]
    if kind == "exact":
        payload = fixtures.synthetic_payload("Same.", "Same.")
        answers = [fixtures.synthetic_raw(payload, deltas=("none", "none"))]
    elif kind == "negative":
        main["span_hard_conflict"] = {"score": "identity_or_slot", "reasoning": "A001 and B001 differ."}
        answers = [main]
    a, calls_a = run_case(baseline, tmp_path / "a", deepcopy(payload), answers)
    b, calls_b = run_case(subject, tmp_path / "b", deepcopy(payload), answers)
    assert calls_a == calls_b
    assert {k: v for k, v in a.items() if k != "version"} == {k: v for k, v in b.items() if k != "version"}


def test_prefix_repair_uses_no_model_call_and_keeps_main_and_original_receipt(tmp_path):
    payload, main, _, good = conflict_fixture()
    good["shared_anchor_ids"] = [sid[1:] for sid in good["shared_anchor_ids"]]
    result, calls = run_case(subject, tmp_path, payload, [main, json.dumps(good)])
    assert result["status"] == "VALID"
    assert len(calls) == 2
    assert result["coverage_anchor_id_repair"]["original_review"] == good
    assert result["coverage_anchor_id_repair"]["additional_model_calls"] == 0
    assert result["public"]["same_duplicate_group"] == "NO"


@pytest.mark.parametrize("repair", ["valid_negative", "repeat", "ordinary_length", "invalid_json", "invalid_anchor"])
def test_one_failure_only_repair_preserves_schema_evidence_and_terminates(tmp_path, repair):
    payload, main, _, good = conflict_fixture()
    answer = json.dumps(good)
    if repair == "repeat":
        answer = response()
    elif repair == "ordinary_length":
        answer = response("incomplete")
    elif repair == "invalid_json":
        answer = "broken"
    elif repair == "invalid_anchor":
        answer = json.dumps({**good, "shared_anchor_ids": ["S999"]})
    result, calls = run_case(subject, tmp_path, payload, [main, response(), answer])
    assert len(calls) == 3
    assert calls[-1] == subject.coverage_format_recovery.repair_request(calls[1])
    assert result["coverage_format_repair"]["attempts"] == 1
    assert "FAILED_MARKER" not in str(calls)
    assert result["status"] == ("VALID" if repair == "valid_negative" else "ENGINEERING_FAILURE")
    if repair == "valid_negative":
        assert result["public"]["same_duplicate_group"] == "NO"
        subject.old.validate_evidence_offsets(result["public"], payload)


def test_nonrepetitive_coverage_length_does_not_expand_retry_scope(tmp_path):
    payload, main, _, _ = conflict_fixture()
    result, calls = run_case(subject, tmp_path, payload, [main, response("ordinary incomplete")])
    assert len(calls) == 2
    assert result["error_code"] == "EXP1_STAGE_FINISH"
    assert "coverage_format_repair" not in result
