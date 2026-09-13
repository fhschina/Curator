# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import checkpoint_preflight as fixtures
from eval.dedup.analysis import exp5_anchor_id_fix as baseline
from eval.dedup.analysis import exp6_repetition_fix2 as subject
from tests.eval.dedup import test_exp1_reproduction_runtime as native
from tests.eval.dedup.test_exp1_conflict_evidence_fix import conflict_fixture, run_case
from tests.eval.dedup.test_main_repetition_recovery import repeated_response


@pytest.fixture(autouse=True)
def response_envelopes(monkeypatch):
    original = native.response
    monkeypatch.setattr(
        native, "response", lambda value: value if isinstance(value, dict) and "choices" in value else original(value)
    )


def test_nontrigger_requests_and_complete_results_remain_exp6_identical(tmp_path):
    payload, main, _, proof = conflict_fixture()
    before, before_calls = run_case(baseline, tmp_path / "before", deepcopy(payload), [main, json.dumps(proof)])
    after, after_calls = run_case(subject, tmp_path / "after", deepcopy(payload), [main, json.dumps(proof)])
    assert before_calls == after_calls
    assert {k: v for k, v in before.items() if k != "version"} == {k: v for k, v in after.items() if k != "version"}


@pytest.mark.parametrize("valid", [True, False])
def test_one_recovery_keeps_existing_citation_validation_and_never_invents_ids(tmp_path, valid):
    payload, main, _, proof = conflict_fixture()
    if not valid:
        main = fixtures.synthetic_raw(
            payload, deltas=("other_substantive_content", "other_substantive_content"), basis="none"
        )
        for field in main.values():
            field["reasoning"] = "Distinct records; evidence references omitted."
        main["span_a_delta"]["reasoning"] = "A-only spans differ."
    result, calls = run_case(subject, tmp_path, payload, [repeated_response(), main, json.dumps(proof)])
    assert result["main_repetition_repair"]["attempts"] == 1
    assert calls[1] == subject.repetition.repair_request(calls[0])
    assert result["status"] == "VALID"
    if valid:
        assert len(calls) == 3
        assert result["public"]["same_duplicate_group"] == "NO"
    else:
        assert len(calls) == 2
        assert result["public"]["same_duplicate_group"] == "UNRESOLVED"
        assert "SPAN_CITATION_ISSUE:A_DELTA_WITHOUT_SIDE_SPAN_CITATION" in result["public"]["reason_codes"]


def test_failed_recovery_still_stops_without_further_main_calls(tmp_path):
    payload = fixtures.synthetic_payload("Same.", "Same.")
    result, calls = run_case(subject, tmp_path, payload, [repeated_response(), repeated_response()])
    assert len(calls) == 2
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["error_code"] == "MAIN_REPETITION_REPAIR_INCOMPLETE"
