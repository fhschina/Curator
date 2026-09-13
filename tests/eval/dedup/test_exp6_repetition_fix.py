# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import checkpoint_preflight as fixtures
from eval.dedup.analysis import exp5_anchor_id_fix as baseline
from eval.dedup.analysis import exp6_repetition_fix as subject
from tests.eval.dedup import test_exp1_reproduction_runtime as native
from tests.eval.dedup.test_critic_retention_v3 import fixture as coverage_fixture
from tests.eval.dedup.test_critic_subject_binding import fixture as subject_fixture
from tests.eval.dedup.test_exp1_conflict_evidence_fix import conflict_fixture, run_case
from tests.eval.dedup.test_main_repetition_recovery import repeated_response


@pytest.fixture(autouse=True)
def response_envelopes(monkeypatch):
    original = native.response
    monkeypatch.setattr(
        native, "response", lambda value: value if isinstance(value, dict) and "choices" in value else original(value)
    )


@pytest.mark.parametrize(
    "kind", ["exact", "negative", "coverage", "anchor_fix", "conflict_fix", "subject_verifier", "main_correction"]
)
def test_nontrigger_paths_match_all_exp6_requests_and_results(tmp_path, kind):
    payload, main, bad, good = conflict_fixture()
    if kind in {"exact", "main_correction"}:
        payload = fixtures.synthetic_payload("Same retained text.", "Same retained text.")
        main = fixtures.synthetic_raw(payload, deltas=("none", "none"))
        answers = [main] if kind == "exact" else [json.dumps(main), main]
    elif kind == "negative":
        main["span_hard_conflict"] = {"score": "identity_or_slot", "reasoning": "Different actual forums."}
        answers = [main]
    elif kind == "conflict_fix":
        answers = [main, json.dumps(bad), json.dumps(good)]
    elif kind == "subject_verifier":
        payload, _, proposal = subject_fixture()
        main = fixtures.synthetic_raw(
            payload,
            profiles=("non_main_only", "non_main_only"),
            deltas=("universal_ui_or_repetition", "universal_ui_or_repetition"),
            basis="verified_equivalent_non_main_message",
        )
        _, _, proof = coverage_fixture(payload["document_a"]["text"], payload["document_b"]["text"])
        verification = {
            "a_subject_kind": "NAMED_ACTUAL_TARGET",
            "b_subject_kind": "NAMED_ACTUAL_TARGET",
            "comparison": "SUPPORTED_DIFFERENT_NAMED_TARGETS",
            "explanation": "Different actual forums.",
        }
        answers = [main, json.dumps(proof), json.dumps(proposal), json.dumps(verification)]
    else:
        if kind == "anchor_fix":
            good["shared_anchor_ids"] = ["S1"]
        answers = [main, json.dumps(good)]
    before, before_requests = run_case(baseline, tmp_path / "before", deepcopy(payload), answers)
    after, after_requests = run_case(subject, tmp_path / "after", deepcopy(payload), answers)
    assert before_requests == after_requests
    assert {k: v for k, v in after.items() if k != "version"} == {k: v for k, v in before.items() if k != "version"}
    assert "main_repetition_repair" not in after


@pytest.mark.parametrize("after_correction", [False, True])
def test_one_repair_bypasses_native_feedback_and_restarts_from_initial_request(tmp_path, after_correction):
    payload = fixtures.synthetic_payload("Same.", "Same.")
    main = fixtures.synthetic_raw(payload, deltas=("none", "none"))
    loop = repeated_response("REPEATED_FAILED_OUTPUT_MARKER, ")
    answers = (["invalid output before repetition"] if after_correction else []) + [loop, main]
    result, requests = run_case(subject, tmp_path, payload, answers)
    assert result["status"] == "VALID"
    assert len(requests) == 2 + int(after_correction)
    assert requests[-1] == subject.repetition.repair_request(requests[0])
    assert "REPEATED_FAILED_OUTPUT_MARKER" not in json.dumps(requests)
    assert "invalid output before repetition" not in json.dumps(requests[-1])
    assert result["main_repetition_repair"]["attempts"] == 1
    assert result["main_repetition_repair"]["trigger_stage"] == ("main-01-02" if after_correction else "main-01-01")
    assert result["stages"][-1]["stage"] == "main-repetition-repair"
    assert result["main_attempts"][-1]["status"] == "VALID"
    saved = json.loads((tmp_path / "responses/test-pair-main-01-01.json").read_text())
    if not after_correction:
        assert saved["raw_response"] == loop


@pytest.mark.parametrize("failure", ["repeat_again", "invalid_json", "invalid_schema", "semantic_validation"])
def test_failed_repair_never_resumes_native_or_outer_main_retries(tmp_path, failure):
    payload = fixtures.synthetic_payload("Same.", "Same.")
    main = fixtures.synthetic_raw(payload, deltas=("none", "none"))
    if failure == "repeat_again":
        repair = repeated_response()
    elif failure == "invalid_json":
        repair = "invalid JSON"
    elif failure == "invalid_schema":
        repair = {"wrong": True}
    else:
        main["span_content_profile_a"]["score"] = "invalid_profile"
        repair = main
    result, requests = run_case(subject, tmp_path, payload, [repeated_response(), repair])
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["public"]["same_duplicate_group"] == "UNRESOLVED"
    assert len(requests) == 2
    assert [s["stage"] for s in result["stages"]] == ["main-01-01", "main-repetition-repair"]
    assert result["main_attempts"][-1]["status"] == "FAILURE"


def test_successful_repair_still_runs_existing_critics_and_can_return_negative(tmp_path):
    payload, main, _, proof = conflict_fixture()
    result, requests = run_case(subject, tmp_path, payload, [repeated_response(), main, json.dumps(proof)])
    assert result["status"] == "VALID"
    assert result["components"]["main"]["same_duplicate_group"] == "YES"
    assert result["public"]["same_duplicate_group"] == "NO"
    assert [s["stage"] for s in result["stages"]] == ["main-01-01", "main-repetition-repair", "coverage"]
    assert len(requests) == 3
    subject.old.validate_evidence_offsets(result["public"], payload)
