# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

from eval.dedup.analysis import retention_v4_proof_replay as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_retention_v4 import payload, review


def test_missing_main_response_is_not_filled_with_a_reference_or_another_repeat():
    row = {"canonical_pair_id": "one", "review_id": "ONE", "payload": payload("Policy.\nNext", "Policy.")}
    old = {
        "canonical_pair_id": "one",
        "review_id": "ONE",
        "repeat": 1,
        "arm": "candidate",
        "status": "ENGINEERING_FAILURE",
        "stages": [],
        "public": subject.pilot.unresolved_judge_output_v4(),
    }

    def missing(_):
        raise DedupEvaluationError("MISSING_RESPONSE", "No response was submitted")

    out = subject.replay_case(row, 1, old, missing)
    assert out["status"] == "ENGINEERING_FAILURE"
    assert out["public"]["same_duplicate_group"] == "UNRESOLVED"


def test_fixed_negative_conflict_replay_needs_no_invented_dependent_critic_call():
    p = payload("Actual account owner is Ann.", "Actual account owner is Anna.")
    raw = review(p, "MAIN_ADDITION", "MAIN_ADDITION", conflict="IDENTITY_CONFLICT")
    raw["a_context_span_id"] = raw["b_context_span_id"] = "S001"
    row = {"canonical_pair_id": "one", "review_id": "ONE", "payload": p}
    old = {
        "canonical_pair_id": "one",
        "review_id": "ONE",
        "repeat": 1,
        "arm": "candidate",
        "status": "ENGINEERING_FAILURE",
        "stages": [{"stage": "main"}],
        "public": subject.pilot.unresolved_judge_output_v4(),
    }
    before = deepcopy((row, raw, old))
    calls = []

    def main_only(stage):
        calls.append(stage)
        assert stage == "main"
        return raw

    out = subject.replay_case(row, 1, old, main_only)
    assert out["status"] == "VALID"
    assert out["public"]["same_duplicate_group"] == "NO"
    assert calls == ["main"]
    assert (row, raw, old) == before


def test_new_positive_needing_unsaved_coverage_is_unavailable_not_a_spliced_success():
    p = payload("Complete policy.\nNext", "Complete policy.")
    raw = review(p, "NON_MAIN_ADDITION")
    row = {"canonical_pair_id": "one", "review_id": "ONE", "payload": p}
    old = {
        "canonical_pair_id": "one",
        "review_id": "ONE",
        "repeat": 1,
        "arm": "candidate",
        "status": "ENGINEERING_FAILURE",
        "stages": [],
        "public": subject.pilot.unresolved_judge_output_v4(),
    }

    def incomplete(stage):
        if stage == "main":
            return raw
        raise DedupEvaluationError("DEPENDENT_STAGE_MISSING", "Coverage was not called")

    out = subject.replay_case(row, 1, old, incomplete)
    assert out["status"] == "ENGINEERING_FAILURE"
    assert out["public"]["same_duplicate_group"] == "UNRESOLVED"
