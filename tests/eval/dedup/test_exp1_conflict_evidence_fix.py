# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import checkpoint_preflight as fixtures
from eval.dedup.analysis import exp1_conflict_evidence_fix as subject
from eval.dedup.validation import sha256_json
from tests.eval.dedup.test_critic_retention_v3 import fixture as coverage_fixture
from tests.eval.dedup.test_critic_subject_binding import fixture as subject_fixture
from tests.eval.dedup.test_exp1_reproduction_runtime import server


def run_case(module, root, payload, answers):
    payload["payload_schema_version"] = "judge-visible-payload-v3"
    row = {"canonical_pair_id": "test-pair", "review_id": "R1", "payload": payload}
    body = subject.old.body(subject.old.main_messages(payload))
    frozen = {"body": body, "request_sha256": sha256_json(body)}
    with server(answers) as (endpoint, requests):
        result = module.execute_case(root, row, endpoint, frozen, subject.old.coverage_renderer())
    return result, requests


def conflict_fixture():
    payload, _, raw = coverage_fixture(
        "Using Forum Alpha accepts these rules.", "Using Forum Beta accepts these rules."
    )
    main = fixtures.synthetic_raw(payload, deltas=("semantically_covered", "semantically_covered"))
    raw.update(conflict="IDENTITY_CONFLICT", a_context_span_id="S001", b_context_span_id="S001")
    repaired = {**raw, "a_context_span_id": "A001", "b_context_span_id": "B001"}
    return payload, main, raw, repaired


@pytest.mark.parametrize("kind", ["exact", "negative", "valid_conflict", "invalid_id", "subject_verifier"])
def test_unaffected_paths_preserve_all_requests_and_results(tmp_path, kind):
    p, main, bad, good = conflict_fixture()
    if kind == "exact":
        p = fixtures.synthetic_payload("Same retained text.", "Same retained text.")
        main = fixtures.synthetic_raw(p, deltas=("none", "none"))
        answers = [main]
    elif kind == "negative":
        main["span_hard_conflict"] = {"score": "identity_or_slot", "reasoning": "A001 and B001 are different forums."}
        answers = [main]
    elif kind == "invalid_id":
        bad["a_context_span_id"] = "A999"
        answers = [main, json.dumps(bad)]
    elif kind == "subject_verifier":
        p, _, proposal = subject_fixture()
        main = fixtures.synthetic_raw(
            p,
            profiles=("non_main_only", "non_main_only"),
            deltas=("universal_ui_or_repetition", "universal_ui_or_repetition"),
            basis="verified_equivalent_non_main_message",
        )
        _, _, review = coverage_fixture(p["document_a"]["text"], p["document_b"]["text"])
        verification = {
            "a_subject_kind": "NAMED_ACTUAL_TARGET",
            "b_subject_kind": "NAMED_ACTUAL_TARGET",
            "comparison": "SUPPORTED_DIFFERENT_NAMED_TARGETS",
            "explanation": "Different actual forums.",
        }
        answers = [main, json.dumps(review), json.dumps(proposal), json.dumps(verification)]
    else:
        answers = [main, json.dumps(good)]
    old, old_requests = run_case(subject.old, tmp_path / "old", deepcopy(p), answers)
    new, new_requests = run_case(subject, tmp_path / "new", deepcopy(p), answers)
    assert new_requests == old_requests
    assert {k: v for k, v in new.items() if k != "version"} == {k: v for k, v in old.items() if k != "version"}
    assert "coverage_evidence_repair" not in new
    assert "response_format" not in new_requests[0]


def test_conflict_evidence_gets_one_repair_without_changing_other_decisions(tmp_path):
    p, main, bad, good = conflict_fixture()
    old, old_requests = run_case(subject.old, tmp_path / "old", deepcopy(p), [main, json.dumps(bad)])
    new, new_requests = run_case(subject, tmp_path / "new", deepcopy(p), [main, json.dumps(bad), json.dumps(good)])
    assert old["error_code"] == "CRITIC_SCOPE_CONFLICT"
    assert new["status"] == "VALID"
    assert new_requests[:2] == old_requests
    assert len(new_requests) == 3
    assert new_requests[2]["response_format"] == old_requests[1]["response_format"]
    assert new_requests[2]["messages"][:2] == old_requests[1]["messages"]
    assert new["raw_main"] == old["raw_main"]
    assert new["components"]["main"] == old["components"]["main"]
    assert new["main_attempts"] == old["main_attempts"]
    assert new["coverage_evidence_repair"]["original_review"] == bad
    assert new["coverage_evidence_repair"]["repaired_review"] == good
    expected, rule = subject.old.coverage.apply_review(old["components"]["main"], p, good)
    assert new["public"] == expected
    assert new["coverage_rule"] == rule
    assert new["public"]["a_can_replace_b"] == new["public"]["b_can_replace_a"] == "NO"
    subject.old.validate_evidence_offsets(new["public"], p)


@pytest.mark.parametrize(
    "field", ["conflict", "a_loss_span_id", "b_loss_span_id", "overlap_basis", "shared_anchor_ids"]
)
def test_evidence_repair_cannot_change_semantic_fields(tmp_path, field):
    p, main, bad, good = conflict_fixture()
    changes = {
        "conflict": "NONE",
        "a_loss_span_id": "A001",
        "b_loss_span_id": "B001",
        "overlap_basis": "INTERFACE_ONLY",
        "shared_anchor_ids": [],
    }
    good[field] = changes[field]
    result, requests = run_case(subject, tmp_path, p, [main, json.dumps(bad), json.dumps(good)])
    assert len(requests) == 3
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["error_code"] == "EXP1_EVIDENCE_REPAIR_SCOPE"


@pytest.mark.parametrize("repair", ["still_shared", "wrong_side", "unknown_span", "invalid_json"])
def test_invalid_repair_remains_failed_and_never_retries_again(tmp_path, repair):
    p, main, bad, good = conflict_fixture()
    if repair == "still_shared":
        good = deepcopy(bad)
    elif repair == "wrong_side":
        good["a_context_span_id"] = "B001"
    elif repair == "unknown_span":
        good["a_context_span_id"] = "A999"
    answer = "not JSON" if repair == "invalid_json" else json.dumps(good)
    result, requests = run_case(subject, tmp_path, p, [main, json.dumps(bad), answer])
    assert len(requests) == 3
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["public"]["same_duplicate_group"] == "UNRESOLVED"
    assert result["coverage_evidence_repair"]["attempts"] == 1


def test_main_format_correction_behavior_is_not_part_of_this_patch(tmp_path):
    p = fixtures.synthetic_payload("Same.", "Same.")
    main = fixtures.synthetic_raw(p, deltas=("none", "none"))
    answers = [json.dumps(main), main]
    old, old_requests = run_case(subject.old, tmp_path / "old", deepcopy(p), answers)
    new, new_requests = run_case(subject, tmp_path / "new", deepcopy(p), answers)
    assert len(new_requests) == 2
    assert old_requests == new_requests
    assert old["main_attempts"] == new["main_attempts"]
    assert old["public"] == new["public"]
