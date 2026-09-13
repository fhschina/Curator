# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import checkpoint_preflight as fixtures
from eval.dedup.analysis import exp5_anchor_id_fix as subject
from tests.eval.dedup.test_critic_retention_v3 import fixture as coverage_fixture
from tests.eval.dedup.test_critic_subject_binding import fixture as subject_fixture
from tests.eval.dedup.test_exp1_conflict_evidence_fix import conflict_fixture, run_case


@pytest.mark.parametrize(
    "kind",
    [
        "exact",
        "negative",
        "containment",
        "conflict_repair",
        "forbidden_repair",
        "invalid_anchor",
        "subject_verifier",
        "main_retry",
    ],
)
def test_unaffected_requests_retries_routing_and_results_match_frozen_exp5(tmp_path, kind):
    payload, main, bad, good = conflict_fixture()
    if kind in {"exact", "main_retry"}:
        payload = fixtures.synthetic_payload("Same retained text.", "Same retained text.")
        main = fixtures.synthetic_raw(payload, deltas=("none", "none"))
        answers = [main] if kind == "exact" else [json.dumps(main), main]
    elif kind == "negative":
        main["span_hard_conflict"] = {"score": "identity_or_slot", "reasoning": "A001 and B001 are different forums."}
        answers = [main]
    elif kind in {"conflict_repair", "forbidden_repair"}:
        if kind == "forbidden_repair":
            good["conflict"] = "NONE"
        answers = [main, json.dumps(bad), json.dumps(good)]
    elif kind == "invalid_anchor":
        good["shared_anchor_ids"] = ["S99"]
        answers = [main, json.dumps(good)]
    elif kind == "subject_verifier":
        payload, _, proposal = subject_fixture()
        main = fixtures.synthetic_raw(
            payload,
            profiles=("non_main_only", "non_main_only"),
            deltas=("universal_ui_or_repetition", "universal_ui_or_repetition"),
            basis="verified_equivalent_non_main_message",
        )
        _, _, review = coverage_fixture(payload["document_a"]["text"], payload["document_b"]["text"])
        verification = {
            "a_subject_kind": "NAMED_ACTUAL_TARGET",
            "b_subject_kind": "NAMED_ACTUAL_TARGET",
            "comparison": "SUPPORTED_DIFFERENT_NAMED_TARGETS",
            "explanation": "Different actual forums.",
        }
        answers = [main, json.dumps(review), json.dumps(proposal), json.dumps(verification)]
    else:
        payload, _, review = coverage_fixture()
        main = fixtures.synthetic_raw(payload, deltas=("none", "semantically_covered"))
        review["b_loss_span_id"] = "B001"
        answers = [main, json.dumps(review)]
    baseline, before = run_case(subject.previous, tmp_path / "baseline", deepcopy(payload), answers)
    patched, after = run_case(subject, tmp_path / "patched", deepcopy(payload), answers)
    assert before == after
    assert {k: v for k, v in patched.items() if k != "version"} == {
        k: v for k, v in baseline.items() if k != "version"
    }
    assert patched["version"] != baseline["version"]
    assert "coverage_anchor_id_repair" not in patched


@pytest.mark.parametrize("kind", ["containment", "identity_conflict", "unsupported_conflict", "subject_verifier"])
def test_repair_reuses_same_calls_and_original_proof_validation(tmp_path, kind):
    payload, main, bad, good = conflict_fixture()
    tail = [json.dumps(good)] if kind == "unsupported_conflict" else []
    if kind == "containment":
        payload, _, good = coverage_fixture()
        main = fixtures.synthetic_raw(payload, deltas=("none", "semantically_covered"))
        good["b_loss_span_id"] = "B001"
    elif kind == "subject_verifier":
        payload, _, proposal = subject_fixture()
        main = fixtures.synthetic_raw(
            payload,
            profiles=("non_main_only", "non_main_only"),
            deltas=("universal_ui_or_repetition", "universal_ui_or_repetition"),
            basis="verified_equivalent_non_main_message",
        )
        _, _, good = coverage_fixture(payload["document_a"]["text"], payload["document_b"]["text"])
        tail = [
            json.dumps(proposal),
            json.dumps(
                {
                    "a_subject_kind": "NAMED_ACTUAL_TARGET",
                    "b_subject_kind": "NAMED_ACTUAL_TARGET",
                    "comparison": "SUPPORTED_DIFFERENT_NAMED_TARGETS",
                    "explanation": "Different actual forums.",
                }
            ),
        ]
    canonical = bad if kind == "unsupported_conflict" else good
    raw = {**canonical, "shared_anchor_ids": [sid[0] + str(int(sid[1:])) for sid in canonical["shared_anchor_ids"]]}
    baseline, baseline_requests = run_case(
        subject.previous, tmp_path / "baseline", deepcopy(payload), [main, json.dumps(raw)]
    )
    patched, requests = run_case(subject, tmp_path / "patched", deepcopy(payload), [main, json.dumps(raw), *tail])
    expected, expected_requests = run_case(
        subject.previous, tmp_path / "canonical", deepcopy(payload), [main, json.dumps(canonical), *tail]
    )
    assert baseline["error_code"] == "CRITIC_SCOPE_ANCHOR"
    assert patched["status"] == expected["status"] == "VALID"
    assert requests[:2] == baseline_requests
    assert requests == expected_requests
    assert patched["public"] == expected["public"]
    assert patched["components"] == expected["components"]
    assert patched["main_attempts"] == baseline["main_attempts"]
    assert patched["raw_main"] == baseline["raw_main"]
    assert [stage["stage"] for stage in patched["stages"]] == [stage["stage"] for stage in expected["stages"]]
    audit = patched["coverage_anchor_id_repair"]
    assert audit["original_review"] == raw
    assert audit["normalized_review"] == canonical
    assert audit["additional_model_calls"] == 0
    assert bool(patched.get("coverage_evidence_repair")) == (kind == "unsupported_conflict")
    response_path = tmp_path / "patched/responses/test-pair-coverage.json"
    response = json.loads(response_path.read_text())
    assert json.loads(response["raw_response"]["choices"][0]["message"]["content"]) == raw
    subject.old.validate_evidence_offsets(patched["public"], payload)
