# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from eval.dedup.judging import critic_subject_proof_verifier as verifier
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_critic_subject_binding import fixture


def verification_fixture():
    payload, base, proposal = fixture()
    payload["subject_proposal"] = proposal
    value = {
        "a_subject_kind": "NAMED_ACTUAL_TARGET",
        "b_subject_kind": "NAMED_ACTUAL_TARGET",
        "comparison": "SUPPORTED_DIFFERENT_NAMED_TARGETS",
        "explanation": "Two specific exemption subjects.",
    }
    return payload, base, value


def test_verifier_only_approves_the_existing_aligned_veto():
    payload, base, value = verification_fixture()
    actual, rule = verifier.apply_review(base, payload, value)
    assert actual == verifier.proposed_result(base, payload)[0]
    assert rule == "VERIFIED_FIXED_SUBJECT_VETO"


@pytest.mark.parametrize("kind", verifier.kinds.KINDS[1:])
def test_generic_or_unsupported_proposal_preserves_exact_coverage(kind):
    payload, base, value = verification_fixture()
    value.update(a_subject_kind=kind, comparison="UNSUPPORTED_COMPARISON")
    assert verifier.apply_review(base, payload, value)[0] == base


def test_verifier_cannot_invent_a_veto_when_original_scope_declines():
    payload, base, value = verification_fixture()
    payload["subject_proposal"]["binding_type"] = "RECORD_SUBJECT"
    assert verifier.apply_review(base, payload, value)[0] == base


def test_bad_upstream_proof_is_not_hidden_by_a_negative_verifier():
    payload, base, value = verification_fixture()
    payload["subject_proposal"]["a_subject_span_id"] = "B001"
    value["comparison"] = "UNSUPPORTED_COMPARISON"
    with pytest.raises(DedupEvaluationError):
        verifier.apply_review(base, payload, value)


def test_contradictory_named_target_attestation_is_invalid():
    payload, base, value = verification_fixture()
    value["a_subject_kind"] = "GENERIC_ROLE_OR_OBJECT"
    with pytest.raises(DedupEvaluationError):
        verifier.apply_review(base, payload, value)


def test_request_keeps_original_full_text_and_selected_targets_without_old_reasoning():
    payload, _, _ = verification_fixture()
    text = verifier.messages(payload, "system")[1]["content"]
    assert payload["document_a"]["text"] in text
    assert payload["document_b"]["text"] in text
    assert payload["subject_proposal"]["explanation"] not in text
    assert "Cedar" in text
    assert "Birch" in text
