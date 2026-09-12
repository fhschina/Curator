# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from eval.dedup.judging import critic_subject_kinds as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_critic_subject_binding import fixture


def named_fixture():
    payload, main, value = fixture()
    value.update(a_subject_kind="NAMED_ACTUAL_TARGET", b_subject_kind="NAMED_ACTUAL_TARGET")
    return payload, main, value


def test_named_bilateral_liability_keeps_existing_veto():
    payload, main, value = named_fixture()
    assert subject.apply_review(main, payload, value)[0]["same_duplicate_group"] == "NO"


@pytest.mark.parametrize("kind", subject.KINDS[1:])
@pytest.mark.parametrize("side", ["a", "b"])
def test_either_generic_role_instruction_or_absent_target_blocks_veto(kind, side):
    payload, main, value = named_fixture()
    value[f"{side}_subject_kind"] = kind
    assert subject.apply_review(main, payload, value)[0] == main


def test_named_targets_do_not_expand_permitted_binding_categories():
    payload, main, value = named_fixture()
    value["binding_type"] = "ACCESS_TARGET"
    assert subject.apply_review(main, payload, value)[0] == main


def test_scope_does_not_hide_invalid_original_evidence():
    payload, main, value = named_fixture()
    value.update(a_subject_kind="GENERIC_ROLE_OR_OBJECT", b_subject_span_id="A001")
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(main, payload, value)


def test_old_output_cannot_be_reused_as_new_named_target_response():
    payload, main, value = fixture()
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(main, payload, value)
