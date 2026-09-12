# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from eval.dedup.judging import critic_subject_scope as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_critic_subject_binding import fixture


@pytest.mark.parametrize("binding", ["LIABILITY_PARTY", "POLICY_SERVICE", "FAILED_OBJECT"])
def test_supported_subject_types_preserve_proven_veto(binding):
    payload, main, value = fixture()
    value["binding_type"] = binding
    assert subject.apply_review(main, payload, value)[0]["same_duplicate_group"] == "NO"


@pytest.mark.parametrize("binding", ["ACCESS_TARGET", "RECORD_SUBJECT"])
def test_other_subject_types_remain_diagnostic_without_overriding_coverage(binding):
    payload, main, value = fixture()
    value["binding_type"] = binding
    assert subject.apply_review(main, payload, value)[0] == main
    value["b_subject_span_id"] = "A001"
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(main, payload, value)
