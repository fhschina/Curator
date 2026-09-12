# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.analysis import sequencing_experiment as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_policy_experiment import (
    test_both_arms_use_typed_replay_and_count_the_actual_native_column as _native_accounting,
)
from tests.eval.dedup.test_policy_experiment import (
    test_both_typed_arms_keep_strict_raw_and_blind_boundaries as _strict_boundaries,
)
from tests.eval.dedup.test_policy_experiment import (
    test_full_native_typed_policy_first_and_retry as _native_boundary,
)
from tests.eval.dedup.test_policy_experiment import (
    test_invalid_loss_only_retries_its_pair_with_exact_feedback as _outer_retry,
)
from tests.eval.dedup.test_typed_experiment import fixture


@pytest.fixture
def experiment():
    return subject.SequencingExperiment(subject.common.ANALYSIS / "v06228_experiment.json")


def test_presentation_audit_verifies_common_policy_packet_and_unchanged_control(experiment):
    packets, _, mains = fixture("coverage")
    audit = experiment.presentation_audit(packets, mains)
    assert len(audit["called_inputs"]) == 3
    assert all(r["message_sha256"]["control"] != r["message_sha256"]["coverage"] for r in audit["called_inputs"])
    assert audit["semantic_policy"]["name"] == "complete_propositions"


def test_presentation_audit_rejects_a_changed_visible_packet(experiment):
    packets, _, mains = fixture("coverage")
    render = experiment.renderers["coverage"]

    def changed(packet):
        messages = deepcopy(render(packet))
        messages[1]["content"] = messages[1]["content"].replace("</semantic_diff>", "changed</semantic_diff>")
        return messages

    experiment.renderers["coverage"] = changed
    with pytest.raises(DedupEvaluationError, match="SEQUENCING_PACKET_CHANGED"):
        experiment.presentation_audit(packets, mains)


@pytest.mark.parametrize("variant", ["control", "coverage"])
def test_native_corrections_use_the_actual_typed_column_and_remain_replayable(experiment, tmp_path, variant):
    _native_accounting(experiment, tmp_path, variant)


@pytest.mark.parametrize("variant", ["control", "coverage"])
def test_outer_retry_preserves_invalid_pair_history_and_feedback(experiment, tmp_path, variant):
    _outer_retry(experiment, tmp_path, variant)


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize("failure", ["raw_null", "wrong_branch", "unknown_echo", "payload", "prompt", "main_leak"])
def test_reordering_does_not_accept_cross_branch_fields_or_change_boundaries(experiment, variant, failure):
    _strict_boundaries(experiment, variant, failure)


@pytest.mark.parametrize("variant", ["control", "coverage"])
@pytest.mark.parametrize(
    "feedback", [None, {"code": "SELECTION_REFERENCE_INVALID", "message": "Retry", "details": {}}]
)
def test_full_native_first_and_retry_accepts_valid_typed_branches_in_any_key_order(
    experiment, httpserver, tmp_path, variant, feedback
):
    _native_boundary(experiment, httpserver, tmp_path, variant, feedback)
