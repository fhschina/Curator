# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import retention_v4_proof as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_retention_v4 import payload, review


def conflict_fixture():
    p = payload("Selected SKU A11 is available.", "Selected SKU B22 is available.")
    value = review(p, "MAIN_ADDITION", "MAIN_ADDITION", conflict="IDENTITY_CONFLICT")
    value["a_context_span_id"] = value["b_context_span_id"] = "S001"
    return p, value


def test_existing_unique_loss_values_and_shared_predicate_form_a_valid_conflict_proof():
    p, value = conflict_fixture()
    before = deepcopy((p, value))
    with pytest.raises(DedupEvaluationError, match="RETENTION_V4_CONFLICT"):
        subject.previous.adapt_main(value, p)
    out = subject.adapt_main(value, p)
    assert out["a_can_replace_b"] == out["b_can_replace_a"] == "NO"
    assert {e["quote"] for e in out["evidence"]} == {"A11", "B22", "Selected SKU"}
    assert (p, value) == before


def test_correct_negation_critic_can_use_its_existing_loss_witness_without_inventing_new_context():
    p = payload("This claim is false: The service is free.", "The service is free.")
    main = subject.previous.adapt_main(review(p), p)
    value = review(p, "MAIN_ADDITION", conflict="NEGATION_CONFLICT")
    value["a_context_span_id"] = value["b_context_span_id"] = "S001"
    out, rule = subject.apply_critic(main, p, value)
    assert out["same_duplicate_group"] == "NO"
    assert out["primary_material_difference"] == "NEGATION_CHANGE"
    assert rule == "SUPPORTED_DIRECTIONAL_VETO"


@pytest.mark.parametrize(
    "damage", ["no_unique", "wrong_side", "fabricated", "minor", "missing_context", "extra_field"]
)
def test_validation_fix_does_not_waive_exact_bilateral_proof_or_severity(damage):
    p, value = conflict_fixture()
    if damage == "no_unique":
        value.update(a_loss_kind="NONE", b_loss_kind="NONE", a_loss_span_id="", b_loss_span_id="")
    elif damage == "wrong_side":
        value["a_loss_span_id"] = "B001"
    elif damage == "fabricated":
        value["b_loss_span_id"] = "B999"
    elif damage == "minor":
        value["material_difference"] = "MINOR"
    elif damage == "missing_context":
        value["b_context_span_id"] = ""
    else:
        value["forged"] = "proof"
    with pytest.raises(DedupEvaluationError):
        subject.adapt_main(value, p)


def test_nonconflict_and_already_negative_paths_are_byte_for_byte_unchanged():
    p = payload("Policy.\nNext page", "Policy.")
    value = review(p, "NON_MAIN_ADDITION")
    main = subject.previous.adapt_main(value, p)
    assert subject.adapt_main(value, p) == main
    assert subject.apply_critic(main, p, value) == subject.previous.apply_critic(main, p, value)
    p, value = conflict_fixture()
    negative = subject.adapt_main(value, p)
    assert subject.apply_critic(negative, p, None)[0] == negative
