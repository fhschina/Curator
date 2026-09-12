# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import schema_v4 as subject
from eval.dedup.judging.retention_v4 import adapt_main
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_retention_v4 import payload, review


def minor_containment():
    p = payload("Policy applies.\nNext page", "Policy applies.")
    return adapt_main(review(p, "NON_MAIN_ADDITION"), p)


def test_minor_containment_is_v4_valid_but_not_silently_backported_to_v3():
    out = minor_containment()
    before = deepcopy(out)
    assert subject.read_versioned_output(out, subject.JUDGE_SCHEMA_V4) == out
    with pytest.raises(DedupEvaluationError):
        subject.read_versioned_output(out, "dedup-judge-output-v3")
    assert out == before


@pytest.mark.parametrize(
    "version", ["dedup-judge-output-v0", "dedup-judge-output-v1", "dedup-judge-output-v2", "dedup-judge-output-v3"]
)
def test_legacy_dispatch_uses_original_reader_without_reinterpreting_values(version):
    old = subject.legacy.unresolved_judge_output(schema_version=version)
    before = deepcopy(old)
    assert subject.read_versioned_output(old, version) == before
    assert old == before


@pytest.mark.parametrize(
    "damage",
    ["both_yes", "group", "severity", "difference", "one_evidence_side", "numeric_confidence", "minhash", "unknown"],
)
def test_new_contract_rejects_inconsistent_or_unversioned_results(damage):
    out = minor_containment()
    if damage == "both_yes":
        out["b_can_replace_a"] = "YES"
    elif damage == "group":
        out["same_duplicate_group"] = "NO"
    elif damage == "severity":
        out["material_difference"] = "NONE"
    elif damage == "difference":
        out["primary_material_difference"] = "DOCUMENT_IDENTITY_CHANGE"
    elif damage == "one_evidence_side":
        out["evidence"] = [e for e in out["evidence"] if e["side"] == "A"]
    elif damage == "numeric_confidence":
        out["confidence_tier"] = 0.9
    elif damage == "minhash":
        out["expected_minhash_action"] = "GROUP"
    else:
        out["relation_type"] = "CUSTOM"
    with pytest.raises(DedupEvaluationError):
        subject.validate_judge_output_v4(out)


def test_minor_two_sided_difference_is_not_forced_to_major_but_real_version_conflict_is():
    p = payload("Article.\nContact", "Article.\nEvents")
    value = adapt_main(review(p, "NON_MAIN_ADDITION", "NON_MAIN_ADDITION"), p)
    assert subject.validate_judge_output_v4(value)["material_difference"] == "MINOR"
    value["relation_type"] = "VERSION_RELATED"
    with pytest.raises(DedupEvaluationError, match="JUDGE_V4_VERSION"):
        subject.validate_judge_output_v4(value)


def test_unresolved_remains_low_confidence_and_unknown_versions_fail():
    out = subject.unresolved_judge_output_v4("TEST:INCOMPLETE")
    assert out["confidence_tier"] == "LOW"
    assert not out["evidence"]
    with pytest.raises(DedupEvaluationError):
        subject.read_versioned_output(out, "dedup-judge-output-v100")
