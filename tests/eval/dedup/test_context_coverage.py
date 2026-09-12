# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import composite_coverage as legacy
from eval.dedup.judging import context_coverage as subject
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_composite_coverage import certificate


def context_certificate(reverse=False, prefix=""):
    texts = (prefix + "本轮入选的完整名单，仅限：甲、乙。", prefix + "本轮入选的完整名单，仅限：甲、乙、丙。")  # noqa: RUF001
    value, payload, _, _ = certificate(
        *(texts[::-1] if reverse else texts), loss_sides=("A", "B"), kind="MEMBERSHIP", conflict=True
    )
    value["contract_version"] = subject.CONTRACT
    for side, field in subject.SIDES.items():
        value[field] = {
            "status": "UNCOVERED",
            "reviewed_unique_ids": value[field]["reviewed_unique_ids"],
            "coverage_explanation": "The exclusive complete list asserts incompatible membership.",
            "uncovered_type": "MEMBERSHIP",
            "source_span_id": value["hard_conflict"][f"{side.lower()}_span_id"],
            "counterpart_span_id": value["hard_conflict"]["b_span_id" if side == "A" else "a_span_id"],
            "counterpart_relation": "CONTRADICTS",
            "retention_consequence": "Replacing either side changes the closed membership assertion.",
        }
    return value, payload


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_context_contradiction_preserves_honest_side_statuses_and_exact_evidence(reverse):
    value, payload = context_certificate(reverse)
    snapshot = deepcopy((value, payload))
    with pytest.raises(DedupEvaluationError):
        legacy.adapt_composite(value | {"contract_version": legacy.CONTRACT}, payload)
    public = subject.adapt_context(value, payload)
    assert public["a_can_replace_b"] == public["b_can_replace_a"] == "NO"
    assert public["primary_material_difference"] == "RESULT_SET_CHANGE"
    assert public["material_difference"] == "MAJOR"
    assert any("丙" in e["quote"] for e in public["evidence"])
    assert {e["side"] for e in public["evidence"]} == {"A", "B"}
    assert all(value[f]["status"] == "UNCOVERED" for f in subject.SIDES.values())
    validate_evidence_offsets(public, payload)
    assert (value, payload) == snapshot
    assert subject.finalize_context(value, None, payload) == public


@pytest.mark.parametrize(
    "failure",
    [
        "no_equivalent",
        "no_conflict",
        "two_shared",
        "unmatched",
        "wrong_kind",
        "missing_inventory",
        "invented",
        "inactive",
        "translation",
        "truncated",
        "v4",
    ],
)
def test_shared_exception_cannot_be_used_as_generic_missing_content_or_unpaired_conflict(failure):  # noqa: C901
    value, payload = context_certificate()
    if failure == "no_equivalent":
        value["a_meaning_in_b"]["counterpart_relation"] = "NO_EQUIVALENT_FOUND"
    elif failure == "no_conflict":
        value["hard_conflict"].update(kind="NONE", a_span_id="", b_span_id="")
    elif failure == "two_shared":
        value["hard_conflict"]["b_span_id"] = "S001"
        value["a_meaning_in_b"]["counterpart_span_id"] = "S001"
    elif failure == "unmatched":
        value["a_meaning_in_b"]["counterpart_span_id"] = "S002"
    elif failure == "wrong_kind":
        value["a_meaning_in_b"]["uncovered_type"] = "MAIN_CONTENT"
    elif failure == "missing_inventory":
        value["b_meaning_in_a"]["reviewed_unique_ids"] = ""
    elif failure == "invented":
        payload["semantic_diff_evidence"]["spans"][0]["a_text"] = "invented"
    elif failure == "inactive":
        value["a_meaning_in_b"]["harmless_unique_ids"] = None
    elif failure == "translation":
        value["translation_status"] = "COMPLETE_FAITHFUL"
    elif failure == "truncated":
        payload["long_document_evidence"]["truncated"] = True
    else:
        value["contract_version"] = legacy.CONTRACT
    with pytest.raises(DedupEvaluationError):
        subject.adapt_context(value, payload)


@pytest.mark.parametrize("basis", ["SHARED_SUBSTANTIVE_CONTENT", "NON_MAIN_MESSAGES", "NO_SHARED_SUBSTANTIVE_CONTENT"])
@pytest.mark.parametrize("loss_sides", [(), ("A",), ("B",), ("A", "B")])
def test_ordinary_unique_witnesses_keep_frozen_adapter_behavior(basis, loss_sides):
    value, payload, _, _ = certificate(
        "Core. Astronomy papers.", "Core. Zoology lectures.", basis=basis, loss_sides=loss_sides
    )
    before = legacy.adapt_composite(value, payload)
    assert subject.adapt_context(value | {"contract_version": subject.CONTRACT}, payload) == before
    validate_evidence_offsets(before, payload)


@pytest.mark.parametrize("kind", ["RECORD_IDENTITY", "STATE_VERSION", "PAGE_ROLE", "MEMBERSHIP", "POLICY_MEANING"])
def test_actual_unique_conflicts_are_unchanged(kind):
    value, payload, _, _ = certificate(
        "Record. Open.", "Record. Closed.", loss_sides=("A", "B"), kind=kind, conflict=True
    )
    assert subject.adapt_context(value | {"contract_version": subject.CONTRACT}, payload) == legacy.adapt_composite(
        value, payload
    )


@pytest.mark.parametrize("basis", ["SHARED_SUBSTANTIVE_CONTENT", "NON_MAIN_MESSAGES"])
def test_whole_translation_and_translated_core_plus_retained_extra_have_distinct_contracts(basis):
    value, payload, _, _ = certificate("Cookies require consent.", "Cookie 需要获得同意。", basis=basis, loss_sides=())
    value.update(contract_version=subject.CONTRACT, translation_status="COMPLETE_FAITHFUL")
    public = subject.adapt_context(value, payload)
    assert public["a_can_replace_b"] == public["b_can_replace_a"] == "YES"
    assert public["material_difference"] == "NONE"
    value, payload, _, _ = certificate("Use footage freely.", "素材可自由使用。可邮件申请定制服务。", basis=basis)
    value.update(contract_version=subject.CONTRACT, translation_status="COMPLETE_FAITHFUL")
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_TRANSLATION"):
        subject.adapt_context(value, payload)
    value["translation_status"] = "OTHER"
    public = subject.adapt_context(value, payload)
    assert public["relation_type"] == (
        "CONTAINMENT" if basis == "SHARED_SUBSTANTIVE_CONTENT" else "RELATED_NON_DUPLICATE"
    )


def test_raw_contract_and_transport_are_not_silently_rewritten():
    value, _ = context_certificate()
    observed = deepcopy(value)
    observed["a_meaning_in_b"]["harmless_unique_ids"] = None
    assert subject.bind_context_response(value, observed)[0] == value
    with pytest.raises(DedupEvaluationError):
        subject.bind_context_response(observed, observed)
    observed["a_meaning_in_b"]["foreign"] = None
    with pytest.raises(DedupEvaluationError):
        subject.bind_context_response(value, observed)


def test_truncation_exact_and_unresolved_ownership_remain_conservative():
    _, payload, _, _ = certificate("Cookies.", "Cookies.", loss_sides=())
    assert subject.owned_output(payload)["relation_type"] == "EXACT"
    payload["long_document_evidence"]["truncated"] = True
    assert subject.owned_output(payload)["confidence_tier"] == "LOW"
    value, payload = context_certificate()
    value["shared_basis"] = "UNRESOLVED"
    assert subject.adapt_context(value, payload)["confidence_tier"] == "LOW"
