# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis.composite_containment_review import SPEC
from eval.dedup.judging import composite_coverage as subject
from eval.dedup.judging.coverage_witness import adapt_coverage_witness
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_selection import selection
from tests.eval.dedup.test_coverage_witness import _case, _uncover
from tests.eval.dedup.test_typed_coverage import typed


def certificate(  # noqa: PLR0913
    a="Library closes Monday.",
    b="Library closes Monday. Independent collection: astronomy papers.",
    *,
    loss_sides=("B",),
    basis="SHARED_SUBSTANTIVE_CONTENT",
    kind="MAIN_CONTENT",
    conflict=False,
):
    main, old, payload = _case(a, b, non_main=basis == "NON_MAIN_MESSAGES")
    unique_sides = {s.get("side") for s in payload["semantic_diff_evidence"]["spans"] if s["kind"] != "SHARED"}
    loss_sides = tuple(s for s in loss_sides if s in unique_sides)
    for side in loss_sides:
        _uncover(old, payload, side, kind)
    value = typed(selection(old))
    value["contract_version"] = subject.CONTRACT
    value["shared_basis"] = basis
    value["basis_explanation"] = (
        "The same closure notice is retained; an independent collection is additional content."
    )
    value.pop("record_scope")
    value.pop("scope_explanation")
    value.update(
        hard_conflict={
            "kind": "NONE",
            "a_span_id": "",
            "b_span_id": "",
            "explanation": "No incompatible retained values.",
        },
        translation_status="OTHER",
        dominant_overlap_source="MAIN_CONTENT",
    )
    if conflict:
        ids = {
            s: next(
                i["span_id"]
                for i in payload["semantic_diff_evidence"]["spans"]
                if i.get("side") == s or (s not in unique_sides and i["kind"] == "SHARED")
            )
            for s in ("A", "B")
        }
        value["hard_conflict"] = {
            "kind": kind,
            "a_span_id": ids["A"],
            "b_span_id": ids["B"],
            "explanation": "Both actual values are mutually incompatible for this subject.",
        }
        for side in loss_sides:
            value[subject.SIDES[side]].update(
                counterpart_relation="CONTRADICTS", counterpart_span_id=ids["B" if side == "A" else "A"]
            )
    return value, payload, main, old


@pytest.mark.parametrize("side", ["A", "B"])
def test_independent_content_is_not_rejected_by_legacy_single_record_scope(side):
    x, xy = "Library closes Monday.", "Library closes Monday. Independent collection: astronomy papers."
    value, payload, main, old = certificate(*((xy, x) if side == "A" else (x, xy)), loss_sides=(side,))
    old["record_scope"] = "DISTINCT_OR_UNBOUND_RECORDS"
    snapshot = deepcopy((value, payload, main, old))
    assert adapt_coverage_witness(main, old, payload)["same_duplicate_group"] == "NO"
    public = subject.adapt_composite(value, payload)
    assert public["relation_type"] == "CONTAINMENT"
    assert public["a_can_replace_b"] == ("YES" if side == "A" else "NO")
    assert public["b_can_replace_a"] == ("YES" if side == "B" else "NO")
    assert public["primary_material_difference"] == "MAIN_CONTENT_ADDITION_DELETION"
    validate_evidence_offsets(public, payload)
    assert (value, payload, main, old) == snapshot


@pytest.mark.parametrize("basis", ["NO_SHARED_SUBSTANTIVE_CONTENT", "NON_MAIN_MESSAGES"])
def test_no_empty_main_containment_even_with_complete_literal_inclusion(basis):
    value, payload, _, _ = certificate("We use cookies.", "We use cookies. Camera K7 costs 500.", basis=basis)
    value["dominant_overlap_source"] = "COOKIE_CONSENT"
    public = subject.adapt_composite(value, payload)
    assert public["a_can_replace_b"] == public["b_can_replace_a"] == "NO"
    assert public["relation_type"] != "CONTAINMENT"


@pytest.mark.parametrize("kind", ["RECORD_IDENTITY", "STATE_VERSION", "PAGE_ROLE", "MEMBERSHIP", "POLICY_MEANING"])
def test_actual_bilateral_conflicts_override_shared_core(kind):
    value, payload, _, _ = certificate(
        "Record. Open.", "Record. Closed.", loss_sides=("A", "B"), kind=kind, conflict=True
    )
    public = subject.adapt_composite(value, payload)
    assert public["same_duplicate_group"] == "NO"
    assert public["material_difference"] == "MAJOR"
    assert public["relation_type"] == ("VERSION_RELATED" if kind == "STATE_VERSION" else "RELATED_NON_DUPLICATE")
    assert {e["quote"] for e in public["evidence"]} >= {"Open", "Closed"}


def test_two_sided_independent_content_is_not_containment():
    value, payload, _, _ = certificate("Core. Astronomy papers.", "Core. Zoology lectures.", loss_sides=("A", "B"))
    assert subject.adapt_composite(value, payload)["same_duplicate_group"] == "NO"


@pytest.mark.parametrize("basis", ["SHARED_SUBSTANTIVE_CONTENT", "NON_MAIN_MESSAGES"])
def test_harmless_controls_are_bilateral_and_faithful_translation_keeps_none(basis):
    value, payload, _, _ = certificate(
        "We use cookies.", "We use cookies. Accept Settings", loss_sides=(), basis=basis
    )
    side = value["b_meaning_in_a"]
    side["harmless_unique_ids"] = side["reviewed_unique_ids"]
    side["opposite_support_ids"] = ""
    public = subject.adapt_composite(value, payload)
    assert public["a_can_replace_b"] == public["b_can_replace_a"] == "YES"
    assert public["material_difference"] == "MINOR"
    value, payload, _, _ = certificate("Cookies require consent.", "Cookie 需要获得同意。", loss_sides=(), basis=basis)
    value["translation_status"] = "COMPLETE_FAITHFUL"
    public = subject.adapt_composite(value, payload)
    assert public["a_can_replace_b"] == public["b_can_replace_a"] == "YES"
    assert public["material_difference"] == "NONE"


def test_fresh_main_ownership_and_critic_changes_are_separable():
    positive, payload, legacy, _ = certificate()
    negative = deepcopy(positive)
    negative["shared_basis"] = "NO_SHARED_SUBSTANTIVE_CONTENT"
    assert subject.critic_route(positive, payload) == ("NEEDS_COMPOSITE_CRITIC", None)
    assert subject.adapt_composite(positive, payload)["same_duplicate_group"] == "YES"
    assert subject.finalize_composite(positive, negative, payload)["same_duplicate_group"] == "NO"
    assert subject.finalize_composite(negative, None, payload)["same_duplicate_group"] == "NO"
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_CRITIC_MISSING"):
        subject.finalize_composite(positive, None, payload)
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_UNREQUESTED_CRITIC"):
        subject.finalize_composite(negative, positive, payload)
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_CONTRACT_INVALID"):
        subject.critic_route(legacy, payload)


@pytest.mark.parametrize(
    "failure",
    [
        "legacy_version",
        "legacy_scope",
        "foreign_field",
        "missing_inventory",
        "unknown_anchor",
        "unmatched_conflict",
        "missing_conflict",
        "false_translation",
        "changed_quote",
        "inactive_witness",
    ],
)
def test_new_contract_rejects_old_responses_and_inconsistent_proofs(failure):
    value, payload, _, _ = certificate()
    actions = {
        "legacy_version": lambda: value.update(contract_version="dedup-retained-coverage-v3"),
        "legacy_scope": lambda: value.update(record_scope="SAME_SUBSTANTIVE_RECORD"),
        "foreign_field": lambda: value["b_meaning_in_a"].update(harmless_unique_ids=""),
        "missing_inventory": lambda: value["b_meaning_in_a"].update(reviewed_unique_ids=""),
        "unknown_anchor": lambda: value.update(anchor_a_ids="A999"),
        "unmatched_conflict": lambda: value["hard_conflict"].update(
            kind="RECORD_IDENTITY", a_span_id="S001", b_span_id="B001"
        ),
        "missing_conflict": lambda: value["b_meaning_in_a"].update(
            counterpart_relation="CONTRADICTS", counterpart_span_id="S001"
        ),
        "false_translation": lambda: value.update(translation_status="COMPLETE_FAITHFUL"),
        "changed_quote": lambda: payload["semantic_diff_evidence"]["spans"][0].update(a_text="invented"),
        "inactive_witness": lambda: value["hard_conflict"].update(a_span_id="S001"),
    }
    actions[failure]()
    with pytest.raises(DedupEvaluationError):
        subject.adapt_composite(value, payload)


def test_strict_raw_binding_only_accepts_known_native_null_padding():
    value, _, _, _ = certificate()
    padded = deepcopy(value)
    padded["a_meaning_in_b"]["source_span_id"] = None
    assert subject.bind_composite_response(value, padded)[0] == value
    with pytest.raises(DedupEvaluationError):
        subject.bind_composite_response(padded, padded)
    padded["a_meaning_in_b"]["unknown"] = None
    with pytest.raises(DedupEvaluationError):
        subject.bind_composite_response(value, padded)


def test_owned_exact_does_not_infer_substantive_role_from_equal_cookie_text():
    _, payload, _, _ = certificate("We use cookies.", "We use cookies.", loss_sides=())
    public = subject.owned_output(payload)
    assert public["relation_type"] == "EXACT"
    assert public["dominant_overlap_source"] == "LOCAL_PASSAGE"
    assert "COMPOSITE_TEXT_ROLE:NOT_INFERRED" in public["reason_codes"]
    assert public["evidence"] == []


@pytest.mark.parametrize("example", json.loads(SPEC.read_text())["examples"], ids=lambda r: r["id"])
@pytest.mark.parametrize("reverse", [False, True])
def test_adapter_against_frozen_paired_synthetic_policy_expectations(example, reverse):
    a, b = (example[k] for k in (("b", "a") if reverse else ("a", "b")))
    directions = example["directions"][::-1] if reverse else example["directions"]
    if example.get("truncated"):
        _, payload, _, _ = certificate(a, b, loss_sides=())
        payload["long_document_evidence"]["truncated"] = True
        public = subject.owned_output(payload)
    else:
        kinds = {
            "actual_sku_conflict": "RECORD_IDENTITY",
            "actual_state_conflict": "STATE_VERSION",
            "actual_role_conflict": "PAGE_ROLE",
            "actual_membership_conflict": "MEMBERSHIP",
        }
        empty = example["id"] in {"cookie_empty_anchor", "disclaimer_empty_anchor"}
        loss_sides = (
            ("A" if not reverse else "B",)
            if empty
            else tuple(s for s, d in zip(("A", "B"), directions[::-1], strict=True) if d == "NO")
        )
        basis = (
            "NO_SHARED_SUBSTANTIVE_CONTENT"
            if empty
            else "NON_MAIN_MESSAGES"
            if example["id"] == "equivalent_nonmain"
            else "SHARED_SUBSTANTIVE_CONTENT"
        )
        value, payload, _, _ = certificate(
            a,
            b,
            loss_sides=loss_sides,
            basis=basis,
            kind=kinds.get(example["id"], "MAIN_CONTENT"),
            conflict=example["id"] in kinds,
        )
        if example["material"] == "MINOR":
            for field in subject.SIDES.values():
                value[field].update(harmless_unique_ids=value[field]["reviewed_unique_ids"], opposite_support_ids="")
        if example["id"] == "faithful_translation":
            value["translation_status"] = "COMPLETE_FAITHFUL"
        public = subject.adapt_composite(value, payload)
    assert [public["a_can_replace_b"], public["b_can_replace_a"]] == directions
    assert public["relation_type"] == example["relation"]
    assert public["material_difference"] == example["material"]
    assert public["primary_material_difference"] == example["primary_difference"]
    validate_evidence_offsets(public, payload)
