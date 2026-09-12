# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.judging import typed_coverage as subject
from eval.dedup.judging.coverage_selection import adapt_selection, compile_selection
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_selection import selection
from tests.eval.dedup.test_coverage_witness import _case, _uncover


def typed(value):
    result = {k: deepcopy(v) for k, v in value.items() if k not in ("a_meaning_in_b", "b_meaning_in_a")}
    result["contract_version"] = subject.CONTRACT
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        old = value[field]
        new = {k: old[k] for k in ("reviewed_unique_ids", "coverage_explanation", "status")}
        if old["status"] == "COVERED":
            new.update(
                harmless_unique_ids=old["reviewed_unique_ids"] if old["coverage_mode"] == "HARMLESS_ONLY" else "",
                opposite_support_ids=old["coverage_counterpart_ids"],
            )
        elif old["status"] == "UNCOVERED":
            new.update(
                {
                    k: old[k]
                    for k in (
                        "source_span_id",
                        "counterpart_span_id",
                        "retention_consequence",
                        "uncovered_type",
                        "counterpart_relation",
                    )
                }
            )
        else:
            new["checked_opposite_ids"] = ""
        result[field] = new
    return result


@pytest.mark.parametrize("side", ["A", "B"])
def test_real_extension_compiles_without_repeating_inactive_fields_or_changing_evidence(side):
    a, b = "FAQ Model Z.", "FAQ Model Z. กฎหมายแพ่งและพาณิชย์ว่าด้วยหุ้นส่วนบริษัท"
    main, old, payload = _case(*((b, a) if side == "A" else (a, b)))
    _uncover(old, payload, side)
    v2 = selection(old)
    new = typed(v2)
    snapshot = deepcopy((new, payload))
    result = subject.compile_typed_coverage(new, payload)
    assert result.selection == v2
    public = adapt_selection(main, result.selection, payload)
    assert public["relation_type"] == "CONTAINMENT"
    assert public["a_can_replace_b"] == ("YES" if side == "A" else "NO")
    assert public["b_can_replace_a"] == ("YES" if side == "B" else "NO")
    assert (new, payload) == snapshot


def test_harmless_context_is_retained_in_audit_without_becoming_a_semantic_counterpart():
    main, old, payload = _case("Policy.", "Policy. Settings", non_main=True)
    v2 = selection(old)
    v2["b_meaning_in_a"].update(coverage_mode="HARMLESS_ONLY", coverage_counterpart_ids="")
    new = typed(v2)
    new["b_meaning_in_a"]["opposite_support_ids"] = "S001"
    result = subject.compile_typed_coverage(new, payload)
    assert result.selection == v2
    assert result.audit["side_derivations"]["b_meaning_in_a"]["context_only_ids"] == ["S001"]
    assert adapt_selection(main, result.selection, payload)["same_duplicate_group"] == "YES"
    old_invalid = deepcopy(v2)
    old_invalid["b_meaning_in_a"]["coverage_counterpart_ids"] = "S001"
    with pytest.raises(DedupEvaluationError):
        compile_selection(old_invalid, payload)


def test_mixed_mode_is_derived_from_a_proper_harmless_subset_with_semantic_support():
    _, old, payload = _case("Core. Settings. Next. Answer here.", "Core. Next. Here is answer.")
    new = typed(selection(old))
    unique = [s["span_id"] for s in payload["semantic_diff_evidence"]["spans"] if s.get("side") == "A"]
    assert len(unique) >= 2
    new["a_meaning_in_b"]["harmless_unique_ids"] = unique[0]
    result = subject.compile_typed_coverage(new, payload)
    assert result.selection["a_meaning_in_b"]["coverage_mode"] == "MIXED"
    assert result.audit["side_derivations"]["a_meaning_in_b"]["retained_unique_ids"]
    new["a_meaning_in_b"]["opposite_support_ids"] = ""
    with pytest.raises(DedupEvaluationError, match="counterparts"):
        subject.compile_typed_coverage(new, payload)


def test_faithful_translation_keeps_none_materiality_and_no_loss_fields_in_raw_branch():
    main, old, payload = _case("Cookies need consent.", "Cookie 需要获得同意。", non_main=True)
    new = typed(selection(old))
    assert "source_span_id" not in new["a_meaning_in_b"]
    result = subject.compile_typed_coverage(new, payload)
    assert result.selection["a_meaning_in_b"]["coverage_mode"] == "SEMANTIC_COUNTERPARTS"
    assert adapt_selection(main, result.selection, payload)["material_difference"] == "NONE"


@pytest.mark.parametrize("kind", ["POLICY_MEANING", "RECORD_IDENTITY", "PAGE_ROLE", "STATE_VERSION", "MEMBERSHIP"])
def test_hard_loss_stays_negative_and_cannot_be_converted_to_harmless(kind):
    main, old, payload = _case(non_main=True)
    _uncover(old, payload, kind=kind)
    new = typed(selection(old))
    result = subject.compile_typed_coverage(new, payload)
    assert adapt_selection(main, result.selection, payload)["same_duplicate_group"] == "NO"
    new["b_meaning_in_a"]["harmless_unique_ids"] = "B001"
    with pytest.raises(DedupEvaluationError):
        subject.compile_typed_coverage(new, payload)


@pytest.mark.parametrize("bad", ["S001", "A999", "B001"])
def test_harmless_ids_must_be_own_unique_not_a_shared_anchor_or_unknown_side(bad):
    _, old, payload = _case("Shared. Left fact.", "Shared. Right fact.")
    new = typed(selection(old))
    new["a_meaning_in_b"]["harmless_unique_ids"] = bad
    with pytest.raises(DedupEvaluationError):
        subject.compile_typed_coverage(new, payload)


@pytest.mark.parametrize("bad", ["", "B999", "S001", "B001,B002"])
def test_uncovered_keeps_strict_nonempty_single_own_source_selection(bad):
    _, old, payload = _case()
    _uncover(old, payload)
    new = typed(selection(old))
    new["b_meaning_in_a"]["source_span_id"] = bad
    with pytest.raises(DedupEvaluationError):
        subject.compile_typed_coverage(new, payload)


@pytest.mark.parametrize("pruning", [False, True])
@pytest.mark.parametrize("status", ["COVERED", "UNCOVERED", "UNRESOLVED"])
def test_real_native_recipe_preserves_all_discriminated_branch_fields(pruning, status):
    from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

    _, old, payload = _case()
    if status == "UNCOVERED":
        _uncover(old, payload)
    elif status == "UNRESOLVED":
        old["b_meaning_in_a"].update(status=status, coverage_mode="NOT_COVERED", coverage_counterpart_ids="")
    new = typed(selection(old))
    recipe = StructuredResponseRecipe(subject.typed_coverage_schema(), pruning=pruning)
    parsed = recipe.parse("```json\n" + json.dumps(new) + "\n```")
    assert parsed == new
    subject.compile_typed_coverage(parsed, payload)


@pytest.mark.parametrize("failure", ["legacy", "extra_null", "wrong_branch", "wrong_type", "missing"])
def test_strict_raw_contract_rejects_legacy_extra_or_malformed_branch(failure):
    _, old, payload = _case()
    new = typed(selection(old))
    if failure == "legacy":
        new = selection(old)
    elif failure == "extra_null":
        new["b_meaning_in_a"]["source_span_id"] = None
    elif failure == "wrong_branch":
        new["b_meaning_in_a"]["status"] = "UNCOVERED"
    elif failure == "wrong_type":
        new["b_meaning_in_a"]["harmless_unique_ids"] = []
    else:
        new["b_meaning_in_a"].pop("opposite_support_ids")
    with pytest.raises(DedupEvaluationError):
        subject.compile_typed_coverage(new, payload)


def test_known_null_padding_is_bound_only_after_strict_original_response_validation():
    _, old, _ = _case()
    new = typed(selection(old))
    observed = deepcopy(new)
    observed["b_meaning_in_a"]["source_span_id"] = None
    snapshot = deepcopy((new, observed))
    bound, audit = subject.bind_typed_response(new, observed)
    assert bound == new
    assert audit["representation_changes"] == {"added_null_keys": 1}
    assert (new, observed) == snapshot
    with pytest.raises(DedupEvaluationError):
        subject.bind_typed_response(observed, observed)


@pytest.mark.parametrize("failure", ["root_null", "foreign_null", "non_null", "missing", "changed_status"])
def test_response_transport_does_not_tolerate_nonstructural_changes(failure):
    _, old, _ = _case()
    new = typed(selection(old))
    observed = deepcopy(new)
    if failure == "root_null":
        observed["invented"] = None
    elif failure == "foreign_null":
        observed["b_meaning_in_a"]["invented"] = None
    elif failure == "non_null":
        observed["b_meaning_in_a"]["source_span_id"] = "B001"
    elif failure == "missing":
        observed["b_meaning_in_a"].pop("opposite_support_ids")
    else:
        observed["b_meaning_in_a"]["status"] = "UNRESOLVED"
    with pytest.raises(DedupEvaluationError):
        subject.bind_typed_response(new, observed)


def test_actual_arrow_struct_union_padding_keeps_strict_raw_and_transport_validation_separate():
    import pyarrow as pa

    _, old, payload = _case()
    covered = typed(selection(old))
    _uncover(old, payload)
    uncovered = typed(selection(old))
    originals = [covered, uncovered]
    echoed = pa.Table.from_pylist([{"response": r} for r in originals]).to_pylist()
    changes = 0
    for value, row in zip(originals, echoed, strict=True):
        bound, audit = subject.bind_typed_response(value, row["response"])
        assert bound == value
        changes += audit["representation_changes"].get("added_null_keys", 0)
        with pytest.raises(DedupEvaluationError):
            subject.validate_typed_shape(row["response"])
    assert changes > 0


def test_incomplete_input_cannot_resolve_but_can_keep_a_typed_low_confidence_abstention():
    main, old, payload = _case()
    payload["long_document_evidence"]["truncated"] = True
    with pytest.raises(DedupEvaluationError):
        subject.compile_typed_coverage(typed(selection(old)), payload)
    old.update(input_status="UNRESOLVED", record_scope="UNRESOLVED", anchor_a_ids="", anchor_b_ids="")
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        old[field].update(status="UNRESOLVED", coverage_mode="NOT_COVERED", coverage_counterpart_ids="")
    compiled = subject.compile_typed_coverage(typed(selection(old)), payload)
    public = adapt_selection(main, compiled.selection, payload)
    assert public["same_duplicate_group"] == "UNRESOLVED"
    assert public["confidence_tier"] == "LOW"
