# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import coverage_selection as subject
from eval.dedup.judging.coverage_witness import adapt_coverage_witness, parse_coverage
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_witness import _case, _uncover


def selection(value):
    result = deepcopy(value)
    result["contract_version"] = subject.CONTRACT
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        side = result[field]
        side["source_span_id"] = side.pop("source_ids")
        side["counterpart_span_id"] = side.pop("counterpart_ids")
        side.pop("source_quote")
        side.pop("counterpart_quote")
    return result


@pytest.mark.parametrize("side", ["A", "B"])
def test_real_extension_preserves_direction_and_complete_original_span(side):
    a, b = "FAQ Model Z.", "FAQ Model Z. Restart the device."
    main, value, payload = _case(*((b, a) if side == "A" else (a, b)))
    _uncover(value, payload, side)
    new = selection(value)
    before = deepcopy((main, new, payload))
    assert subject.compile_selection(new, payload) == value
    result = subject.adapt_selection(main, new, payload)
    assert result == adapt_coverage_witness(main, value, payload)
    assert result["relation_type"] == "CONTAINMENT"
    assert result["a_can_replace_b"] == ("YES" if side == "A" else "NO")
    assert result["b_can_replace_a"] == ("YES" if side == "B" else "NO")
    assert (main, new, payload) == before


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Same original paragraph.\n\nSame original paragraph.", "Same original paragraph."),
        ("The responsible party.", "The responsi\u00adble party."),
        ("Cookies need consent.", "Cookie 需要获得同意。"),
    ],
)
def test_nonexact_bilateral_semantic_coverage_never_requires_exact_scope(a, b):
    main, value, payload = _case(a, b, non_main=True)
    new = selection(value)
    result = subject.adapt_selection(main, new, payload)
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "YES"
    assert result["relation_type"] != "EXACT"
    new["record_scope"] = "IDENTICAL_TEXT"
    with pytest.raises(DedupEvaluationError):
        subject.compile_selection(new, payload)


def test_selecting_thai_span_preserves_combining_characters_without_model_transcription():
    main, value, payload = _case("FAQ. กฎหมายแพ่งและพาณิชย์ว่าด้วยหุ้นส่วนบริษัท", "FAQ.")
    _uncover(value, payload, "A")
    value["a_meaning_in_b"]["source_quote"] = "misspelled Thai text"
    with pytest.raises(DedupEvaluationError):
        parse_coverage(value, payload)
    new = selection(value)
    compiled = subject.compile_selection(new, payload)
    span = next(s for s in payload["semantic_diff_evidence"]["spans"] if s.get("side") == "A")
    assert compiled["a_meaning_in_b"]["source_quote"] == span["text"]
    assert subject.adapt_selection(main, new, payload)["same_duplicate_group"] == "YES"
    with pytest.raises(DedupEvaluationError):
        subject.compile_selection(value, payload)


def test_shared_chrome_does_not_bind_new_listing_when_review_identifies_unbound_scope():
    main, value, payload = _case("Store. Home. Cart.", "Store. Home. Cart. Product X, SKU 123.")
    _uncover(value, payload)
    value["record_scope"] = "DISTINCT_OR_UNBOUND_RECORDS"
    result = subject.adapt_selection(main, selection(value), payload)
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == "NO"
    assert result["material_difference"] == "MAJOR"


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "range",
        "list",
        "wrong_side",
        "shared_source",
        "missing_field",
        "old_quote",
        "inactive_witness",
        "unreviewed",
        "fake_contradiction",
    ],
)
def test_invalid_selection_or_coverage_consistency_is_rejected(bad):
    _, value, payload = _case()
    _uncover(value, payload)
    new = selection(value)
    side = new["b_meaning_in_a"]
    if bad == "missing_field":
        side.pop("counterpart_span_id")
    elif bad == "old_quote":
        side["source_quote"] = "Restart the device."
    else:
        key, replacement = {
            "unknown": ("source_span_id", "B999"),
            "range": ("source_span_id", "B001-B002"),
            "list": ("source_span_id", "B001 S001"),
            "wrong_side": ("counterpart_span_id", "B001"),
            "shared_source": ("source_span_id", "S001"),
            "inactive_witness": ("status", "COVERED"),
            "unreviewed": ("reviewed_unique_ids", ""),
            "fake_contradiction": ("counterpart_relation", "CONTRADICTS"),
        }[bad]
        side[key] = replacement
    with pytest.raises(DedupEvaluationError):
        subject.compile_selection(new, payload)


@pytest.mark.parametrize("bad", ["duplicate_id", "alignment", "too_long", "wrong_kind", "bool_offset"])
def test_every_inventory_span_is_verified_even_when_no_witness_selects_it(bad):
    _, value, payload = _case()
    span = payload["semantic_diff_evidence"]["spans"][-1]
    if bad == "duplicate_id":
        payload["semantic_diff_evidence"]["spans"].append(deepcopy(span))
    elif bad == "alignment":
        span["text"] = "Wrong text."
    elif bad == "too_long":
        span["text"] = "x" * 241
        payload["document_b"]["text"] += span["text"]
        span["start_char"] = len(payload["document_b"]["text"]) - 241
        span["end_char"] = len(payload["document_b"]["text"])
    elif bad == "wrong_kind":
        span["side"] = "A"
    else:
        span["start_char"] = True
    with pytest.raises(DedupEvaluationError, match="FOLLOWUP_BOUNDARY_SELECTION_PACKET"):
        subject.compile_selection(selection(value), payload)


def test_schema_removes_generated_quotes_and_excludes_exact_without_changing_v1():
    schema = subject.selection_schema()
    fields = set(schema["properties"]["a_meaning_in_b"]["properties"])
    assert schema["properties"]["a_meaning_in_b"] == schema["properties"]["b_meaning_in_a"]
    assert {"source_span_id", "counterpart_span_id"} <= fields
    assert not {"source_quote", "counterpart_quote", "source_ids", "counterpart_ids"} & fields
    assert "IDENTICAL_TEXT" not in schema["properties"]["record_scope"]["enum"]
    assert "IDENTICAL_TEXT" in subject.original.coverage_schema()["properties"]["record_scope"]["enum"]
