# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.analysis.presentation_diagnostic import (
    diagnostic_checks,
    message_renderer,
    presentation_record,
    token_preflight,
    validate_presentation_config,
)
from eval.dedup.judging.payload import _semantic_diff_packet
from eval.dedup.validation import DedupEvaluationError, sha256_json


def _input():
    text_a = "Continued browsing means consent."
    text_b = "Wishlist\n\n" + text_a + "\nAccept"
    payload = {
        "payload_schema_version": "judge-visible-payload-v3",
        "document_a": {"text": text_a},
        "document_b": {"text": text_b},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": _semantic_diff_packet(text_a, text_b, truncated=False),
    }
    return {
        "canonical_pair_id": "private-pair-id",
        "payload": payload,
        "judge_payload_hash": sha256_json(payload),
        "repair_feedback": None,
    }


def test_native_ndd_messages_change_only_visible_presentation_not_policy_or_rubric():
    validate_presentation_config()
    row = _input()
    before = message_renderer("control")(row)
    after = message_renderer("ordered")(row)
    assert before[0] == after[0]
    assert before[1]["content"].split("<semantic_diff")[0] == after[1]["content"].split("<semantic_diff")[0]
    assert before[1]["content"].split("</semantic_diff>")[1] == after[1]["content"].split("</semantic_diff>")[1]
    assert row["payload"]["document_a"]["text"] in after[1]["content"]
    assert row["payload"]["document_b"]["text"] in after[1]["content"]
    for span in row["payload"]["semantic_diff_evidence"]["spans"]:
        assert span["span_id"] in after[1]["content"]
        if span["kind"] == "SHARED":
            assert f"A[{span['a_start_char']}:{span['a_end_char']}]" in after[1]["content"]
        else:
            assert f"{span['side']}[{span['start_char']}:{span['end_char']}]" in after[1]["content"]


def test_short_index_preview_does_not_remove_the_middle_of_the_original_text():
    row = _input()
    text = "Anchor " + "x" * 80 + " MIDDLE_FACT " + "y" * 80 + " end."
    row["payload"]["document_a"]["text"] = row["payload"]["document_b"]["text"] = text
    row["payload"]["semantic_diff_evidence"] = _semantic_diff_packet(text, text, truncated=False)
    content = message_renderer("ordered")(row)[1]["content"]
    full, index = content.split("</original_order_visible_text>")
    assert full.count(text) == 2
    assert "MIDDLE_FACT" not in index
    assert "(...)" in index
    assert "[S001 SHARED]" in index


def test_index_coordinates_are_unicode_characters_and_leave_blind_payload_unchanged():
    row = _input()
    text_a, text_b = "继续浏览即同意。🙂", "收藏\n继续浏览即同意。🙂\n接受"
    row["payload"]["document_a"]["text"] = text_a
    row["payload"]["document_b"]["text"] = text_b
    row["payload"]["semantic_diff_evidence"] = _semantic_diff_packet(text_a, text_b, truncated=False)
    row["judge_payload_hash"] = sha256_json(row["payload"])
    original = deepcopy(row)
    record = presentation_record(row)
    assert record["presentation_full_available"] is True
    assert "继续浏览即同意。🙂" in record["presentation_index"]
    assert row == original
    assert record["payload"] == original["payload"]
    assert record["judge_payload_hash"] == sha256_json(original["payload"])


@pytest.mark.parametrize(("field", "value"), [("a_start_char", -1), ("a_end_char", 9999), ("a_text", "wrong")])
def test_index_rejects_span_coordinates_or_quotes_not_matching_original_text(field, value):
    row = _input()
    shared = next(s for s in row["payload"]["semantic_diff_evidence"]["spans"] if s["kind"] == "SHARED")
    shared[field] = value
    with pytest.raises(DedupEvaluationError, match="PRESENTATION_SPAN_MISMATCH"):
        presentation_record(row)


def test_native_renderer_preserves_text_as_data_without_exposing_reference_or_pair_metadata():
    row = _input()
    row["review_id"] = "hidden-review"
    row["human_same_duplicate_group"] = "hidden-label"
    row["main"] = "hidden-main"
    row["payload"]["document_a"]["text"] += "\nLiteral {{ review_id }} and <tag> remain source text."
    rendered = message_renderer("ordered")(row)
    content = rendered[1]["content"]
    assert "Literal {{ review_id }} and <tag> remain source text." in content
    for hidden in ("hidden-review", "hidden-label", "hidden-main", "private-pair-id"):
        assert hidden not in content


@pytest.mark.parametrize("failure", ["truncated", "incomplete", "missing_a", "missing_b"])
def test_incomplete_packets_do_not_gain_fabricated_full_text(failure):
    row = _input()
    if failure == "truncated":
        row["payload"]["long_document_evidence"]["truncated"] = True
    elif failure == "incomplete":
        row["payload"]["semantic_diff_evidence"]["status"] = "INCOMPLETE_LIMIT"
    else:
        row["payload"][f"document_{failure[-1]}"]["text"] = None
    content = message_renderer("ordered")(row)[1]["content"]
    assert '<original_order_visible_text status="UNAVAILABLE_INCOMPLETE"/>' in content
    assert "<document_a>" not in content
    assert "[S001 SHARED]" in content


def test_retry_feedback_remains_safe_structured_feedback_with_same_evidence():
    row = _input()
    row["repair_feedback"] = {
        "code": "LOCAL_NDD_OUTPUT_INVALID",
        "details": {},
        "message": "missing bilateral evidence",
    }
    content = message_renderer("ordered")(row)[1]["content"]
    assert "<repair_retry>" in content
    assert "missing bilateral evidence" in content
    assert row["payload"]["document_b"]["text"] in content


def test_token_preflight_counts_native_schema_and_never_truncates_or_drops_rows():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return list(range(150))

    rows = [_input(), {**_input(), "canonical_pair_id": "second"}]
    original = deepcopy(rows)
    result = token_preflight(rows, Tokenizer(), context_budget=500, output_budget=100, safety_margin=50)
    assert len(calls) == len(result["rows"]) == 4
    assert all(
        call[1] == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False} for call in calls
    )
    assert any("record_binding_verdict" in m["content"] for m in calls[0][0])
    assert rows == original
    with pytest.raises(DedupEvaluationError, match="PRESENTATION_CONTEXT_BUDGET"):
        token_preflight(rows, Tokenizer(), context_budget=200, output_budget=100, safety_margin=50)


@pytest.mark.parametrize("failure", ["guard", "target", "primary", "over_group"])
def test_target_improvement_cannot_hide_a_negative_or_aggregate_regression(failure):
    control = {
        "local_gates": {"passed": True},
        "metrics": {"weighted": {"primary_decision_exact": 0.75}, "over_group": 25},
    }
    candidate = deepcopy(control)
    targets = {"passed": True}
    assert all(diagnostic_checks(candidate, control, targets).values())
    if failure == "guard":
        candidate["local_gates"]["passed"] = False
    elif failure == "target":
        targets["passed"] = False
    elif failure == "primary":
        candidate["metrics"]["weighted"]["primary_decision_exact"] = 0.74
    else:
        candidate["metrics"]["over_group"] = 26
    assert not all(diagnostic_checks(candidate, control, targets).values())
