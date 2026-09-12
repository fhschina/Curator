# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.analysis.local_context_diagnostic import (
    MAX_CONTEXT_CHARS,
    MAX_LINE_CHARS,
    message_renderer,
    presentation_record,
    token_preflight,
    validate_presentation_config,
)
from eval.dedup.analysis.presentation_diagnostic import message_renderer as historical_renderer
from eval.dedup.judging.payload import _semantic_diff_packet
from eval.dedup.validation import DedupEvaluationError, sha256_json


def _input(text_a="Continuing to browse means consent.", text_b="By browsing you consent.\nAccept"):
    payload = {
        "payload_schema_version": "judge-visible-payload-v3",
        "document_a": {"text": text_a},
        "document_b": {"text": text_b},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": _semantic_diff_packet(text_a, text_b, truncated=False),
    }
    return {
        "canonical_pair_id": "private-pair",
        "payload": payload,
        "judge_payload_hash": sha256_json(payload),
        "repair_feedback": None,
    }


def test_native_context_messages_preserve_system_rubric_and_every_original_quote():
    validate_presentation_config()
    row = _input()
    before = message_renderer("control")(row)
    assert before == historical_renderer("control")(row)
    after = message_renderer("ordered")(row)
    assert before[0] == after[0]
    old, new = before[1]["content"], after[1]["content"]
    assert old.split("</semantic_diff>")[0] == new.split("</semantic_diff>")[0]
    assert old.split("</semantic_diff>")[1] == new.split("</original_line_context>")[1]
    assert row["payload"]["document_a"]["text"] in new
    assert "<original_line_context>" in new


def test_multiple_unique_fragments_of_a_line_are_reconnected_once_with_existing_ids():
    row = _input("Keep browsing, means consent.", "By browsing means consent.\nAccept")
    original = deepcopy(row)
    record = presentation_record(row)
    context = record["local_context"]
    assert context.count("Keep browsing, means consent.") == 1
    assert context.count("By browsing means consent.") == 1
    assert "existing spans" in context
    assert "Accept" not in context
    assert row == original
    assert record["judge_payload_hash"] == sha256_json(row["payload"])


@pytest.mark.parametrize(
    ("text_a", "text_b"),
    [("Same paragraph.", "Same paragraph."), ("Same paragraph.", "Title\nSame paragraph."), ("Red", "Blue")],
)
def test_no_fragmented_line_means_exactly_unchanged_native_request(text_a, text_b):
    row = _input(text_a, text_b)
    assert presentation_record(row)["local_context"] == ""
    assert message_renderer("ordered")(row) == message_renderer("control")(row)


@pytest.mark.parametrize("failure", ["truncated", "incomplete", "missing_a", "missing_b"])
def test_incomplete_inputs_keep_original_prompt_without_new_context(failure):
    row = _input()
    if failure == "truncated":
        row["payload"]["long_document_evidence"]["truncated"] = True
    elif failure == "incomplete":
        row["payload"]["semantic_diff_evidence"]["status"] = "INCOMPLETE_LIMIT"
    else:
        row["payload"][f"document_{failure[-1]}"]["text"] = None
    assert presentation_record(row)["local_context_status"] == "UNAVAILABLE_INCOMPLETE"
    assert message_renderer("ordered")(row) == message_renderer("control")(row)


def test_long_line_is_not_clipped_and_original_complete_quotes_are_preserved():
    text = "prefix " + "word " * MAX_LINE_CHARS
    row = _input(text + "alpha", text + "beta")
    record = presentation_record(row)
    assert record["local_context"] == ""
    assert record["local_context_skipped_long_lines"] == 2
    assert message_renderer("ordered")(row) == message_renderer("control")(row)


def test_supplement_budget_omits_only_supplement_not_rows_or_original_evidence():
    lines_a = [f"Line {i} " + "detail " * 20 + "alpha." for i in range(30)]
    lines_b = [f"Line {i} " + "detail " * 20 + "beta." for i in range(30)]
    assert len("\n".join(lines_a + lines_b)) > MAX_CONTEXT_CHARS
    row = _input("\n".join(lines_a), "\n".join(lines_b))
    record = presentation_record(row)
    assert record["local_context_status"] == "UNAVAILABLE_SUPPLEMENT_BUDGET"
    assert record["local_context"] == ""
    assert message_renderer("ordered")(row) == message_renderer("control")(row)


def test_unicode_offsets_and_literal_source_are_not_rendered_as_template_instructions():
    row = _input("继续浏览,即同意。🙂", "浏览即同意。🙂\n接受")
    record = presentation_record(row)
    assert "A[0:10]" in record["local_context"]
    assert "继续浏览,即同意。🙂" in record["local_context"]
    row = _input("Literal {{ review_id }} stays data, yes.", "Literal {{ review_id }} stays data, no.")
    row["review_id"] = "hidden-review"
    row["human_same_duplicate_group"] = "hidden-label"
    row["main"] = "hidden-main"
    content = message_renderer("ordered")(row)[1]["content"]
    assert "{{ review_id }}" in content
    assert all(x not in content for x in ("hidden-review", "hidden-label", "hidden-main", "private-pair"))


def test_invalid_span_index_fails_before_supplement_generation():
    row = _input()
    shared = next(s for s in row["payload"]["semantic_diff_evidence"]["spans"] if s["kind"] == "SHARED")
    shared["a_text"] = "not in the input"
    with pytest.raises(DedupEvaluationError, match="PRESENTATION_SPAN_MISMATCH"):
        presentation_record(row)


def test_retry_feedback_does_not_mutate_visible_context():
    row = _input()
    before = presentation_record(row)
    row["repair_feedback"] = {"code": "LOCAL_NDD_OUTPUT_INVALID", "details": {}, "message": "missing evidence"}
    after = presentation_record(row)
    assert before["local_context"] == after["local_context"]
    content = message_renderer("ordered")(row)[1]["content"]
    assert "<repair_retry>" in content
    assert "missing evidence" in content


def test_preflight_measures_both_actual_native_messages_and_never_truncates_inputs():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return list(range(200))

    row = _input()
    before = deepcopy(row)
    result = token_preflight([row], Tokenizer(), context_budget=400, output_budget=100, safety_margin=50)
    assert len(calls) == len(result["rows"]) == 2
    assert "record_binding_verdict" in calls[0][0][1]["content"]
    assert row == before
    with pytest.raises(DedupEvaluationError, match="PRESENTATION_CONTEXT_BUDGET"):
        token_preflight([row], Tokenizer(), context_budget=300, output_budget=100, safety_margin=50)
