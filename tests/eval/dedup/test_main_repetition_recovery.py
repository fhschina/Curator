# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
from copy import deepcopy

import pytest

from eval.dedup.judging import main_repetition_recovery as subject
from tests.eval.dedup.test_exp1_reproduction_runtime import response


def repeated_response(pattern="item_alpha, item_beta, "):
    result = response('{"reasoning":"' + pattern * 4096)
    result["choices"][0]["finish_reason"] = "length"
    return result


@pytest.mark.parametrize("pattern", ["item_alpha, ", "item_alpha, item_beta, ", "重复说明 ", "word ", "x"])
def test_detects_long_periodic_incomplete_answers_without_domain_specific_strings(pattern):
    raw = repeated_response(pattern)
    saved = deepcopy(raw)
    found = subject.detect(raw)
    assert found is not None
    assert found["period_agreement"] == 1
    assert found["tail_characters"] == 4096
    assert found["period_characters"] <= len(pattern)
    assert raw == saved
    assert "item_alpha" not in json.dumps(found)


@pytest.mark.parametrize(
    "kind",
    ["stop", "content_filter", "short", "whitespace", "nonrepetitive", "complete_json", "fenced_json", "two_choices"],
)
def test_nonqualifying_outputs_preserve_native_handling(kind):
    raw = repeated_response()
    choice = raw["choices"][0]
    if kind in {"stop", "content_filter"}:
        choice["finish_reason"] = kind
    elif kind == "short":
        choice["message"]["content"] = '{"reasoning":"' + "x" * 100
    elif kind == "whitespace":
        choice["message"]["content"] = '{"reasoning":"' + " " * 8192
    elif kind == "nonrepetitive":
        choice["message"]["content"] = '{"reasoning":"' + "".join(
            hashlib.sha256(str(i).encode()).hexdigest() for i in range(100)
        )
    elif kind in {"complete_json", "fenced_json"}:
        text = json.dumps({"reasoning": "item_alpha, " * 4096})
        choice["message"]["content"] = text if kind == "complete_json" else "```json\n" + text + "\n```"
    else:
        raw["choices"].append(deepcopy(choice))
    assert subject.detect(raw) is None


@pytest.mark.parametrize("raw", [None, {}, {"choices": None}, {"choices": []}, {"choices": [None]}, {"choices": [{}]}])
def test_malformed_envelopes_are_left_to_original_validation(raw):
    assert subject.detect(raw) is None


def test_feedback_preserves_request_and_contains_no_failed_output_or_decision_hint():
    initial = {
        "model": "original",
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": "original rubric"},
            {"role": "user", "content": [{"type": "text", "text": "original documents"}]},
        ],
    }
    saved = deepcopy(initial)
    fixed = subject.repair_request(initial)
    assert initial == saved
    assert fixed["messages"][:-1] == initial["messages"]
    assert {k: v for k, v in fixed.items() if k != "messages"} == {k: v for k, v in initial.items() if k != "messages"}
    assert fixed["messages"][-1] == {"role": "user", "content": [{"type": "text", "text": subject.FEEDBACK}]}
    assert len(subject.FEEDBACK) < 1000
    assert all(word not in subject.FEEDBACK for word in ("P01329", "YES", "NO", "CONTAINMENT", "__utmk"))
