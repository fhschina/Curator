# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import coverage_format_recovery as subject


def response(text=None, finish="length"):
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": text if text is not None else '{"explanation":"FAILED_MARKER' + "\t" * 5000},
            }
        ]
    }


def test_incomplete_tab_loop_qualifies_and_feedback_omits_failed_output():
    raw = response()
    finding = subject.detect(raw)
    assert finding["tail_tab_fraction"] == 1.0
    initial = {
        "messages": [{"role": "user", "content": "original input"}],
        "temperature": 0,
        "max_tokens": 4096,
        "response_format": {"original": True},
    }
    before = deepcopy(initial)
    repaired = subject.repair_request(initial)
    assert repaired["messages"][:-1] == initial["messages"]
    assert {k: v for k, v in repaired.items() if k != "messages"} == {
        k: v for k, v in initial.items() if k != "messages"
    }
    assert initial == before
    assert "FAILED_MARKER" not in str(repaired)


@pytest.mark.parametrize(
    "raw",
    [
        response(finish="stop"),
        response("{}" + "\t" * 5000),
        response("x" * 5000),
        response(" " * 5000),
        response("x\t" * 3000),
        response("short"),
        {},
        {"choices": []},
    ],
)
def test_complete_other_failures_and_non_tab_content_do_not_trigger(raw):
    assert subject.detect(raw) is None
