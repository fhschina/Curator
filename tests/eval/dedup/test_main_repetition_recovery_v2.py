# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

from eval.dedup.judging import main_repetition_recovery_v2 as subject
from tests.eval.dedup.test_main_repetition_recovery import repeated_response


def test_only_recovery_feedback_changes_while_detection_and_initial_request_stay_fixed():
    initial = {
        "model": "original",
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [{"role": "system", "content": "fixed rubric"}, {"role": "user", "content": "fixed input"}],
    }
    saved = deepcopy(initial)
    old = subject.previous.repair_request(initial)
    new = subject.repair_request(initial)
    assert initial == saved
    assert new["messages"][:-1] == old["messages"][:-1] == initial["messages"]
    assert new["messages"][-1] != old["messages"][-1]
    assert {k: v for k, v in new.items() if k != "messages"} == {k: v for k, v in old.items() if k != "messages"}
    loop = repeated_response()
    assert subject.detect(loop) == subject.previous.detect(loop)
    loop["choices"][0]["finish_reason"] = "stop"
    assert subject.detect(loop) is subject.previous.detect(loop) is None
