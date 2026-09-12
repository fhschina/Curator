# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_schema_probe as subject


def raw(value, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(value)}}]}


@pytest.mark.parametrize("mode", subject.MODES[1:])
def test_probe_schema_is_not_disclosed_in_prompt_and_ignored_parameter_is_not_success(mode):
    body = subject.request_body(mode, "PRIVATE_SCHEMA_SENTINEL")
    assert "PRIVATE_SCHEMA_SENTINEL" not in json.dumps(body["messages"])
    assert (
        subject.classify(mode, "PRIVATE_SCHEMA_SENTINEL", raw({"probe": "PROMPT_VALUE"})) == "CONSTRAINT_NOT_OBSERVED"
    )
    assert (
        subject.classify(mode, "PRIVATE_SCHEMA_SENTINEL", raw({"probe": "PRIVATE_SCHEMA_SENTINEL"}))
        == "SCHEMA_CONSTRAINT_OBSERVED"
    )
    assert (
        subject.classify(mode, "PRIVATE_SCHEMA_SENTINEL", raw({"probe": "PRIVATE_SCHEMA_SENTINEL", "extra": True}))
        == "CONSTRAINT_NOT_OBSERVED"
    )


def test_json_object_control_and_incomplete_or_duplicate_key_outputs():
    assert subject.classify("json_object", "S", raw({"probe": "PROMPT_VALUE"})) == "BASELINE_PROMPT_FOLLOWED"
    assert subject.classify("guided_json", "S", raw({"probe": "S"}, "length")) == "INCOMPLETE_OUTPUT"
    duplicate = {"choices": [{"finish_reason": "stop", "message": {"content": '{"probe":"S","probe":"S"}'}}]}
    assert subject.classify("guided_json", "S", duplicate) == "INVALID_OUTPUT"
