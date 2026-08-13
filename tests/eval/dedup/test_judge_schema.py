# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json

import pytest

from eval.dedup.judging.client import _json_mode_system_prompt
from eval.dedup.judging.schema import judge_output_schema, validate_judge_output
from eval.dedup.validation import DedupEvaluationError


def valid_output() -> dict:
    return {
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
        "relation_type": "EXACT",
        "material_difference": "NONE",
        "fuzzy_scope": "IN_SCOPE",
        "confidence": 0.99,
        "reason_codes": [],
        "evidence": [],
    }


def test_valid_judge_output() -> None:
    assert validate_judge_output(valid_output())["relation_type"] == "EXACT"


def test_judge_output_rejects_extra_field() -> None:
    value = {**valid_output(), "chain_of_thought": "hidden"}
    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value)
    assert error.value.issue.code == "JUDGE_SCHEMA_INVALID"
    assert error.value.issue.details["missing_fields"] == []
    assert error.value.issue.details["extra_fields"] == ["chain_of_thought"]


def test_judge_output_reports_missing_field() -> None:
    value = valid_output()
    del value["evidence"]
    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value)
    assert error.value.issue.code == "JUDGE_SCHEMA_INVALID"
    assert error.value.issue.details["missing_fields"] == ["evidence"]
    assert error.value.issue.details["extra_fields"] == []


def test_judge_output_rejects_consistency_conflict() -> None:
    value = {**valid_output(), "relation_type": "UNRELATED"}
    with pytest.raises(DedupEvaluationError) as error:
        validate_judge_output(value)
    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_json_mode_prompt_embeds_frozen_schema() -> None:
    marker = "Use each required key exactly once:\n"

    prompt = _json_mode_system_prompt("base prompt")

    assert json.loads(prompt.split(marker, 1)[1]) == judge_output_schema()
