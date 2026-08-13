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
from types import SimpleNamespace

from eval.dedup.config import JudgeConfig
from eval.dedup.judging.run import _judge_one


def _valid_response() -> str:
    return json.dumps(
        {
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
    )


class FlakyClient:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.prompts: list[str] = []

    def judge(self, *, system_prompt: str, payload: dict) -> str:
        del payload
        self.prompts.append(system_prompt)
        self.calls += 1
        return "{" if self.calls <= self.failures else _valid_response()


class SchemaMismatchClient:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.prompts: list[str] = []

    def judge(self, *, system_prompt: str, payload: dict) -> str:
        del payload
        self.prompts.append(system_prompt)
        self.calls += 1
        value = json.loads(_valid_response())
        if self.calls <= self.failures:
            del value["evidence"]
            value["chain_of_thought"] = "must not be persisted"
        return json.dumps(value)


def _inputs() -> tuple[dict, dict, SimpleNamespace]:
    row = {
        "evaluation_run_id": "fixture-run",
        "sut_run_id": "fixture-sut",
        "canonical_pair_id": "cp1_fixture",
        "canonical_pair_id_version": "cp1",
        "judge_payload_hash": "payload-hash",
    }
    payload = {
        "document_a": {"text": "alpha"},
        "document_b": {"text": "alpha"},
        "long_document_evidence": {"windows": []},
    }
    config = SimpleNamespace(
        judge=JudgeConfig(
            backend="stub",
            base_url="http://fixture.invalid/v1",
            model="fixture-judge",
            api_key_env="UNUSED",
            structured_output_mode="json_schema",
            thinking=False,
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=1024,
            concurrency=2,
            requests_per_minute=60,
            timeout_seconds=1.0,
            max_retries=2,
            max_visible_tokens=128,
            window_tokens=32,
            window_overlap_tokens=4,
            prompt_version="fixture-prompt",
            schema_version="dedup-judge-output-v0",
        )
    )
    return row, payload, config


def test_judge_retries_schema_failure_then_succeeds() -> None:
    row, payload, config = _inputs()
    client = FlakyClient(failures=1)
    result = _judge_one(row, client=client, prompt="fixture", payload=payload, config=config)
    assert result["record_type"] == "result"
    assert result["attempts"] == 2
    assert result["retried"] is True
    assert "REPAIR RETRY" in client.prompts[1]
    assert len(result["provider_response_sha256"]) == 64
    assert result["deterministic_repair_events"] == []


def test_judge_retry_receives_safe_structured_field_feedback() -> None:
    row, payload, config = _inputs()
    client = SchemaMismatchClient(failures=1)
    result = _judge_one(row, client=client, prompt="fixture", payload=payload, config=config)
    assert result["record_type"] == "result"
    assert result["attempts"] == 2
    assert '"missing_fields":["evidence"]' in client.prompts[1]
    assert '"extra_fields":["chain_of_thought"]' in client.prompts[1]
    assert "must not be persisted" not in client.prompts[1]


def test_judge_preserves_terminal_error_after_retry_budget() -> None:
    row, payload, config = _inputs()
    client = SchemaMismatchClient(failures=3)
    result = _judge_one(row, client=client, prompt="fixture", payload=payload, config=config)
    assert result["record_type"] == "error"
    assert result["attempts"] == 3
    assert len(result["errors"]) == 3
    for error in result["errors"]:
        assert error["validation_issue"]["code"] == "JUDGE_SCHEMA_INVALID"
        assert error["validation_issue"]["details"]["missing_fields"] == ["evidence"]
        assert error["validation_issue"]["details"]["extra_fields"] == ["chain_of_thought"]
        assert "must not be persisted" not in json.dumps(error)
