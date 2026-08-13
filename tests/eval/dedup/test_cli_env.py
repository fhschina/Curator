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

import json
import os
from pathlib import Path

from eval.dedup.cli import _load_repository_env, _probe_judge


def test_repository_env_loads_silently_without_overriding_process_env(tmp_path: Path, monkeypatch, capsys) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "DEDUP_TEST_FROM_DOTENV=dotenv-value\nDEDUP_TEST_PROCESS_WINS=dotenv-value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEDUP_TEST_FROM_DOTENV", raising=False)
    monkeypatch.setenv("DEDUP_TEST_PROCESS_WINS", "process-value")

    _load_repository_env(dotenv_path)

    assert os.environ["DEDUP_TEST_FROM_DOTENV"] == "dotenv-value"
    assert os.environ["DEDUP_TEST_PROCESS_WINS"] == "process-value"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


class _ProbeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def judge(self, *, system_prompt: str, payload: dict) -> str:
        del system_prompt, payload
        return next(self.responses)


def test_preflight_probe_uses_frozen_retry_budget() -> None:
    valid = json.dumps(
        {
            "same_duplicate_group": "YES",
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "EXACT",
            "material_difference": "NONE",
            "fuzzy_scope": "IN_SCOPE",
            "confidence": 1.0,
            "reason_codes": [],
            "evidence": [],
        }
    )
    payload = {
        "document_a": {"text": "alpha"},
        "document_b": {"text": "alpha"},
        "long_document_evidence": {"windows": []},
    }

    attempts = _probe_judge(
        _ProbeClient(['{"a_can_replace_b":"alpha"}', valid]),
        prompt="fixture",
        payload=payload,
        max_retries=2,
    )

    assert attempts == 2
