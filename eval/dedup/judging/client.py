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

"""Isolated model-provider interface for blind pair judging."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Protocol

from eval.dedup.config import JudgeConfig
from eval.dedup.contracts import DuplicateAnswer, FuzzyScope, MaterialDifference, RelationType
from eval.dedup.judging.schema import judge_output_schema, validate_judge_output
from eval.dedup.validation import DedupEvaluationError, require


class JudgeClient(Protocol):
    def judge(self, *, system_prompt: str, payload: dict[str, Any]) -> str: ...


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval
        if delay:
            time.sleep(delay)


def _json_mode_system_prompt(system_prompt: str) -> str:
    schema = json.dumps(judge_output_schema(), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return (
        f"{system_prompt}\n\nOUTPUT CONTRACT (non-negotiable): emit only one compact JSON object. "
        "Use exactly these keys once: same_duplicate_group, a_can_replace_b, b_can_replace_a, "
        "relation_type, material_difference, fuzzy_scope, confidence, reason_codes, evidence. "
        "Never emit prose, Markdown, duplicate keys, or text before or after the JSON object. "
        "Use only enum values from the supplied schema. If uncertain, use UNRESOLVED values and "
        "INSUFFICIENT_EVIDENCE. Set evidence to [] unless every quote and character offset is exact "
        "in the visible text. Return exactly one JSON object satisfying this JSON Schema. "
        f"Use each required key exactly once:\n{schema}"
    )


class StubJudgeClient:
    """Deterministic fixture backend that exercises the complete judge contract."""

    def judge(self, *, system_prompt: str, payload: dict[str, Any]) -> str:
        del system_prompt
        text_a = payload["document_a"]["text"]
        text_b = payload["document_b"]["text"]
        if text_a is None or text_b is None:
            answer = {
                "same_duplicate_group": DuplicateAnswer.UNRESOLVED,
                "a_can_replace_b": DuplicateAnswer.UNRESOLVED,
                "b_can_replace_a": DuplicateAnswer.UNRESOLVED,
                "relation_type": RelationType.UNRESOLVED,
                "material_difference": MaterialDifference.UNRESOLVED,
                "fuzzy_scope": FuzzyScope.UNRESOLVED,
                "confidence": 0.2,
                "reason_codes": ["INSUFFICIENT_EVIDENCE"],
                "evidence": [],
            }
        else:
            normalized_a = " ".join(text_a.split())
            normalized_b = " ".join(text_b.split())
            same = normalized_a == normalized_b
            containment = normalized_a in normalized_b or normalized_b in normalized_a
            answer = {
                "same_duplicate_group": DuplicateAnswer.YES if same else DuplicateAnswer.NO,
                "a_can_replace_b": DuplicateAnswer.YES if same else DuplicateAnswer.NO,
                "b_can_replace_a": DuplicateAnswer.YES if same else DuplicateAnswer.NO,
                "relation_type": RelationType.EXACT
                if same
                else RelationType.CONTAINMENT
                if containment
                else RelationType.UNRELATED,
                "material_difference": MaterialDifference.NONE if same else MaterialDifference.MAJOR,
                "fuzzy_scope": FuzzyScope.IN_SCOPE if same or containment else FuzzyScope.OUT_OF_SCOPE,
                "confidence": 0.99,
                "reason_codes": [] if same else ["INSERTION_DELETION"] if containment else ["TOPIC_ONLY"],
                "evidence": [],
            }
        validate_judge_output(answer)
        return json.dumps(answer, separators=(",", ":"))


class NvidiaOpenAIJudgeClient:
    def __init__(self, config: JudgeConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            msg = "openai is required for the NVIDIA Inference Hub judge backend"
            raise RuntimeError(msg) from exc
        credential = os.environ.get(config.api_key_env, "").strip()
        require(
            credential,
            "MISSING_API_KEY",
            "judge API credential environment variable is empty",
            variable=config.api_key_env,
        )
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=credential, timeout=config.timeout_seconds)
        self.limiter = RateLimiter(config.requests_per_minute)

    def judge(self, *, system_prompt: str, payload: dict[str, Any]) -> str:
        self.limiter.wait()
        response_format: dict[str, Any]
        request_system_prompt = system_prompt
        if self.config.structured_output_mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "dedup_evaluation_v0", "strict": True, "schema": judge_output_schema()},
            }
        elif self.config.structured_output_mode == "json_object_plus_local_schema":
            response_format = {"type": "json_object"}
            request_system_prompt = _json_mode_system_prompt(system_prompt)
        else:
            raise DedupEvaluationError(
                "UNSUPPORTED_STRUCTURED_OUTPUT_MODE",
                "judge structured output mode must be frozen by preflight",
                mode=self.config.structured_output_mode,
            )
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": request_system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_output_tokens,
            response_format=response_format,
            extra_body={"chat_template_kwargs": {"thinking": self.config.thinking}},
            stream=False,
        )
        content = response.choices[0].message.content
        require(isinstance(content, str) and content, "EMPTY_JUDGE_RESPONSE", "judge returned no message content")
        return content


def create_judge_client(config: JudgeConfig) -> JudgeClient:
    if config.backend == "stub":
        return StubJudgeClient()
    if config.backend == "nvidia_openai":
        return NvidiaOpenAIJudgeClient(config)
    raise DedupEvaluationError("UNSUPPORTED_JUDGE_BACKEND", "judge backend is not supported", backend=config.backend)
