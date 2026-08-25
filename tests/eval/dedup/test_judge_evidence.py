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

from types import SimpleNamespace

import pytest

from eval.dedup.judging.payload import (
    VISIBLE_PAYLOAD_V0,
    VISIBLE_PAYLOAD_V1,
    align_evidence_offsets,
    assert_blind_payload,
    build_visible_payload,
    validate_evidence_offsets,
)
from eval.dedup.judging.schema import JUDGE_SCHEMA_V0, JUDGE_SCHEMA_V1
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


def test_evidence_offsets_must_match_visible_text() -> None:
    value = valid_output()
    value["evidence"] = [{"side": "A", "start_char": 0, "end_char": 5, "quote": "wrong"}]
    payload = {
        "document_a": {"text": "alpha"},
        "document_b": {"text": "beta"},
        "long_document_evidence": {"windows": []},
    }
    with pytest.raises(DedupEvaluationError) as error:
        validate_evidence_offsets(value, payload)
    assert error.value.issue.code == "JUDGE_EVIDENCE_OFFSET_INVALID"


def test_evidence_offsets_are_realigned_only_to_visible_text() -> None:
    value = valid_output()
    value["evidence"] = [{"side": "A", "start_char": 0, "end_char": 5, "quote": "alpha"}]
    payload = {
        "document_a": {"text": "prefix alpha suffix"},
        "document_b": {"text": "beta"},
        "long_document_evidence": {"windows": []},
    }

    aligned, events = align_evidence_offsets(value, payload)

    assert aligned["evidence"] == [{"side": "A", "start_char": 7, "end_char": 12, "quote": "alpha"}]
    assert events == [{"index": 0, "action": "realign_offsets", "side": "A"}]
    validate_evidence_offsets(aligned, payload)


def test_unalignable_evidence_is_dropped_without_changing_decision() -> None:
    value = valid_output()
    value["evidence"] = [{"side": "A", "start_char": 0, "end_char": 7, "quote": "missing"}]
    payload = {
        "document_a": {"text": None},
        "document_b": {"text": "beta"},
        "long_document_evidence": {
            "windows": [
                {"side": "A", "start_char": 100, "end_char": 112, "text": "visible text"},
            ]
        },
    }

    aligned, events = align_evidence_offsets(value, payload)

    assert aligned["same_duplicate_group"] == value["same_duplicate_group"]
    assert aligned["evidence"] == []
    assert events == [{"index": 0, "action": "drop_unalignable", "side": "A"}]


class _WhitespaceCounter:
    def count_many(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]


def _payload_config(schema_version: str) -> SimpleNamespace:
    return SimpleNamespace(
        schema_version=schema_version,
        max_visible_tokens=128,
        window_tokens=32,
        window_overlap_tokens=4,
    )


def test_v1_visible_payload_removes_all_document_metadata() -> None:
    document_a = {
        "text": "same visible body",
        "url": "https://metadata-a.invalid/private",
        "timestamp": "2026-08-24T00:00:00Z",
        "language": "secret-a",
    }
    document_b = {
        "text": "same visible body",
        "url": "https://metadata-b.invalid/private",
        "timestamp": "2025-01-01T00:00:00Z",
        "language": "secret-b",
    }

    payload, _ = build_visible_payload(
        document_a,
        document_b,
        counter=_WhitespaceCounter(),
        config=_payload_config(JUDGE_SCHEMA_V1),
    )

    assert payload["payload_schema_version"] == VISIBLE_PAYLOAD_V1
    assert payload["document_a"] == {"text": "same visible body"}
    assert payload["document_b"] == {"text": "same visible body"}
    assert payload["long_document_evidence"] == {"truncated": False, "windows": []}
    assert_blind_payload(payload)


def test_v0_visible_payload_contract_remains_metadata_preserving() -> None:
    document = {
        "text": "same visible body",
        "url": "https://example.invalid/doc",
        "timestamp": "2026-08-24T00:00:00Z",
        "language": "en",
    }

    payload, digest = build_visible_payload(
        document,
        document,
        counter=_WhitespaceCounter(),
        config=_payload_config(JUDGE_SCHEMA_V0),
    )

    assert payload["payload_schema_version"] == VISIBLE_PAYLOAD_V0
    assert payload["document_a"]["metadata"]["url"] == document["url"]
    assert payload["long_document_evidence"]["token_counts"] == {"A": 3, "B": 3}
    assert digest == "ae0fb9b3cae3545f8559ee8107db838b80084ddd01a3f2f81bbe130a64ce15e8"


def test_v1_blind_payload_guard_rejects_metadata_reintroduction() -> None:
    payload = {
        "payload_schema_version": VISIBLE_PAYLOAD_V1,
        "document_a": {"text": "alpha", "metadata": {"language": "en"}},
        "document_b": {"text": "alpha"},
        "long_document_evidence": {"truncated": False, "windows": []},
    }

    with pytest.raises(ValueError, match="permits only text"):
        assert_blind_payload(payload)
