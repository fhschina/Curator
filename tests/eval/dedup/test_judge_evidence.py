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

import pytest

from eval.dedup.judging.payload import align_evidence_offsets, validate_evidence_offsets
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
