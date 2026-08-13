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

"""Construct and hash the exact blind judge-visible payload."""

from __future__ import annotations

import hashlib
from typing import Any

from eval.dedup.config import JudgeConfig
from eval.dedup.contracts import canonical_json_bytes
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.judging.long_document import prepare_long_document_evidence

EVIDENCE_ALIGNMENT_VERSION = "visible-evidence-align-v1"


def _neutral_metadata(document: dict[str, Any], text: str) -> dict[str, Any]:
    return {
        "url": document.get("url"),
        "crawl_timestamp": document.get("timestamp") or document.get("crawl_timestamp"),
        "language": document.get("language"),
        "character_count": len(text),
    }


def build_visible_payload(
    document_a: dict[str, Any],
    document_b: dict[str, Any],
    *,
    counter: TokenCounter,
    config: JudgeConfig,
) -> tuple[dict[str, Any], str]:
    evidence = prepare_long_document_evidence(
        document_a["text"],
        document_b["text"],
        counter=counter,
        config=config,
    )
    payload = {
        "payload_schema_version": "judge-visible-payload-v1",
        "document_a": {
            "metadata": _neutral_metadata(document_a, document_a["text"]),
            "text": evidence["text_a"],
        },
        "document_b": {
            "metadata": _neutral_metadata(document_b, document_b["text"]),
            "text": evidence["text_b"],
        },
        "long_document_evidence": {
            "truncated": evidence["truncated"],
            "token_counts": evidence["token_counts"],
            "windows": evidence["windows"],
        },
    }
    return payload, hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _locate_visible_quote(
    item: dict[str, Any],
    *,
    side: str,
    quote: str,
    payload: dict[str, Any],
) -> tuple[int, int] | None:
    document = payload["document_a" if side == "A" else "document_b"]
    full_text = document["text"]
    start = item.get("start_char")
    end = item.get("end_char")
    if full_text is not None:
        if (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start <= end <= len(full_text)
            and full_text[start:end] == quote
        ):
            return start, end
        position = full_text.find(quote)
        return (position, position + len(quote)) if position >= 0 else None

    windows = payload["long_document_evidence"]["windows"]
    for window in windows:
        if (
            window["side"] == side
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and window["start_char"] <= start <= end <= window["end_char"]
            and window["text"][start - window["start_char"] : end - window["start_char"]] == quote
        ):
            return start, end
    for window in windows:
        if window["side"] != side:
            continue
        position = window["text"].find(quote)
        if position >= 0:
            start = window["start_char"] + position
            return start, start + len(quote)
    return None


def align_evidence_offsets(
    value: Any,
    payload: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    """Align quoted evidence only against text visible in the blind payload."""

    if not isinstance(value, dict) or not isinstance(value.get("evidence"), list):
        return value, []
    repaired = []
    events = []
    for index, item in enumerate(value["evidence"]):
        if not isinstance(item, dict):
            events.append({"index": index, "action": "drop_unalignable"})
            continue
        side = item.get("side")
        quote = item.get("quote")
        if side not in {"A", "B"} or not isinstance(quote, str) or not quote or len(quote) > 240:
            events.append({"index": index, "action": "drop_unalignable"})
            continue
        location = _locate_visible_quote(item, side=side, quote=quote, payload=payload)
        if location is None:
            events.append({"index": index, "action": "drop_unalignable", "side": side})
            continue
        normalized = {
            "side": side,
            "start_char": location[0],
            "end_char": location[1],
            "quote": quote,
        }
        repaired.append(normalized)
        if normalized != item:
            events.append({"index": index, "action": "realign_offsets", "side": side})
    if repaired == value["evidence"]:
        return value, events
    return {**value, "evidence": repaired}, events


def validate_evidence_offsets(result: dict[str, Any], payload: dict[str, Any]) -> None:
    """Require every quoted span to match text that was actually visible to the judge."""

    from eval.dedup.validation import require

    windows = payload["long_document_evidence"]["windows"]
    for index, item in enumerate(result["evidence"]):
        side = item["side"]
        start = int(item["start_char"])
        end = int(item["end_char"])
        quote = item["quote"]
        document = payload["document_a" if side == "A" else "document_b"]
        full_text = document["text"]
        if full_text is not None:
            valid = 0 <= start <= end <= len(full_text) and full_text[start:end] == quote
        else:
            valid = any(
                window["side"] == side
                and window["start_char"] <= start <= end <= window["end_char"]
                and window["text"][start - window["start_char"] : end - window["start_char"]] == quote
                for window in windows
            )
        require(
            valid,
            "JUDGE_EVIDENCE_OFFSET_INVALID",
            "judge evidence does not match a visible text span",
            evidence_index=index,
            side=side,
        )


def assert_blind_payload(payload: dict[str, Any]) -> None:
    hidden_keys = {
        "predicted_group_id",
        "predicted_cluster_key",
        "keeper_doc_id",
        "removed_doc_id",
        "retriever_bitmask",
        "selection_probability",
        "track",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in hidden_keys:
                    msg = f"judge payload leaked hidden field: {key}"
                    raise ValueError(msg)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
