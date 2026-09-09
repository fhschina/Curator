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

import difflib
import hashlib
import re
import unicodedata
from typing import Any

from eval.dedup.config import AnyJudgeConfig
from eval.dedup.contracts import canonical_json_bytes
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.judging.long_document import prepare_long_document_evidence
from eval.dedup.judging.schema import JUDGE_SCHEMA_V0

EVIDENCE_ALIGNMENT_VERSION = "visible-evidence-align-v1"
VISIBLE_PAYLOAD_V0 = "judge-visible-payload-v1"
VISIBLE_PAYLOAD_V1 = "judge-visible-payload-v2"
VISIBLE_PAYLOAD_V2 = "judge-visible-payload-v3"
SEMANTIC_DIFF_VERSION = "visible-semantic-diff-v1"

_DIFF_TOKEN_PATTERN = re.compile(r"\w{1,120}|[^\w\s]", re.UNICODE)
_MAX_DIFF_SPAN_CHARS = 220
_MAX_DIFF_SPANS_PER_KIND = 160


def _normalized_diff_tokens(text: str) -> list[tuple[int, int, str]]:
    return [
        (match.start(), match.end(), unicodedata.normalize("NFKC", match.group()).casefold())
        for match in _DIFF_TOKEN_PATTERN.finditer(text)
    ]


def _chunk_token_range(
    tokens: list[tuple[int, int, str]], start: int, end: int, *, paired: list[tuple[int, int, str]] | None = None
) -> list[tuple[int, int]]:
    chunks = []
    cursor = start
    while cursor < end:
        chunk_start = cursor
        while cursor < end:
            span_length = tokens[cursor][1] - tokens[chunk_start][0]
            paired_length = 0
            if paired is not None:
                paired_length = paired[cursor - start][1] - paired[chunk_start - start][0]
            if cursor > chunk_start and max(span_length, paired_length) > _MAX_DIFF_SPAN_CHARS:
                break
            cursor += 1
        chunks.append((chunk_start, cursor))
    return chunks


def _semantic_diff_packet(text_a: str | None, text_b: str | None, *, truncated: bool) -> dict[str, Any]:
    if truncated or text_a is None or text_b is None:
        return {
            "contract_version": SEMANTIC_DIFF_VERSION,
            "status": "UNAVAILABLE_TRUNCATED",
            "tokenization": "nfkc-casefold-visible-token-v1",
            "span_counts": {"SHARED": 0, "A_ONLY": 0, "B_ONLY": 0},
            "spans": [],
        }

    tokens_a = _normalized_diff_tokens(text_a)
    tokens_b = _normalized_diff_tokens(text_b)
    normalized_a = [token[2] for token in tokens_a]
    normalized_b = [token[2] for token in tokens_b]
    raw_spans: dict[str, list[dict[str, Any]]] = {"SHARED": [], "A_ONLY": [], "B_ONLY": []}

    def add_side(kind: str, side: str, tokens: list[tuple[int, int, str]], start: int, end: int) -> None:
        text = text_a if side == "A" else text_b
        for chunk_start, chunk_end in _chunk_token_range(tokens, start, end):
            start_char = tokens[chunk_start][0]
            end_char = tokens[chunk_end - 1][1]
            raw_spans[kind].append(
                {
                    "kind": kind,
                    "side": side,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": text[start_char:end_char],
                }
            )

    matcher = difflib.SequenceMatcher(a=normalized_a, b=normalized_b, autojunk=False)
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            paired_tokens = tokens_b[b_start:b_end]
            for chunk_start, chunk_end in _chunk_token_range(
                tokens_a, a_start, a_end, paired=paired_tokens
            ):
                paired_start = b_start + chunk_start - a_start
                paired_end = paired_start + chunk_end - chunk_start
                a_start_char = tokens_a[chunk_start][0]
                a_end_char = tokens_a[chunk_end - 1][1]
                b_start_char = tokens_b[paired_start][0]
                b_end_char = tokens_b[paired_end - 1][1]
                raw_spans["SHARED"].append(
                    {
                        "kind": "SHARED",
                        "a_start_char": a_start_char,
                        "a_end_char": a_end_char,
                        "a_text": text_a[a_start_char:a_end_char],
                        "b_start_char": b_start_char,
                        "b_end_char": b_end_char,
                        "b_text": text_b[b_start_char:b_end_char],
                    }
                )
        else:
            if tag in {"delete", "replace"}:
                add_side("A_ONLY", "A", tokens_a, a_start, a_end)
            if tag in {"insert", "replace"}:
                add_side("B_ONLY", "B", tokens_b, b_start, b_end)

    counts = {kind: len(items) for kind, items in raw_spans.items()}
    complete = all(count <= _MAX_DIFF_SPANS_PER_KIND for count in counts.values())
    spans = []
    prefixes = {"SHARED": "S", "A_ONLY": "A", "B_ONLY": "B"}
    for kind in ("SHARED", "A_ONLY", "B_ONLY"):
        for index, item in enumerate(raw_spans[kind][:_MAX_DIFF_SPANS_PER_KIND], start=1):
            spans.append({"span_id": f"{prefixes[kind]}{index:03d}", **item})
    return {
        "contract_version": SEMANTIC_DIFF_VERSION,
        "status": "COMPLETE" if complete else "INCOMPLETE_LIMIT",
        "tokenization": "nfkc-casefold-visible-token-v1",
        "span_counts": counts,
        "spans": spans,
    }


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
    config: AnyJudgeConfig,
) -> tuple[dict[str, Any], str]:
    evidence = prepare_long_document_evidence(
        document_a["text"],
        document_b["text"],
        counter=counter,
        config=config,
    )
    payload_version = getattr(
        config,
        "visible_payload_version",
        VISIBLE_PAYLOAD_V0 if config.schema_version == JUDGE_SCHEMA_V0 else VISIBLE_PAYLOAD_V1,
    )
    if payload_version == VISIBLE_PAYLOAD_V0:
        payload = {
            "payload_schema_version": VISIBLE_PAYLOAD_V0,
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
    elif payload_version == VISIBLE_PAYLOAD_V1:
        payload = {
            "payload_schema_version": VISIBLE_PAYLOAD_V1,
            "document_a": {"text": evidence["text_a"]},
            "document_b": {"text": evidence["text_b"]},
            "long_document_evidence": {
                "truncated": evidence["truncated"],
                "windows": evidence["windows"],
            },
        }
    elif payload_version == VISIBLE_PAYLOAD_V2:
        payload = {
            "payload_schema_version": VISIBLE_PAYLOAD_V2,
            "document_a": {"text": evidence["text_a"]},
            "document_b": {"text": evidence["text_b"]},
            "long_document_evidence": {
                "truncated": evidence["truncated"],
                "windows": evidence["windows"],
            },
            "semantic_diff_evidence": _semantic_diff_packet(
                evidence["text_a"], evidence["text_b"], truncated=evidence["truncated"]
            ),
        }
    else:
        msg = f"unsupported visible payload version: {payload_version}"
        raise ValueError(msg)
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

    if payload.get("payload_schema_version") in {VISIBLE_PAYLOAD_V1, VISIBLE_PAYLOAD_V2}:
        for side in ("document_a", "document_b"):
            document = payload.get(side)
            if not isinstance(document, dict) or set(document) != {"text"}:
                msg = f"{payload['payload_schema_version']} permits only text in {side}"
                raise ValueError(msg)
        long_evidence = payload.get("long_document_evidence")
        if not isinstance(long_evidence, dict) or set(long_evidence) != {"truncated", "windows"}:
            msg = f"{payload['payload_schema_version']} leaked non-evidence fields"
            raise ValueError(msg)
    if payload.get("payload_schema_version") == VISIBLE_PAYLOAD_V2:
        packet = payload.get("semantic_diff_evidence")
        if (
            not isinstance(packet, dict)
            or set(packet) != {"contract_version", "status", "tokenization", "span_counts", "spans"}
            or packet.get("contract_version") != SEMANTIC_DIFF_VERSION
        ):
            msg = f"{VISIBLE_PAYLOAD_V2} requires the deterministic semantic-diff packet"
            raise ValueError(msg)
        status = packet["status"]
        spans = packet["spans"]
        counts = packet["span_counts"]
        if (
            status not in {"COMPLETE", "INCOMPLETE_LIMIT", "UNAVAILABLE_TRUNCATED"}
            or not isinstance(spans, list)
            or not isinstance(counts, dict)
            or set(counts) != {"SHARED", "A_ONLY", "B_ONLY"}
        ):
            msg = f"{VISIBLE_PAYLOAD_V2} has an invalid semantic-diff state"
            raise ValueError(msg)
        expected_ids = []
        for kind, prefix in (("SHARED", "S"), ("A_ONLY", "A"), ("B_ONLY", "B")):
            expected_ids.extend(
                f"{prefix}{index:03d}"
                for index in range(1, min(int(counts[kind]), _MAX_DIFF_SPANS_PER_KIND) + 1)
            )
        if [span.get("span_id") for span in spans if isinstance(span, dict)] != expected_ids:
            msg = f"{VISIBLE_PAYLOAD_V2} semantic-diff IDs are not stable and contiguous"
            raise ValueError(msg)
        for span in spans:
            kind = span.get("kind")
            if kind == "SHARED":
                for side, key in (("A", "a"), ("B", "b")):
                    text = payload[f"document_{key}"]["text"]
                    start = span.get(f"{key}_start_char")
                    end = span.get(f"{key}_end_char")
                    if not isinstance(text, str) or text[start:end] != span.get(f"{key}_text"):
                        msg = f"{VISIBLE_PAYLOAD_V2} shared span does not match visible side {side}"
                        raise ValueError(msg)
            elif kind in {"A_ONLY", "B_ONLY"}:
                side = span.get("side")
                key = str(side).lower()
                text = payload.get(f"document_{key}", {}).get("text")
                start = span.get("start_char")
                end = span.get("end_char")
                if side not in {"A", "B"} or not isinstance(text, str) or text[start:end] != span.get("text"):
                    msg = f"{VISIBLE_PAYLOAD_V2} side-only span does not match visible text"
                    raise ValueError(msg)
            else:
                msg = f"{VISIBLE_PAYLOAD_V2} semantic-diff span kind is invalid"
                raise ValueError(msg)
