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

"""Adapt span-aligned dedup pairs without reinterpreting the main judgment."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

MAX_EVIDENCE_CHARS = 240
DECISION_OPTIONS = {
    "a_can_replace_b": ["yes", "no", "unresolved"],
    "b_can_replace_a": ["yes", "no", "unresolved"],
    "relation_type": [
        "exact",
        "canonical_exact",
        "near_surface",
        "containment",
        "version_related",
        "related_non_duplicate",
        "unrelated",
        "unresolved",
    ],
    "material_difference": ["none", "minor", "major", "unresolved"],
    "primary_material_difference": [
        "none",
        "main_content_addition_deletion",
        "entity_slot_change",
        "document_identity_change",
        "number_change",
        "date_time_change",
        "product_version_change",
        "result_set_change",
        "page_role_change",
        "legal_context_change",
        "negation_change",
        "code_literal_change",
        "code_output_change",
        "other_material",
        "unresolved",
    ],
    "dominant_overlap_source": [
        "main_content",
        "shared_page_template",
        "site_chrome",
        "cookie_consent",
        "legal_policy_template",
        "error_auth_paywall",
        "local_passage",
        "parser_artifact",
        "none",
        "unresolved",
    ],
    "primary_risk_factor": [
        "none",
        "boilerplate_dominated_similarity",
        "template_slot_collision",
        "identifier_underweighting",
        "topic_only_similarity",
        "list_snapshot_collision",
        "page_role_collision",
        "legal_context_collision",
        "long_document_local_overlap",
        "translation_equivalence",
        "paraphrase_equivalence",
        "containment_asymmetry",
        "extraction_or_payload_limit",
        "parser_artifact_dominance",
        "other",
    ],
    "confidence_tier": ["high", "medium", "low"],
}


def require(condition: bool, message: str, *, pair_id: object) -> None:
    if not condition:
        message = f"Critic pair {pair_id!r}: {message}"
        raise ValueError(message)


def validate_decision(decision: dict[str, str], *, pair_id: object) -> None:
    """Validate summary fields, without imposing legacy main-citation requirements."""
    require(set(decision) == set(DECISION_OPTIONS), "expected eight decision fields", pair_id=pair_id)
    for name, options in DECISION_OPTIONS.items():
        require(isinstance(decision[name], str) and decision[name] in options, f"invalid {name}", pair_id=pair_id)
    a, b = decision["a_can_replace_b"], decision["b_can_replace_a"]
    relation, material = decision["relation_type"], decision["material_difference"]
    primary = decision["primary_material_difference"]
    if "unresolved" in (a, b, relation):
        require(
            a == b == relation == material == primary == decision["dominant_overlap_source"] == "unresolved"
            and decision["primary_risk_factor"] == "extraction_or_payload_limit"
            and decision["confidence_tier"] == "low",
            "inconsistent unresolved main decision",
            pair_id=pair_id,
        )
        return
    require(
        "unresolved" not in (material, primary, decision["dominant_overlap_source"]),
        "resolved decision contains unresolved summary fields",
        pair_id=pair_id,
    )
    if relation in {"exact", "canonical_exact"}:
        consistent = a == b == "yes" and material == "none"
    elif relation == "near_surface":
        consistent = a == b == "yes" and material in {"none", "minor"}
    elif relation == "containment":
        consistent = {a, b} == {"yes", "no"} and material == "major" and primary == "main_content_addition_deletion"
    else:
        consistent = a == b == "no" and material == "major"
    require(consistent, "replacement directions, relation and materiality disagree", pair_id=pair_id)
    require(
        (material == "none") == (primary == "none"), "materiality and primary difference disagree", pair_id=pair_id
    )


def normalize_main(record: dict[str, Any], source_judge: str) -> dict[str, str]:
    pair_id = record.get("pair_id")
    raw = record.get(source_judge)
    require(isinstance(raw, dict), f"missing main result {source_judge!r}", pair_id=pair_id)
    decision = {}
    for name in DECISION_OPTIONS:
        item = raw.get(name)
        require(isinstance(item, dict), f"missing main score {name!r}", pair_id=pair_id)
        decision[name] = item.get("score")
    validate_decision(decision, pair_id=pair_id)
    return decision


def _offset(span: dict[str, Any], key: str, pair_id: object) -> int:
    value = span.get(key)
    # Arrow/Pandas roundtrips promote nullable integer members of span structs to floats.
    require(
        type(value) is int or (type(value) is float and value.is_integer()),
        f"missing or non-integral {key} for {span.get('span_id')}",
        pair_id=pair_id,
    )
    return int(value)


def _slice(span: dict[str, Any], document: str, prefix: str, pair_id: object) -> tuple[str, int, int]:
    start, end = (_offset(span, prefix + key, pair_id) for key in ("start_char", "end_char"))
    text = span.get(prefix + "text")
    require(
        type(start) is int
        and type(end) is int
        and 0 <= start < end <= len(document)
        and isinstance(text, str)
        and document[start:end] == text,
        f"invalid original offsets/text for {span.get('span_id')}",
        pair_id=pair_id,
    )
    return text, start, end


def adapt_alignment(record: dict[str, Any]) -> dict[str, Any]:
    """Copy canonical evidence while retaining padded windows as reading context."""
    pair_id = record.get("pair_id")
    require(pair_id is not None, "missing pair_id", pair_id=pair_id)
    documents = {side: record.get(f"text_{side.lower()}") for side in ("A", "B")}
    require(
        all(isinstance(text, str) for text in documents.values()), "text_a/text_b must be strings", pair_id=pair_id
    )
    packet = record.get("semantic_diff")
    require(isinstance(packet, dict), "missing semantic_diff", pair_id=pair_id)
    require(
        isinstance(packet.get("status"), str) and packet["status"] in {"COMPLETE", "INCOMPLETE_LIMIT"},
        "invalid packet status",
        pair_id=pair_id,
    )
    flags = [record.get("truncated"), *(packet.get(key) for key in ("truncated", "truncated_a", "truncated_b"))]
    require(all(type(flag) is bool for flag in flags), "truncation flags must be booleans", pair_id=pair_id)
    require(flags[0] == flags[1] == (flags[2] or flags[3]), "inconsistent truncation flags", pair_id=pair_id)
    spans = packet.get("spans")
    require(isinstance(spans, list), "missing span list", pair_id=pair_id)
    canonical, seen, counts = [], set(), Counter()
    for span in spans:
        require(isinstance(span, dict), "invalid span object", pair_id=pair_id)
        sid, kind = span.get("span_id"), span.get("kind")
        require(
            isinstance(sid, str) and re.fullmatch(r"[ABS]\d{3}", sid) is not None and sid not in seen,
            "span IDs must be valid and unique",
            pair_id=pair_id,
        )
        require(
            isinstance(kind, str) and kind in {"SHARED", "A_ONLY", "B_ONLY"},
            f"invalid kind for {sid}",
            pair_id=pair_id,
        )
        require(sid[0] == ("S" if kind == "SHARED" else kind[0]), f"ID/kind mismatch for {sid}", pair_id=pair_id)
        seen.add(sid)
        counts[kind] += 1
        item = dict(span)
        if kind == "SHARED":
            for side, document in documents.items():
                text, start, end = _slice(span, document, side.lower() + "_", pair_id)
                require(len(text) <= MAX_EVIDENCE_CHARS, f"oversized shared span {sid}", pair_id=pair_id)
                item.update({side.lower() + "_start_char": start, side.lower() + "_end_char": end})
        else:
            side, document = kind[0], documents[kind[0]]
            require(span.get("side", side) == side, f"wrong side for {sid}", pair_id=pair_id)
            context_text, context_start, context_end = _slice(span, document, "", pair_id)
            start, end = (_offset(span, key, pair_id) for key in ("delta_start_char", "delta_end_char"))
            require(
                context_start <= start < end <= context_end,
                f"missing or invalid delta boundaries for {sid}",
                pair_id=pair_id,
            )
            require(end - start <= MAX_EVIDENCE_CHARS, f"oversized delta span {sid}", pair_id=pair_id)
            item.update(
                side=side,
                start_char=start,
                end_char=end,
                text=document[start:end],
                context_text=context_text,
                context_start_char=context_start,
                context_end_char=context_end,
            )
        canonical.append(item)
    declared = packet.get("span_counts")
    require(isinstance(declared, dict), "missing span_counts", pair_id=pair_id)
    for kind in ("SHARED", "A_ONLY", "B_ONLY"):
        count = declared.get(kind)
        require(type(count) is int and count >= counts[kind], "invalid span_counts", pair_id=pair_id)
        if packet["status"] == "COMPLETE":
            require(count == counts[kind], "COMPLETE packet has missing spans", pair_id=pair_id)
    return {**packet, "spans": canonical}


def complete_text_equality(record: dict[str, Any], packet: dict[str, Any]) -> bool:
    return (
        packet["status"] == "COMPLETE"
        and packet["truncated"] is False
        and bool(record["text_a"].strip())
        and record["text_a"] == record["text_b"]
    )
