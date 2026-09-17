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

"""Adapt Data Designer score objects to the selected Judge contract."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from eval.dedup.core.validation import DedupEvaluationError, require
from eval.dedup.judging.boundary_critic import arbitrate_boundary_review, parse_boundary_review
from eval.dedup.judging.output_schema import JUDGE_SCHEMA, validate_judge_output
from eval.dedup.judging.payload import align_evidence_offsets
from eval.dedup.judging.record_scope import (
    arbitrate_record_scope,
    arbitrate_record_scope_v8,
    arbitrate_record_scope_v9,
    complete_visible_equality,
    parse_record_scope,
)
from eval.dedup.judging.retained_conflict import arbitrate_retained_conflict, parse_retained_conflict
from eval.dedup.judging.scoped_critic import arbitrate_scoped_review, parse_scoped_review

JUDGE_COLUMN = "qwen_minhash_fuzzy_dedup_judge"
SEMANTIC_JUDGE_COLUMN = "qwen_dedup_semantic_judge"
RECORD_BINDING_CRITIC_COLUMN = "qwen_dedup_record_binding_critic"

_REASON_FIELDS = {
    "reason_number_change": "NUMBER_CHANGE",
    "reason_date_time_change": "DATE_TIME_CHANGE",
    "reason_product_version_change": "PRODUCT_VERSION_CHANGE",
    "reason_url_change": "URL_CHANGE",
    "reason_named_entity_change": "NAMED_ENTITY_CHANGE",
    "reason_negation_change": "NEGATION_CHANGE",
    "reason_code_literal_change": "CODE_LITERAL_CHANGE",
    "reason_code_output_change": "CODE_OUTPUT_CHANGE",
    "reason_insertion_deletion": "INSERTION_DELETION",
    "reason_boilerplate": "BOILERPLATE",
    "reason_parser_noise": "PARSER_NOISE",
    "reason_language_mismatch": "LANGUAGE_MISMATCH",
    "reason_topic_only": "TOPIC_ONLY",
    "reason_insufficient_evidence": "INSUFFICIENT_EVIDENCE",
    "reason_other_material": "OTHER_MATERIAL",
}
_V0_FIELDS = (
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
    "fuzzy_scope",
)
_V2_MODEL_FIELDS = (
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
)
_V3_MODEL_FIELDS = (
    *_V2_MODEL_FIELDS,
    "primary_material_difference",
    "dominant_overlap_source",
    "primary_risk_factor",
    "confidence_tier",
)
_V3_SEMANTIC_LEDGER_V1_FIELDS = (
    "content_profile_a",
    "content_profile_b",
    "shared_content_basis",
    "hard_conflict",
    "decisive_difference_location",
)
_V3_SEMANTIC_LEDGER_V2_FIELDS = (
    *_V3_SEMANTIC_LEDGER_V1_FIELDS,
    "record_alignment",
    "non_main_difference",
    "translation_status",
)
_V3_SEMANTIC_LEDGER_V3_FIELDS = (
    *_V3_SEMANTIC_LEDGER_V2_FIELDS,
    "record_identity_support",
    "overlap_scope",
    "surface_delta_type",
)
_V3_SEMANTIC_LEDGER_V4_FIELDS = (
    *_V3_SEMANTIC_LEDGER_V3_FIELDS,
    "boundary_delta_class",
)
_V3_SPAN_LEDGER_FIELDS = (
    "span_content_profile_a",
    "span_content_profile_b",
    "span_shared_basis",
    "span_a_delta",
    "span_b_delta",
    "span_hard_conflict",
    "span_translation_status",
)
_V3_SPAN_LEDGER_OPTIONS = {
    "span_content_profile_a": {"SUBSTANTIVE_MAIN", "NON_MAIN_ONLY", "UNREADABLE"},
    "span_content_profile_b": {"SUBSTANTIVE_MAIN", "NON_MAIN_ONLY", "UNREADABLE"},
    "span_shared_basis": {
        "VERIFIED_SUBSTANTIVE_RECORD",
        "VERIFIED_EQUIVALENT_NON_MAIN_MESSAGE",
        "NONE",
        "UNRESOLVED",
    },
    "span_a_delta": {
        "NONE",
        "SEMANTICALLY_COVERED",
        "UNIVERSAL_UI_OR_REPETITION",
        "SAME_RECORD_CONTENT_EXTENSION",
        "RECORD_IDENTITY_ROLE_OR_STATE_CHANGE",
        "MATERIAL_NON_MAIN_MESSAGE_CHANGE",
        "OTHER_SUBSTANTIVE_CONTENT",
        "UNRESOLVED",
    },
    "span_b_delta": {
        "NONE",
        "SEMANTICALLY_COVERED",
        "UNIVERSAL_UI_OR_REPETITION",
        "SAME_RECORD_CONTENT_EXTENSION",
        "RECORD_IDENTITY_ROLE_OR_STATE_CHANGE",
        "MATERIAL_NON_MAIN_MESSAGE_CHANGE",
        "OTHER_SUBSTANTIVE_CONTENT",
        "UNRESOLVED",
    },
    "span_hard_conflict": {
        "NONE",
        "IDENTITY_OR_SLOT",
        "STATE_OR_VERSION",
        "PAGE_ROLE",
        "LIST_MEMBERSHIP",
        "LEGAL_CONTEXT",
        "OTHER_MATERIAL",
        "UNRESOLVED",
    },
    "span_translation_status": {
        "COMPLETE_FAITHFUL",
        "PARTIAL_OR_ADDITIVE",
        "NOT_TRANSLATION",
        "UNRESOLVED",
    },
}
_RECORD_BINDING_CRITIC_OPTIONS = {
    "ATOMIC_SAME_RECORD_EXTENSION",
    "BENIGN_NON_RECORD_DELTA",
    "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
    "TWO_SIDED_OR_CONFLICTING",
    "NON_MAIN_POLICY_OR_STATE_CHANGE",
    "NOT_APPLICABLE",
    "UNRESOLVED",
}
_V3_SEMANTIC_LEDGER_OPTIONS = {
    "content_profile_a": {"SUBSTANTIVE_MAIN", "NON_MAIN_ONLY", "UNREADABLE"},
    "content_profile_b": {"SUBSTANTIVE_MAIN", "NON_MAIN_ONLY", "UNREADABLE"},
    "shared_content_basis": {
        "SUBSTANTIVE_ANCHOR",
        "EQUIVALENT_NON_MAIN_MESSAGE",
        "NONE",
        "UNRESOLVED",
    },
    "hard_conflict": {
        "NONE",
        "IDENTITY_OR_SLOT",
        "STATE_OR_VERSION",
        "PAGE_ROLE",
        "LIST_MEMBERSHIP",
        "LEGAL_CONTEXT",
        "OTHER_MATERIAL",
        "UNRESOLVED",
    },
    "decisive_difference_location": {
        "NONE",
        "NON_MAIN_ONLY",
        "A_ONLY_MAIN_ADDITION",
        "B_ONLY_MAIN_ADDITION",
        "TWO_SIDED_MAIN_DIVERGENCE",
        "UNRESOLVED",
    },
    "record_alignment": {
        "SAME_SUBSTANTIVE_RECORD",
        "SAME_NON_MAIN_MESSAGE",
        "GENERIC_OR_TEMPLATE_OVERLAP_ONLY",
        "DIFFERENT_RECORD_OR_ROLE",
        "UNRESOLVED",
    },
    "non_main_difference": {
        "NONE_OR_IGNORABLE_CHROME",
        "MATERIAL_MESSAGE_OR_STATE",
        "NOT_APPLICABLE",
        "UNRESOLVED",
    },
    "translation_status": {
        "COMPLETE_FAITHFUL",
        "PARTIAL_OR_ADDITIVE",
        "NOT_TRANSLATION",
        "UNRESOLVED",
    },
    "record_identity_support": {
        "VERBATIM_SHARED_IDENTIFIER",
        "DISTINCTIVE_MULTI_FACT_IDENTITY",
        "SAME_NON_MAIN_MESSAGE",
        "NOT_PROVEN",
        "DIFFERENT_OR_CONFLICTING",
        "UNRESOLVED",
    },
    "overlap_scope": {
        "DOCUMENT_WIDE_SAME_RECORD",
        "COMPLETE_NON_MAIN_MESSAGE",
        "LOCAL_PASSAGE_ONLY",
        "GENERIC_FAMILY_OR_TEMPLATE_ONLY",
        "NONE",
        "UNRESOLVED",
    },
    "surface_delta_type": {
        "NONE_OR_FORMATTING",
        "NAV_LABEL_BYLINE_ONLY",
        "REDUNDANT_REPETITION_ONLY",
        "MATERIAL_PROPOSITION_OR_RECORD",
        "NOT_APPLICABLE",
        "UNRESOLVED",
    },
    "boundary_delta_class": {
        "NONE_OR_FORMATTING",
        "UNIVERSAL_UI_OR_REDUNDANT_REPETITION",
        "SAME_RECORD_CONTENT_EXTENSION",
        "RECORD_IDENTITY_ROLE_OR_STATE_CHANGE",
        "MATERIAL_NON_MAIN_MESSAGE_CHANGE",
        "TWO_SIDED_CONTENT_CHANGE",
        "UNRESOLVED",
    },
}
_LABELED_QUOTE_PATTERN = re.compile(
    r"\b([AB]):\s*(?:\"([^\"]{1,2000})\"|“([^”]{1,2000})”|'([^']{1,2000})'|"
    r"\u2018([^\u2019]{1,2000})\u2019|«([^»]{1,2000})»)"
)
_UNLABELED_QUOTE_PATTERNS = (
    re.compile(r'"([^"\n]{2,240})"'),
    re.compile(r"“([^”\n]{2,240})”"),
    re.compile(r"(?<!\w)'([^'\n]{2,240})'(?!\w)"),
    re.compile(r"\u2018([^\u2019\n]{2,240})\u2019"),
)
_ELLIPSIS_PATTERN = re.compile(r"(?:\.{3,}|\u2026)")
_PARENTHETICAL_ANNOTATION_PATTERN = re.compile(r"\(([^()\n]{1,24})\)")
_SPAN_ID_PATTERN = re.compile(r"\b[ABS]\d{3}\b")

_CONFIDENCE_VALUES = {
    "0.2": 0.2,
    "0.5": 0.5,
    "0.75": 0.75,
    "0.9": 0.9,
    "0.98": 0.98,
}


def _score(judge: dict[str, Any], name: str) -> Any:
    value = judge.get(name)
    require(
        isinstance(value, dict) and set(value) >= {"score", "reasoning"},
        "LOCAL_NDD_OUTPUT_INVALID",
        "NDD rubric result is missing score or reasoning",
        field=name,
    )
    return value["score"]


def _visible_fragments(payload: dict[str, Any], side: str) -> list[str]:
    document = payload["document_a" if side == "A" else "document_b"]
    text = document.get("text")
    if isinstance(text, str):
        return [text]
    return [
        str(window["text"])
        for window in payload["long_document_evidence"]["windows"]
        if window.get("side") == side and isinstance(window.get("text"), str)
    ]


def _fallback_visible_quote(payload: dict[str, Any], side: str) -> str | None:
    other = "\n".join(_visible_fragments(payload, "B" if side == "A" else "A"))
    candidates: list[str] = []
    for fragment in _visible_fragments(payload, side):
        for line in fragment.splitlines():
            quote = line.strip()
            if len(quote) > 180:
                quote = quote[:180].rstrip()
            if quote:
                candidates.append(quote)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda quote: (
            quote not in other,
            min(sum(character.isalnum() for character in quote), 120),
            min(len(quote), 160),
        ),
    )


def _explicit_quote_variants(quote: str) -> list[tuple[str, bool, bool, bool]]:
    """Return literal model-provided spans, including substantial ellipsis-delimited excerpts."""

    variants = [(quote, False, False, False)] if len(quote) <= 240 else [(quote[:240].rstrip(), False, False, True)]
    if _ELLIPSIS_PATTERN.search(quote) is not None:
        for raw_excerpt in _ELLIPSIS_PATTERN.split(quote):
            excerpt = raw_excerpt.strip(" \t\r\n.,;:!?\u2026-\u2013\u2014")
            if len(excerpt) >= 6 and sum(character.isalnum() for character in excerpt) >= 5:
                variants.append((excerpt[:240].rstrip(), True, False, len(excerpt) > 240))
    punctuation_excerpt = quote.rstrip(".,;:!?\u3002\uff01\uff1f\uff1b\uff1a")
    if (
        punctuation_excerpt != quote
        and len(punctuation_excerpt) >= 12
        and sum(character.isalnum() for character in punctuation_excerpt) >= 8
    ):
        variants.append((punctuation_excerpt, False, True, False))
    return variants


def _is_short_latin_annotation(value: str) -> bool:
    return any("LATIN" in unicodedata.name(character, "") for character in value) and all(
        character.isalpha()
        or character.isspace()
        or unicodedata.category(character).startswith("M")
        or character in "'-"
        for character in value
    )


def _quote_match_key(value: str) -> tuple[str, list[int]]:
    skipped: set[int] = set()
    for match in _PARENTHETICAL_ANNOTATION_PATTERN.finditer(value):
        if _is_short_latin_annotation(match.group(1)):
            skipped.update(range(match.start(), match.end()))
    for match in re.finditer(r"\\[nrt]", value):
        skipped.update(range(match.start(), match.end()))
    characters: list[str] = []
    source_indices: list[int] = []
    for index, character in enumerate(value):
        if index in skipped or character.isspace():
            continue
        for normalized in unicodedata.normalize("NFKD", character).casefold():
            if unicodedata.category(normalized).startswith("M"):
                continue
            characters.append(normalized)
            source_indices.append(index)
    return "".join(characters), source_indices


def _minimum_quote_key_length(value: str) -> int:
    if any("LATIN" in unicodedata.name(character, "") or character.isdigit() for character in value):
        return 8
    return 3


def _snap_quote_to_visible_format(payload: dict[str, Any], side: str, quote: str) -> str | None:
    """Recover exact visible bytes when extraction inserted only presentation annotations."""

    quote_key, _ = _quote_match_key(quote)
    if len(quote_key) < _minimum_quote_key_length(quote_key):
        return None
    for fragment in _visible_fragments(payload, side):
        fragment_key, source_indices = _quote_match_key(fragment)
        start = fragment_key.find(quote_key)
        if start < 0:
            continue
        snapped = fragment[source_indices[start] : source_indices[start + len(quote_key) - 1] + 1].strip()
        if snapped and len(snapped) <= 240:
            return snapped
    return None


def _snap_longest_quote_prefix(payload: dict[str, Any], side: str, quote: str) -> str | None:
    """Return the longest substantial exact-normalized prefix of an explicitly labeled quote."""

    quote_key, _ = _quote_match_key(quote)
    minimum = max(_minimum_quote_key_length(quote_key), min(24, len(quote_key) // 3))
    if len(quote_key) < minimum:
        return None
    best: tuple[int, str, list[int], int] | None = None
    for fragment in _visible_fragments(payload, side):
        fragment_key, source_indices = _quote_match_key(fragment)
        low, high = minimum, len(quote_key)
        best_length = 0
        best_start = -1
        while low <= high:
            middle = (low + high) // 2
            start = fragment_key.find(quote_key[:middle])
            if start >= 0:
                best_length = middle
                best_start = start
                low = middle + 1
            else:
                high = middle - 1
        if best_length and (best is None or best_length > best[0]):
            best = (best_length, fragment, source_indices, best_start)
    if best is None:
        return None
    length, fragment, source_indices, start = best
    snapped = fragment[source_indices[start] : source_indices[start + length - 1] + 1].strip()
    if not snapped or len(snapped) > 240 or sum(character.isalnum() for character in snapped) < 8:
        return None
    return snapped


def _extract_quote_evidence(
    value: dict[str, Any], payload: dict[str, Any] | None, *, allow_fallback: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if payload is None:
        return [], []
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    ellipsis_excerpts: set[tuple[str, str]] = set()
    punctuation_excerpts: set[tuple[str, str]] = set()
    prefix_excerpts: set[tuple[str, str]] = set()
    format_snapped: set[tuple[str, str]] = set()

    def add_candidate(side: str, quote: str, *, allow_prefix_excerpt: bool) -> None:
        for variant, is_ellipsis_excerpt, is_punctuation_excerpt, is_prefix_excerpt in _explicit_quote_variants(quote):
            key = (side, variant)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"side": side, "start_char": 0, "end_char": len(variant), "quote": variant})
            if is_ellipsis_excerpt:
                ellipsis_excerpts.add(key)
            if is_punctuation_excerpt:
                punctuation_excerpts.add(key)
            if is_prefix_excerpt:
                prefix_excerpts.add(key)
            snapped = _snap_quote_to_visible_format(payload, side, variant)
            snapped_key = (side, snapped) if snapped is not None else None
            if snapped is None and allow_prefix_excerpt:
                snapped = _snap_longest_quote_prefix(payload, side, variant)
                if snapped is not None:
                    snapped_key = (side, snapped)
                    prefix_excerpts.add(snapped_key)
            if snapped is None or snapped == variant or snapped_key in seen:
                continue
            seen.add(snapped_key)
            candidates.append({"side": side, "start_char": 0, "end_char": len(snapped), "quote": snapped})
            format_snapped.add(snapped_key)

    reasonings = [
        item["reasoning"]
        for item in value.values()
        if isinstance(item, dict) and isinstance(item.get("reasoning"), str)
    ]
    for reasoning in reasonings:
        for match in _LABELED_QUOTE_PATTERN.finditer(reasoning):
            side = match.group(1)
            quote = next(group for group in match.groups()[1:] if group is not None)
            add_candidate(side, quote, allow_prefix_excerpt=True)
    for reasoning in reasonings:
        for pattern in _UNLABELED_QUOTE_PATTERNS:
            for match in pattern.finditer(reasoning):
                quote = match.group(1)
                for side in ("A", "B"):
                    add_candidate(side, quote, allow_prefix_excerpt=False)
    aligned, _ = align_evidence_offsets({"evidence": candidates}, payload)
    by_side = {side: next((item for item in aligned["evidence"] if item["side"] == side), None) for side in ("A", "B")}
    fallback_events = []
    for side in ("A", "B"):
        if by_side[side] is not None or not allow_fallback:
            continue
        quote = _fallback_visible_quote(payload, side)
        if quote is None:
            continue
        fallback, _ = align_evidence_offsets(
            {"evidence": [{"side": side, "start_char": 0, "end_char": len(quote), "quote": quote}]},
            payload,
        )
        if fallback["evidence"]:
            by_side[side] = fallback["evidence"][0]
            fallback_events.append({"action": "add_visible_excerpt_evidence", "side": side})
    selected = [item for side in ("A", "B") if (item := by_side[side]) is not None]
    selected_keys = {(item["side"], item["quote"]) for item in selected}
    selected.extend(item for item in aligned["evidence"] if (item["side"], item["quote"]) not in selected_keys)
    selected = selected[:4]
    ellipsis_events = [
        {"action": "align_explicit_ellipsis_excerpt", "side": item["side"]}
        for item in selected
        if (item["side"], item["quote"]) in ellipsis_excerpts
    ]
    punctuation_events = [
        {"action": "align_explicit_punctuation_excerpt", "side": item["side"]}
        for item in selected
        if (item["side"], item["quote"]) in punctuation_excerpts
    ]
    prefix_events = [
        {"action": "align_explicit_prefix_excerpt", "side": item["side"]}
        for item in selected
        if (item["side"], item["quote"]) in prefix_excerpts
    ]
    format_events = [
        {"action": "snap_model_quote_to_visible_format", "side": item["side"]}
        for item in selected
        if (item["side"], item["quote"]) in format_snapped
    ]
    return selected, [*fallback_events, *ellipsis_events, *punctuation_events, *prefix_events, *format_events]


def _derive_same_duplicate_group(a_to_b: str, b_to_a: str) -> str:
    if "YES" in {a_to_b, b_to_a}:
        return "YES"
    if a_to_b == b_to_a == "NO":
        return "NO"
    return "UNRESOLVED"


def _optional_v3_semantic_ledger(value: dict[str, Any]) -> dict[str, str] | None:
    all_fields = set(_V3_SEMANTIC_LEDGER_V4_FIELDS)
    present = {field for field in all_fields if field in value}
    if not present:
        return None
    v3_only = set(_V3_SEMANTIC_LEDGER_V3_FIELDS) - set(_V3_SEMANTIC_LEDGER_V2_FIELDS)
    v2_only = set(_V3_SEMANTIC_LEDGER_V2_FIELDS) - set(_V3_SEMANTIC_LEDGER_V1_FIELDS)
    v4_only = set(_V3_SEMANTIC_LEDGER_V4_FIELDS) - set(_V3_SEMANTIC_LEDGER_V3_FIELDS)
    if present & v4_only:
        expected = _V3_SEMANTIC_LEDGER_V4_FIELDS
    elif present & v3_only:
        expected = _V3_SEMANTIC_LEDGER_V3_FIELDS
    elif present & v2_only:
        expected = _V3_SEMANTIC_LEDGER_V2_FIELDS
    else:
        expected = _V3_SEMANTIC_LEDGER_V1_FIELDS
    require(
        present == set(expected),
        "LOCAL_NDD_OUTPUT_INVALID",
        "V3 semantic ledger must provide every gate field together",
        missing_fields=sorted(set(expected) - present),
    )
    ledger = {}
    for field in expected:
        score = _score(value, field)
        require(
            isinstance(score, str) and score.upper() in _V3_SEMANTIC_LEDGER_OPTIONS[field],
            "LOCAL_NDD_OUTPUT_INVALID",
            "semantic-ledger score is invalid",
            field=field,
        )
        ledger[field] = score.upper()
    return ledger


def _optional_v3_span_ledger(  # noqa: PLR0911 - each citation contract fails closed independently
    value: dict[str, Any], payload: dict[str, Any] | None
) -> tuple[dict[str, str], list[str], str | None] | None:
    present = {field for field in _V3_SPAN_LEDGER_FIELDS if field in value}
    if not present:
        return None
    require(
        present == set(_V3_SPAN_LEDGER_FIELDS),
        "LOCAL_NDD_OUTPUT_INVALID",
        "V3 span ledger must provide every field together",
        missing_fields=sorted(set(_V3_SPAN_LEDGER_FIELDS) - present),
    )
    ledger = {}
    citations: list[str] = []
    for field in _V3_SPAN_LEDGER_FIELDS:
        score = _score(value, field)
        require(
            isinstance(score, str) and score.upper() in _V3_SPAN_LEDGER_OPTIONS[field],
            "LOCAL_NDD_OUTPUT_INVALID",
            "span-ledger score is invalid",
            field=field,
        )
        ledger[field] = score.upper()
        reasoning = value[field]["reasoning"]
        if isinstance(reasoning, str):
            citations.extend(_SPAN_ID_PATTERN.findall(reasoning.upper()))

    packet = payload.get("semantic_diff_evidence") if isinstance(payload, dict) else None
    if not isinstance(packet, dict):
        return ledger, list(dict.fromkeys(citations)), "MISSING_SEMANTIC_DIFF_PACKET"
    known_ids = {
        str(span.get("span_id")) for span in packet.get("spans", []) if isinstance(span, dict) and span.get("span_id")
    }
    ordered_citations = list(dict.fromkeys(citations))
    if any(span_id not in known_ids for span_id in ordered_citations):
        return ledger, ordered_citations, "UNKNOWN_SPAN_CITATION"
    counts = packet.get("span_counts", {})
    for side in ("A", "B"):
        count = counts.get(f"{side}_ONLY") if isinstance(counts, dict) else None
        delta = ledger[f"span_{side.lower()}_delta"]
        if count == 0 and delta != "NONE":
            return ledger, ordered_citations, f"{side}_DELTA_WITHOUT_VISIBLE_SIDE_SPANS"
        if isinstance(count, int) and count > 0 and delta == "NONE":
            return ledger, ordered_citations, f"{side}_SIDE_SPANS_CLASSIFIED_AS_NONE"

    def has(prefix: str) -> bool:
        return any(span_id.startswith(prefix) for span_id in ordered_citations)

    basis = ledger["span_shared_basis"]
    if basis.startswith("VERIFIED_") and not (has("S") or (has("A") and has("B"))):
        return ledger, ordered_citations, "SHARED_BASIS_WITHOUT_BILATERAL_SPAN_CITATION"
    for side in ("a", "b"):
        delta = ledger[f"span_{side}_delta"]
        if delta not in {"NONE", "UNRESOLVED"} and not has(side.upper()):
            return ledger, ordered_citations, f"{side.upper()}_DELTA_WITHOUT_SIDE_SPAN_CITATION"
    if ledger["span_hard_conflict"] not in {"NONE", "UNRESOLVED"} and not (has("A") and has("B")):
        return ledger, ordered_citations, "HARD_CONFLICT_WITHOUT_BILATERAL_SPAN_CITATION"
    if ledger["span_translation_status"] == "COMPLETE_FAITHFUL" and not (has("A") and has("B")):
        return ledger, ordered_citations, "TRANSLATION_WITHOUT_BILATERAL_SPAN_CITATION"
    exact_text = payload.get("document_a", {}).get("text") == payload.get("document_b", {}).get("text")
    if "UNRESOLVED" not in set(ledger.values()) and not exact_text and not (has("S") or (has("A") and has("B"))):
        return ledger, ordered_citations, "RESOLVED_LEDGER_WITHOUT_BILATERAL_SPAN_CITATION"
    return ledger, ordered_citations, None


def _optional_record_binding_critic(  # noqa: PLR0911 - each critic contract failure is explicit
    value: dict[str, Any] | None, payload: dict[str, Any] | None
) -> tuple[str, list[str], str | None] | None:
    if value is None:
        return None
    require(
        isinstance(value, dict),
        "LOCAL_NDD_OUTPUT_INVALID",
        "record-binding critic result must be an object",
    )
    verdict = _score(value, "record_binding_verdict")
    require(
        isinstance(verdict, str) and verdict.upper() in _RECORD_BINDING_CRITIC_OPTIONS,
        "LOCAL_NDD_OUTPUT_INVALID",
        "record-binding critic verdict is invalid",
        field="record_binding_verdict",
    )
    normalized = verdict.upper()
    reasoning = value["record_binding_verdict"].get("reasoning")
    citations = list(dict.fromkeys(_SPAN_ID_PATTERN.findall(reasoning.upper()))) if isinstance(reasoning, str) else []
    packet = payload.get("semantic_diff_evidence") if isinstance(payload, dict) else None
    if not isinstance(packet, dict):
        return normalized, citations, "CRITIC_MISSING_SEMANTIC_DIFF_PACKET"
    if packet.get("status") != "COMPLETE":
        return normalized, citations, f"CRITIC_SEMANTIC_DIFF_{packet.get('status') or 'UNAVAILABLE'}"
    known_ids = {
        str(span.get("span_id")) for span in packet.get("spans", []) if isinstance(span, dict) and span.get("span_id")
    }
    if any(span_id not in known_ids for span_id in citations):
        return normalized, citations, "CRITIC_UNKNOWN_SPAN_CITATION"
    if normalized in {"UNRESOLVED", "NOT_APPLICABLE"}:
        return normalized, citations, None
    has_bilateral_basis = any(span_id.startswith("S") for span_id in citations) or (
        any(span_id.startswith("A") for span_id in citations) and any(span_id.startswith("B") for span_id in citations)
    )
    if not has_bilateral_basis:
        return normalized, citations, "CRITIC_VERDICT_WITHOUT_BILATERAL_BASIS"
    if normalized in {
        "ATOMIC_SAME_RECORD_EXTENSION",
        "BENIGN_NON_RECORD_DELTA",
        "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
        "TWO_SIDED_OR_CONFLICTING",
        "NON_MAIN_POLICY_OR_STATE_CHANGE",
    } and not any(span_id.startswith(("A", "B")) for span_id in citations):
        return normalized, citations, "CRITIC_VERDICT_WITHOUT_UNIQUE_SPAN"
    return normalized, citations, None


def _unresolved_span_target() -> dict[str, str]:
    return {
        "a_can_replace_b": "UNRESOLVED",
        "b_can_replace_a": "UNRESOLVED",
        "relation_type": "UNRESOLVED",
        "material_difference": "UNRESOLVED",
        "primary_material_difference": "UNRESOLVED",
        "dominant_overlap_source": "UNRESOLVED",
        "primary_risk_factor": "EXTRACTION_OR_PAYLOAD_LIMIT",
        "confidence_tier": "LOW",
    }


def _apply_record_binding_critic(
    replacements: dict[str, str],
    ledger: dict[str, str],
    critic: tuple[str, list[str], str | None] | None,
) -> tuple[dict[str, str], str | None]:
    if critic is None or replacements.get("relation_type") != "CONTAINMENT":
        return replacements, None
    verdict, citations, issue = critic
    if issue is not None or verdict in {"UNRESOLVED", "NOT_APPLICABLE"}:
        return _unresolved_span_target(), issue or f"CRITIC_{verdict}"
    if verdict == "ATOMIC_SAME_RECORD_EXTENSION":
        extension_prefix = "A" if ledger["span_a_delta"] == "SAME_RECORD_CONTENT_EXTENSION" else "B"
        if not any(span_id.startswith(extension_prefix) for span_id in citations):
            return _unresolved_span_target(), "CRITIC_EXTENSION_SIDE_WITHOUT_CITATION"
        return replacements, "CRITIC_CONFIRMED_ATOMIC_EXTENSION"
    if verdict == "BENIGN_NON_RECORD_DELTA":
        return (
            {
                "a_can_replace_b": "YES",
                "b_can_replace_a": "YES",
                "relation_type": "NEAR_SURFACE",
                "material_difference": "MINOR",
                "primary_material_difference": "OTHER_MATERIAL",
            },
            "CRITIC_RECLASSIFIED_BENIGN_DELTA",
        )
    if verdict in {"SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT", "TWO_SIDED_OR_CONFLICTING"}:
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": "RELATED_NON_DUPLICATE",
                "material_difference": "MAJOR",
                "primary_material_difference": (
                    "PAGE_ROLE_CHANGE" if verdict == "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT" else "OTHER_MATERIAL"
                ),
            },
            f"CRITIC_VETO_{verdict}",
        )
    message = f"unhandled record-binding critic verdict: {verdict}"
    raise AssertionError(message)


def _apply_calibrated_record_binding_critic(  # noqa: PLR0911 - ordered policy gates fail closed
    replacements: dict[str, str],
    ledger: dict[str, str],
    critic: tuple[str, list[str], str | None] | None,
) -> tuple[dict[str, str], str | None]:
    """Apply the V0.6.2.8 critic without reopening calibrated non-main equivalence."""

    if critic is None or "YES" not in {
        replacements.get("a_can_replace_b"),
        replacements.get("b_can_replace_a"),
    }:
        return replacements, None
    verdict, citations, issue = critic
    profile_a = ledger["span_content_profile_a"]
    profile_b = ledger["span_content_profile_b"]
    relation = replacements.get("relation_type")

    if profile_a == profile_b == "NON_MAIN_ONLY":
        if verdict != "NON_MAIN_POLICY_OR_STATE_CHANGE":
            return replacements, None
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": "RELATED_NON_DUPLICATE",
                "material_difference": "MAJOR",
                "primary_material_difference": "OTHER_MATERIAL",
            },
            "CALIBRATED_CRITIC_VETO_NON_MAIN_POLICY_OR_STATE_CHANGE",
        )

    if issue is not None or verdict == "UNRESOLVED":
        return _unresolved_span_target(), issue or "CALIBRATED_CRITIC_UNRESOLVED"
    if verdict in {
        "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
        "TWO_SIDED_OR_CONFLICTING",
        "NON_MAIN_POLICY_OR_STATE_CHANGE",
    }:
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": "RELATED_NON_DUPLICATE",
                "material_difference": "MAJOR",
                "primary_material_difference": (
                    "PAGE_ROLE_CHANGE" if verdict == "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT" else "OTHER_MATERIAL"
                ),
            },
            f"CALIBRATED_CRITIC_VETO_{verdict}",
        )
    if relation != "CONTAINMENT":
        return replacements, None
    if verdict == "ATOMIC_SAME_RECORD_EXTENSION":
        extension_prefix = "A" if ledger["span_a_delta"] == "SAME_RECORD_CONTENT_EXTENSION" else "B"
        if not any(span_id.startswith(extension_prefix) for span_id in citations):
            return _unresolved_span_target(), "CALIBRATED_CRITIC_EXTENSION_SIDE_WITHOUT_CITATION"
        return replacements, "CALIBRATED_CRITIC_CONFIRMED_ATOMIC_EXTENSION"
    if verdict == "BENIGN_NON_RECORD_DELTA":
        return (
            {
                "a_can_replace_b": "YES",
                "b_can_replace_a": "YES",
                "relation_type": "NEAR_SURFACE",
                "material_difference": "MINOR",
                "primary_material_difference": "OTHER_MATERIAL",
            },
            "CALIBRATED_CRITIC_RECLASSIFIED_BENIGN_DELTA",
        )
    if verdict == "NOT_APPLICABLE":
        return _unresolved_span_target(), "CALIBRATED_CRITIC_NOT_APPLICABLE_TO_CONTAINMENT"
    message = f"unhandled calibrated record-binding critic verdict: {verdict}"
    raise AssertionError(message)


def _apply_asymmetric_record_binding_critic(  # noqa: PLR0911 - ordered policy gates are intentionally explicit
    replacements: dict[str, str],
    ledger: dict[str, str],
    critic: tuple[str, list[str], str | None] | None,
) -> tuple[dict[str, str], str | None]:
    """Apply V0.6.2.9 arbitration without letting the critic erase owned decisions."""

    if critic is None or "YES" not in {
        replacements.get("a_can_replace_b"),
        replacements.get("b_can_replace_a"),
    }:
        return replacements, None
    verdict, citations, issue = critic
    profile_a = ledger["span_content_profile_a"]
    profile_b = ledger["span_content_profile_b"]
    relation = replacements.get("relation_type")
    translation = ledger["span_translation_status"]
    extension_prefix = None
    if ledger["span_a_delta"] == "SAME_RECORD_CONTENT_EXTENSION":
        extension_prefix = "A"
    elif ledger["span_b_delta"] == "SAME_RECORD_CONTENT_EXTENSION":
        extension_prefix = "B"

    # The main span ledger owns complete non-main-message equivalence and
    # material non-main changes. The record-binding critic is not calibrated
    # to distinguish those cases reliably.
    if profile_a == profile_b == "NON_MAIN_ONLY":
        return replacements, "ASYMMETRIC_CRITIC_DEFERRED_TO_NON_MAIN_LEDGER"

    if relation == "CONTAINMENT" and extension_prefix is not None:
        # Partial/additive translation is already established by bilateral
        # semantic spans. A lexical record-binding critic cannot reverse its
        # direction, but disagreement remains visible through LOW confidence.
        if translation == "PARTIAL_OR_ADDITIVE":
            if issue is not None or verdict in {
                "UNRESOLVED",
                "NOT_APPLICABLE",
                "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT",
                "TWO_SIDED_OR_CONFLICTING",
                "NON_MAIN_POLICY_OR_STATE_CHANGE",
            }:
                return (
                    {**replacements, "confidence_tier": "LOW"},
                    "ASYMMETRIC_CRITIC_DEFERRED_TO_ADDITIVE_TRANSLATION_LEDGER",
                )
            if verdict == "ATOMIC_SAME_RECORD_EXTENSION" and not any(
                span_id.startswith(extension_prefix) for span_id in citations
            ):
                return (
                    {**replacements, "confidence_tier": "LOW"},
                    "ASYMMETRIC_CRITIC_ADDITIVE_TRANSLATION_SIDE_DISAGREEMENT",
                )
            return replacements, "ASYMMETRIC_CRITIC_CONFIRMED_ADDITIVE_TRANSLATION"

        if issue is not None or verdict == "UNRESOLVED":
            return _unresolved_span_target(), issue or "ASYMMETRIC_CRITIC_UNRESOLVED"
        if verdict == "ATOMIC_SAME_RECORD_EXTENSION":
            if not any(span_id.startswith(extension_prefix) for span_id in citations):
                return _unresolved_span_target(), "ASYMMETRIC_CRITIC_EXTENSION_SIDE_WITHOUT_CITATION"
            return replacements, "ASYMMETRIC_CRITIC_CONFIRMED_ATOMIC_EXTENSION"
        if verdict == "BENIGN_NON_RECORD_DELTA":
            return replacements, "ASYMMETRIC_CRITIC_BENIGN_DID_NOT_OVERRIDE_EXTENSION"
        if verdict in {"SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT", "TWO_SIDED_OR_CONFLICTING"}:
            return (
                {
                    "a_can_replace_b": "NO",
                    "b_can_replace_a": "NO",
                    "relation_type": "RELATED_NON_DUPLICATE",
                    "material_difference": "MAJOR",
                    "primary_material_difference": (
                        "PAGE_ROLE_CHANGE" if verdict == "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT" else "OTHER_MATERIAL"
                    ),
                },
                f"ASYMMETRIC_CRITIC_VETO_{verdict}",
            )
        if verdict == "NON_MAIN_POLICY_OR_STATE_CHANGE":
            return (
                {
                    "a_can_replace_b": "NO",
                    "b_can_replace_a": "NO",
                    "relation_type": "RELATED_NON_DUPLICATE",
                    "material_difference": "MAJOR",
                    "primary_material_difference": "LEGAL_CONTEXT_CHANGE",
                },
                "ASYMMETRIC_CRITIC_VETO_NON_MAIN_POLICY_OR_STATE_CHANGE",
            )
        if verdict == "NOT_APPLICABLE":
            return (
                _unresolved_span_target(),
                "ASYMMETRIC_CRITIC_NOT_APPLICABLE_TO_CONTAINMENT",
            )

    if issue is not None or verdict == "UNRESOLVED":
        return _unresolved_span_target(), issue or "ASYMMETRIC_CRITIC_UNRESOLVED"
    if verdict in {"SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT", "TWO_SIDED_OR_CONFLICTING"}:
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": "RELATED_NON_DUPLICATE",
                "material_difference": "MAJOR",
                "primary_material_difference": (
                    "PAGE_ROLE_CHANGE" if verdict == "SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT" else "OTHER_MATERIAL"
                ),
            },
            f"ASYMMETRIC_CRITIC_VETO_{verdict}",
        )
    return replacements, None


def _v3_span_ledger_target(  # noqa: PLR0911 - ordered immutable policy gates exit independently
    parsed: dict[str, Any],
    ledger: dict[str, str],
    payload: dict[str, Any] | None,
    citation_issue: str | None,
    *,
    complete_message_translation: bool = False,
) -> tuple[dict[str, str], str]:
    packet = payload.get("semantic_diff_evidence") if isinstance(payload, dict) else None
    packet_status = packet.get("status") if isinstance(packet, dict) else None
    profile_a = ledger["span_content_profile_a"]
    profile_b = ledger["span_content_profile_b"]
    basis = ledger["span_shared_basis"]
    delta_a = ledger["span_a_delta"]
    delta_b = ledger["span_b_delta"]
    conflict = ledger["span_hard_conflict"]
    translation = ledger["span_translation_status"]
    if (
        citation_issue is not None
        or packet_status != "COMPLETE"
        or "UNREADABLE" in {profile_a, profile_b}
        or "UNRESOLVED" in {basis, delta_a, delta_b, conflict, translation}
    ):
        return _unresolved_span_target(), citation_issue or f"SEMANTIC_DIFF_{packet_status or 'UNAVAILABLE'}"

    def reject(rule: str) -> tuple[dict[str, str], str]:
        if conflict == "STATE_OR_VERSION":
            relation = "VERSION_RELATED"
        elif parsed["relation_type"] in {"VERSION_RELATED", "RELATED_NON_DUPLICATE", "UNRELATED"}:
            relation = parsed["relation_type"]
        else:
            relation = "RELATED_NON_DUPLICATE"
        if conflict == "PAGE_ROLE":
            primary = "PAGE_ROLE_CHANGE"
        elif conflict == "LIST_MEMBERSHIP":
            primary = "RESULT_SET_CHANGE"
        elif conflict == "LEGAL_CONTEXT":
            primary = "LEGAL_CONTEXT_CHANGE"
        elif conflict == "STATE_OR_VERSION":
            primary = _specific_no_no_difference(parsed, conflict)
        elif conflict == "IDENTITY_OR_SLOT" or "RECORD_IDENTITY_ROLE_OR_STATE_CHANGE" in {delta_a, delta_b}:
            primary = "DOCUMENT_IDENTITY_CHANGE"
        else:
            primary = _specific_no_no_difference(parsed, conflict)
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": relation,
                "material_difference": "MAJOR",
                "primary_material_difference": primary,
            },
            rule,
        )

    if conflict != "NONE":
        return reject("SPAN_HARD_CONFLICT_VETO")
    if profile_a != profile_b:
        return reject("SPAN_CONTENT_PROFILE_MISMATCH")
    if basis == "NONE":
        return reject("SPAN_NO_VERIFIED_SHARED_RECORD_BASIS")
    if translation == "COMPLETE_FAITHFUL":
        verified_translation = profile_a == profile_b == "SUBSTANTIVE_MAIN" and basis == "VERIFIED_SUBSTANTIVE_RECORD"
        if complete_message_translation:
            verified_translation |= (
                profile_a == profile_b == "NON_MAIN_ONLY" and basis == "VERIFIED_EQUIVALENT_NON_MAIN_MESSAGE"
            )
            if {delta_a, delta_b} - {"NONE", "SEMANTICALLY_COVERED", "UNIVERSAL_UI_OR_REPETITION"}:
                return reject("SPAN_TRANSLATION_WITH_UNCOVERED_DELTA")
        if verified_translation:
            return (
                {
                    "a_can_replace_b": "YES",
                    "b_can_replace_a": "YES",
                    "relation_type": "NEAR_SURFACE",
                    "material_difference": "NONE",
                    "primary_material_difference": "NONE",
                },
                "SPAN_COMPLETE_FAITHFUL_TRANSLATION",
            )
        return reject("SPAN_UNVERIFIED_TRANSLATION_IDENTITY")

    benign = {"NONE", "SEMANTICALLY_COVERED", "UNIVERSAL_UI_OR_REPETITION"}
    if profile_a == "NON_MAIN_ONLY":
        if basis != "VERIFIED_EQUIVALENT_NON_MAIN_MESSAGE":
            return reject("SPAN_NO_EQUIVALENT_NON_MAIN_MESSAGE")
        if delta_a not in benign or delta_b not in benign:
            return reject("SPAN_MATERIAL_NON_MAIN_DELTA")
    elif basis != "VERIFIED_SUBSTANTIVE_RECORD":
        return reject("SPAN_NO_VERIFIED_SUBSTANTIVE_RECORD")

    extensions = [delta_a == "SAME_RECORD_CONTENT_EXTENSION", delta_b == "SAME_RECORD_CONTENT_EXTENSION"]
    if any(extensions):
        if profile_a != "SUBSTANTIVE_MAIN" or sum(extensions) != 1:
            return reject("SPAN_INVALID_CONTENT_EXTENSION")
        other_delta = delta_b if extensions[0] else delta_a
        if other_delta not in benign:
            return reject("SPAN_CONTENT_EXTENSION_WITH_UNCOVERED_OPPOSITE_DELTA")
        return (
            {
                "a_can_replace_b": "YES" if extensions[0] else "NO",
                "b_can_replace_a": "YES" if extensions[1] else "NO",
                "relation_type": "CONTAINMENT",
                "material_difference": "MAJOR",
                "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
            },
            "SPAN_VERIFIED_SAME_RECORD_CONTENT_EXTENSION",
        )
    if delta_a not in benign or delta_b not in benign:
        return reject("SPAN_UNCOVERED_MATERIAL_DELTA")

    document_a = payload.get("document_a", {}).get("text") if isinstance(payload, dict) else None
    document_b = payload.get("document_b", {}).get("text") if isinstance(payload, dict) else None
    counts = packet.get("span_counts", {}) if isinstance(packet, dict) else {}
    if document_a == document_b:
        relation = "EXACT"
    elif counts.get("A_ONLY") == counts.get("B_ONLY") == 0:
        relation = "CANONICAL_EXACT"
    else:
        relation = "NEAR_SURFACE"
    material = "MINOR" if "UNIVERSAL_UI_OR_REPETITION" in {delta_a, delta_b} else "NONE"
    return (
        {
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": relation,
            "material_difference": material,
            "primary_material_difference": "OTHER_MATERIAL" if material == "MINOR" else "NONE",
        },
        "SPAN_VERIFIED_BIDIRECTIONAL_EQUIVALENCE",
    )


def _span_citation_evidence(
    payload: dict[str, Any] | None, citations: list[str]
) -> tuple[list[dict[str, Any]], str | None]:
    packet = payload.get("semantic_diff_evidence") if isinstance(payload, dict) else None
    if not isinstance(packet, dict):
        return [], "MISSING_SEMANTIC_DIFF_PACKET"
    by_id = {
        str(span["span_id"]): span for span in packet.get("spans", []) if isinstance(span, dict) and "span_id" in span
    }
    candidates = []
    for span_id in citations:
        span = by_id.get(span_id)
        if span is None:
            return [], "UNKNOWN_SPAN_CITATION"
        if span.get("kind") == "SHARED":
            for side, key in (("A", "a"), ("B", "b")):
                candidates.append(
                    {
                        "side": side,
                        "start_char": span[f"{key}_start_char"],
                        "end_char": span[f"{key}_end_char"],
                        "quote": span[f"{key}_text"],
                    }
                )
        else:
            candidates.append(
                {
                    "side": span["side"],
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                    "quote": span["text"],
                }
            )
    selected = []
    for side in ("A", "B"):
        match = next((item for item in candidates if item["side"] == side), None)
        if match is None:
            return [], "NONEXACT_RESULT_WITHOUT_BILATERAL_SPAN_EVIDENCE"
        selected.append(match)
    selected_keys = {(item["side"], item["start_char"], item["end_char"]) for item in selected}
    selected.extend(
        item for item in candidates if (item["side"], item["start_char"], item["end_char"]) not in selected_keys
    )
    return selected[:4], None


def _specific_no_no_difference(parsed: dict[str, Any], hard_conflict: str) -> str:
    primary = parsed["primary_material_difference"]
    specific = {
        "ENTITY_SLOT_CHANGE",
        "DOCUMENT_IDENTITY_CHANGE",
        "NUMBER_CHANGE",
        "DATE_TIME_CHANGE",
        "PRODUCT_VERSION_CHANGE",
        "RESULT_SET_CHANGE",
        "PAGE_ROLE_CHANGE",
        "LEGAL_CONTEXT_CHANGE",
        "NEGATION_CHANGE",
        "CODE_LITERAL_CHANGE",
        "CODE_OUTPUT_CHANGE",
    }
    if hard_conflict == "PAGE_ROLE":
        return "PAGE_ROLE_CHANGE"
    if hard_conflict == "LIST_MEMBERSHIP":
        return "RESULT_SET_CHANGE"
    if hard_conflict == "LEGAL_CONTEXT":
        return "LEGAL_CONTEXT_CHANGE"
    if primary in specific:
        return primary
    if hard_conflict == "IDENTITY_OR_SLOT":
        return "DOCUMENT_IDENTITY_CHANGE"
    return "OTHER_MATERIAL"


def _v3_semantic_ledger_target(  # noqa: PLR0911 - immutable ledger versions dispatch independently
    parsed: dict[str, Any], ledger: dict[str, str]
) -> tuple[dict[str, str], str]:
    """Resolve directions from the ordered semantic gate ledger."""

    if "boundary_delta_class" in ledger:
        return _v3_semantic_ledger_v4_target(parsed, ledger)
    if "record_identity_support" in ledger:
        return _v3_semantic_ledger_v3_target(parsed, ledger)
    if "record_alignment" in ledger:
        return _v3_semantic_ledger_v2_target(parsed, ledger)

    profile_a = ledger["content_profile_a"]
    profile_b = ledger["content_profile_b"]
    basis = ledger["shared_content_basis"]
    conflict = ledger["hard_conflict"]
    difference = ledger["decisive_difference_location"]

    if (
        "UNREADABLE" in {profile_a, profile_b}
        or basis == "UNRESOLVED"
        or conflict == "UNRESOLVED"
        or difference == "UNRESOLVED"
    ):
        return (
            {
                "a_can_replace_b": "UNRESOLVED",
                "b_can_replace_a": "UNRESOLVED",
                "relation_type": "UNRESOLVED",
                "material_difference": "UNRESOLVED",
                "primary_material_difference": "UNRESOLVED",
                "dominant_overlap_source": "UNRESOLVED",
                "primary_risk_factor": "EXTRACTION_OR_PAYLOAD_LIMIT",
                "confidence_tier": "LOW",
            },
            "UNREADABLE_OR_UNRESOLVED_GATE",
        )

    no_no_rule: str | None = None
    if conflict != "NONE":
        no_no_rule = "HARD_CONFLICT_VETO"
    elif profile_a != profile_b:
        no_no_rule = "SUBSTANTIVE_NON_MAIN_PROFILE_MISMATCH"
    elif profile_a == "SUBSTANTIVE_MAIN" and basis != "SUBSTANTIVE_ANCHOR":
        no_no_rule = "NO_SHARED_SUBSTANTIVE_ANCHOR"
    elif profile_a == "NON_MAIN_ONLY" and basis != "EQUIVALENT_NON_MAIN_MESSAGE":
        no_no_rule = "NON_MAIN_MESSAGE_NOT_EQUIVALENT"
    elif difference == "TWO_SIDED_MAIN_DIVERGENCE":
        no_no_rule = "TWO_SIDED_MAIN_DIVERGENCE"
    elif profile_a == "NON_MAIN_ONLY" and difference in {
        "A_ONLY_MAIN_ADDITION",
        "B_ONLY_MAIN_ADDITION",
    }:
        no_no_rule = "NON_MAIN_PROFILE_CANNOT_SUPPORT_MAIN_ADDITION"

    if no_no_rule is not None:
        if conflict == "STATE_OR_VERSION":
            relation = "VERSION_RELATED"
        elif parsed["relation_type"] in {"VERSION_RELATED", "RELATED_NON_DUPLICATE", "UNRELATED"}:
            relation = parsed["relation_type"]
        elif no_no_rule in {"SUBSTANTIVE_NON_MAIN_PROFILE_MISMATCH", "NO_SHARED_SUBSTANTIVE_ANCHOR"}:
            relation = "UNRELATED"
        else:
            relation = "RELATED_NON_DUPLICATE"
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": relation,
                "material_difference": "MAJOR",
                "primary_material_difference": _specific_no_no_difference(parsed, conflict),
            },
            no_no_rule,
        )

    if difference in {"NONE", "NON_MAIN_ONLY"}:
        material = "NONE" if difference == "NONE" else "MINOR"
        relation = (
            parsed["relation_type"]
            if material == "NONE" and parsed["relation_type"] in {"EXACT", "CANONICAL_EXACT"}
            else "NEAR_SURFACE"
        )
        return (
            {
                "a_can_replace_b": "YES",
                "b_can_replace_a": "YES",
                "relation_type": relation,
                "material_difference": material,
                "primary_material_difference": "NONE" if material == "NONE" else "OTHER_MATERIAL",
            },
            "BIDIRECTIONAL_EQUIVALENCE",
        )

    if difference in {"A_ONLY_MAIN_ADDITION", "B_ONLY_MAIN_ADDITION"}:
        return (
            {
                "a_can_replace_b": "YES" if difference == "A_ONLY_MAIN_ADDITION" else "NO",
                "b_can_replace_a": "YES" if difference == "B_ONLY_MAIN_ADDITION" else "NO",
                "relation_type": "CONTAINMENT",
                "material_difference": "MAJOR",
                "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
            },
            "NONEMPTY_SUBSTANTIVE_CONTAINMENT",
        )

    message = f"unhandled semantic-ledger difference: {difference}"
    raise AssertionError(message)


def _v3_semantic_ledger_v2_target(parsed: dict[str, Any], ledger: dict[str, str]) -> tuple[dict[str, str], str]:
    """Resolve V0.6.2.3 directions from record alignment and boundary diagnostics."""

    profile_a = ledger["content_profile_a"]
    profile_b = ledger["content_profile_b"]
    basis = ledger["shared_content_basis"]
    conflict = ledger["hard_conflict"]
    difference = ledger["decisive_difference_location"]
    alignment = ledger["record_alignment"]
    non_main = ledger["non_main_difference"]
    translation = ledger["translation_status"]
    if "UNREADABLE" in {profile_a, profile_b} or "UNRESOLVED" in {
        basis,
        conflict,
        difference,
        alignment,
        non_main,
        translation,
    }:
        return (
            {
                "a_can_replace_b": "UNRESOLVED",
                "b_can_replace_a": "UNRESOLVED",
                "relation_type": "UNRESOLVED",
                "material_difference": "UNRESOLVED",
                "primary_material_difference": "UNRESOLVED",
                "dominant_overlap_source": "UNRESOLVED",
                "primary_risk_factor": "EXTRACTION_OR_PAYLOAD_LIMIT",
                "confidence_tier": "LOW",
            },
            "UNREADABLE_OR_UNRESOLVED_GATE",
        )

    no_no_rule: str | None = None
    if conflict != "NONE":
        no_no_rule = "HARD_CONFLICT_VETO"
    elif (
        translation == "COMPLETE_FAITHFUL"
        and profile_a == profile_b == "SUBSTANTIVE_MAIN"
        and basis == "SUBSTANTIVE_ANCHOR"
        and alignment == "SAME_SUBSTANTIVE_RECORD"
    ):
        return (
            {
                "a_can_replace_b": "YES",
                "b_can_replace_a": "YES",
                "relation_type": "NEAR_SURFACE",
                "material_difference": "NONE",
                "primary_material_difference": "NONE",
            },
            "COMPLETE_FAITHFUL_TRANSLATION",
        )
    elif profile_a != profile_b:
        no_no_rule = "SUBSTANTIVE_NON_MAIN_PROFILE_MISMATCH"
    elif profile_a == "SUBSTANTIVE_MAIN" and (basis != "SUBSTANTIVE_ANCHOR" or alignment != "SAME_SUBSTANTIVE_RECORD"):
        no_no_rule = "NO_CONFIRMED_SAME_SUBSTANTIVE_RECORD"
    elif profile_a == "NON_MAIN_ONLY" and (
        basis != "EQUIVALENT_NON_MAIN_MESSAGE" or alignment != "SAME_NON_MAIN_MESSAGE"
    ):
        no_no_rule = "NO_EQUIVALENT_NON_MAIN_MESSAGE"
    elif non_main == "MATERIAL_MESSAGE_OR_STATE":
        no_no_rule = "MATERIAL_NON_MAIN_DIFFERENCE"
    elif difference == "TWO_SIDED_MAIN_DIVERGENCE":
        no_no_rule = "TWO_SIDED_MAIN_DIVERGENCE"
    elif profile_a == "NON_MAIN_ONLY" and difference in {
        "A_ONLY_MAIN_ADDITION",
        "B_ONLY_MAIN_ADDITION",
    }:
        no_no_rule = "NON_MAIN_PROFILE_CANNOT_SUPPORT_MAIN_ADDITION"

    if no_no_rule is not None:
        if conflict == "STATE_OR_VERSION":
            relation = "VERSION_RELATED"
        elif parsed["relation_type"] in {"VERSION_RELATED", "RELATED_NON_DUPLICATE", "UNRELATED"}:
            relation = parsed["relation_type"]
        elif alignment == "GENERIC_OR_TEMPLATE_OVERLAP_ONLY" or profile_a != profile_b:
            relation = "UNRELATED"
        else:
            relation = "RELATED_NON_DUPLICATE"
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": relation,
                "material_difference": "MAJOR",
                "primary_material_difference": _specific_no_no_difference(parsed, conflict),
            },
            no_no_rule,
        )

    if difference in {"NONE", "NON_MAIN_ONLY"}:
        if difference == "NON_MAIN_ONLY" and non_main != "NONE_OR_IGNORABLE_CHROME":
            message = f"non-main delta is incompatible with equivalence: {non_main}"
            raise AssertionError(message)
        material = "NONE" if difference == "NONE" else "MINOR"
        relation = (
            parsed["relation_type"]
            if material == "NONE" and parsed["relation_type"] in {"EXACT", "CANONICAL_EXACT"}
            else "NEAR_SURFACE"
        )
        return (
            {
                "a_can_replace_b": "YES",
                "b_can_replace_a": "YES",
                "relation_type": relation,
                "material_difference": material,
                "primary_material_difference": "NONE" if material == "NONE" else "OTHER_MATERIAL",
            },
            "BIDIRECTIONAL_EQUIVALENCE",
        )

    if difference in {"A_ONLY_MAIN_ADDITION", "B_ONLY_MAIN_ADDITION"}:
        require(
            profile_a == profile_b == "SUBSTANTIVE_MAIN"
            and basis == "SUBSTANTIVE_ANCHOR"
            and alignment == "SAME_SUBSTANTIVE_RECORD",
            "LOCAL_NDD_OUTPUT_INVALID",
            "V0.6.2.3 containment requires a confirmed same-record substantive anchor",
        )
        return (
            {
                "a_can_replace_b": "YES" if difference == "A_ONLY_MAIN_ADDITION" else "NO",
                "b_can_replace_a": "YES" if difference == "B_ONLY_MAIN_ADDITION" else "NO",
                "relation_type": "CONTAINMENT",
                "material_difference": "MAJOR",
                "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
            },
            "NONEMPTY_SAME_RECORD_CONTAINMENT",
        )

    message = f"unhandled V0.6.2.3 semantic-ledger difference: {difference}"
    raise AssertionError(message)


def _v3_semantic_ledger_v3_target(  # noqa: PLR0911 - ordered policy gates exit immediately
    parsed: dict[str, Any], ledger: dict[str, str]
) -> tuple[dict[str, str], str]:
    """Resolve V0.6.2.4 only after verifiable scope and surface gates."""

    identity = ledger["record_identity_support"]
    scope = ledger["overlap_scope"]
    surface = ledger["surface_delta_type"]
    if "UNRESOLVED" in {identity, scope, surface}:
        return (
            {
                "a_can_replace_b": "UNRESOLVED",
                "b_can_replace_a": "UNRESOLVED",
                "relation_type": "UNRESOLVED",
                "material_difference": "UNRESOLVED",
                "primary_material_difference": "UNRESOLVED",
                "dominant_overlap_source": "UNRESOLVED",
                "primary_risk_factor": "EXTRACTION_OR_PAYLOAD_LIMIT",
                "confidence_tier": "LOW",
            },
            "UNRESOLVED_VERIFIABLE_BOUNDARY_GATE",
        )

    if ledger["hard_conflict"] != "NONE":
        return _v3_semantic_ledger_v2_target(parsed, ledger)

    def reject(rule: str) -> tuple[dict[str, str], str]:
        relation = (
            parsed["relation_type"]
            if parsed["relation_type"] in {"VERSION_RELATED", "RELATED_NON_DUPLICATE", "UNRELATED"}
            else "RELATED_NON_DUPLICATE"
        )
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": relation,
                "material_difference": "MAJOR",
                "primary_material_difference": _specific_no_no_difference(parsed, ledger["hard_conflict"]),
            },
            rule,
        )

    if surface == "MATERIAL_PROPOSITION_OR_RECORD":
        return reject("MATERIAL_SURFACE_PROPOSITION_OR_RECORD")
    if identity == "DIFFERENT_OR_CONFLICTING":
        return reject("DIFFERENT_OR_CONFLICTING_RECORD_IDENTITY")

    profile_a = ledger["content_profile_a"]
    profile_b = ledger["content_profile_b"]
    if profile_a == profile_b == "SUBSTANTIVE_MAIN":
        if identity not in {"VERBATIM_SHARED_IDENTIFIER", "DISTINCTIVE_MULTI_FACT_IDENTITY"}:
            return reject("NO_VERIFIABLE_SUBSTANTIVE_RECORD_IDENTITY")
        if scope != "DOCUMENT_WIDE_SAME_RECORD":
            return reject("NON_DOCUMENT_WIDE_SUBSTANTIVE_OVERLAP")
    elif profile_a == profile_b == "NON_MAIN_ONLY":
        if identity != "SAME_NON_MAIN_MESSAGE" or scope != "COMPLETE_NON_MAIN_MESSAGE":
            return reject("NO_VERIFIABLE_COMPLETE_NON_MAIN_MESSAGE")
        if surface not in {
            "NONE_OR_FORMATTING",
            "NAV_LABEL_BYLINE_ONLY",
            "REDUNDANT_REPETITION_ONLY",
        }:
            return reject("NON_MAIN_SURFACE_NOT_BENIGN")

    if ledger["decisive_difference_location"] == "NON_MAIN_ONLY" and surface not in {
        "NONE_OR_FORMATTING",
        "NAV_LABEL_BYLINE_ONLY",
        "REDUNDANT_REPETITION_ONLY",
    }:
        return reject("NON_MAIN_SURFACE_NOT_BENIGN")
    return _v3_semantic_ledger_v2_target(parsed, ledger)


def _v3_semantic_ledger_v4_target(  # noqa: PLR0911 - policy gates intentionally exit in order
    parsed: dict[str, Any], ledger: dict[str, str]
) -> tuple[dict[str, str], str]:
    """Resolve V0.6.2.5 from an explicit boundary-delta class without a second quote contract."""

    boundary = ledger["boundary_delta_class"]
    identity = ledger["record_identity_support"]
    scope = ledger["overlap_scope"]
    surface = ledger["surface_delta_type"]
    profile_a = ledger["content_profile_a"]
    profile_b = ledger["content_profile_b"]
    basis = ledger["shared_content_basis"]
    conflict = ledger["hard_conflict"]
    difference = ledger["decisive_difference_location"]
    alignment = ledger["record_alignment"]
    non_main = ledger["non_main_difference"]
    translation = ledger["translation_status"]

    if "UNREADABLE" in {profile_a, profile_b} or "UNRESOLVED" in {
        basis,
        conflict,
        difference,
        alignment,
        non_main,
        translation,
        identity,
        scope,
        surface,
        boundary,
    }:
        return (
            {
                "a_can_replace_b": "UNRESOLVED",
                "b_can_replace_a": "UNRESOLVED",
                "relation_type": "UNRESOLVED",
                "material_difference": "UNRESOLVED",
                "primary_material_difference": "UNRESOLVED",
                "dominant_overlap_source": "UNRESOLVED",
                "primary_risk_factor": "EXTRACTION_OR_PAYLOAD_LIMIT",
                "confidence_tier": "LOW",
            },
            "UNREADABLE_OR_UNRESOLVED_BOUNDARY_CLASS",
        )

    def reject(rule: str, *, default_primary: str = "OTHER_MATERIAL") -> tuple[dict[str, str], str]:
        if conflict == "STATE_OR_VERSION":
            relation = "VERSION_RELATED"
        elif parsed["relation_type"] in {"VERSION_RELATED", "RELATED_NON_DUPLICATE", "UNRELATED"}:
            relation = parsed["relation_type"]
        elif scope in {"LOCAL_PASSAGE_ONLY", "GENERIC_FAMILY_OR_TEMPLATE_ONLY", "NONE"}:
            relation = "UNRELATED"
        else:
            relation = "RELATED_NON_DUPLICATE"
        primary = _specific_no_no_difference(parsed, conflict)
        if primary == "OTHER_MATERIAL":
            primary = default_primary
        return (
            {
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": relation,
                "material_difference": "MAJOR",
                "primary_material_difference": primary,
            },
            rule,
        )

    if conflict != "NONE":
        return reject("HARD_CONFLICT_VETO")
    if identity == "DIFFERENT_OR_CONFLICTING":
        return reject("DIFFERENT_OR_CONFLICTING_RECORD_IDENTITY", default_primary="DOCUMENT_IDENTITY_CHANGE")
    if boundary == "RECORD_IDENTITY_ROLE_OR_STATE_CHANGE":
        return reject("BOUNDARY_RECORD_IDENTITY_ROLE_OR_STATE_CHANGE", default_primary="DOCUMENT_IDENTITY_CHANGE")
    if boundary == "MATERIAL_NON_MAIN_MESSAGE_CHANGE":
        return reject("BOUNDARY_MATERIAL_NON_MAIN_MESSAGE_CHANGE")
    if boundary == "TWO_SIDED_CONTENT_CHANGE":
        return reject("BOUNDARY_TWO_SIDED_CONTENT_CHANGE")
    if profile_a != profile_b:
        return reject("SUBSTANTIVE_NON_MAIN_PROFILE_MISMATCH")

    if boundary == "SAME_RECORD_CONTENT_EXTENSION":
        if difference not in {"A_ONLY_MAIN_ADDITION", "B_ONLY_MAIN_ADDITION"}:
            return reject("BOUNDARY_EXTENSION_WITHOUT_ONE_SIDED_COVERAGE")
        if profile_a != "SUBSTANTIVE_MAIN":
            return reject("NON_MAIN_PROFILE_CANNOT_SUPPORT_CONTENT_EXTENSION")
        if (
            basis != "SUBSTANTIVE_ANCHOR"
            or alignment != "SAME_SUBSTANTIVE_RECORD"
            or identity not in {"VERBATIM_SHARED_IDENTIFIER", "DISTINCTIVE_MULTI_FACT_IDENTITY"}
            or scope != "DOCUMENT_WIDE_SAME_RECORD"
        ):
            return reject("UNVERIFIED_SAME_RECORD_CONTENT_EXTENSION")
        return (
            {
                "a_can_replace_b": "YES" if difference == "A_ONLY_MAIN_ADDITION" else "NO",
                "b_can_replace_a": "YES" if difference == "B_ONLY_MAIN_ADDITION" else "NO",
                "relation_type": "CONTAINMENT",
                "material_difference": "MAJOR",
                "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
            },
            "VERIFIED_SAME_RECORD_CONTENT_EXTENSION",
        )

    if boundary not in {"NONE_OR_FORMATTING", "UNIVERSAL_UI_OR_REDUNDANT_REPETITION"}:
        message = f"unhandled V0.6.2.5 boundary class: {boundary}"
        raise AssertionError(message)
    if difference not in {"NONE", "NON_MAIN_ONLY"}:
        return reject("BENIGN_BOUNDARY_WITHOUT_EQUIVALENT_COVERAGE")
    if boundary == "NONE_OR_FORMATTING" and surface not in {"NONE_OR_FORMATTING", "NOT_APPLICABLE"}:
        return reject("NONE_BOUNDARY_HAS_NONTRIVIAL_SURFACE_DELTA")
    if boundary == "UNIVERSAL_UI_OR_REDUNDANT_REPETITION" and surface not in {
        "NAV_LABEL_BYLINE_ONLY",
        "REDUNDANT_REPETITION_ONLY",
    }:
        return reject("BENIGN_BOUNDARY_HAS_NONBENIGN_SURFACE_DELTA")

    if profile_a == "SUBSTANTIVE_MAIN":
        if (
            basis != "SUBSTANTIVE_ANCHOR"
            or alignment != "SAME_SUBSTANTIVE_RECORD"
            or identity not in {"VERBATIM_SHARED_IDENTIFIER", "DISTINCTIVE_MULTI_FACT_IDENTITY"}
            or scope != "DOCUMENT_WIDE_SAME_RECORD"
        ):
            return reject("NO_VERIFIED_SUBSTANTIVE_EQUIVALENCE")
    elif (
        basis != "EQUIVALENT_NON_MAIN_MESSAGE"
        or alignment != "SAME_NON_MAIN_MESSAGE"
        or identity != "SAME_NON_MAIN_MESSAGE"
        or scope != "COMPLETE_NON_MAIN_MESSAGE"
        or non_main != "NONE_OR_IGNORABLE_CHROME"
    ):
        return reject("NO_VERIFIED_COMPLETE_NON_MAIN_EQUIVALENCE")

    material = "NONE" if boundary == "NONE_OR_FORMATTING" and difference == "NONE" else "MINOR"
    relation = (
        parsed["relation_type"]
        if material == "NONE" and parsed["relation_type"] in {"EXACT", "CANONICAL_EXACT"}
        else "NEAR_SURFACE"
    )
    return (
        {
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": relation,
            "material_difference": material,
            "primary_material_difference": "NONE" if material == "NONE" else "OTHER_MATERIAL",
        },
        "VERIFIED_BIDIRECTIONAL_EQUIVALENCE",
    )


def _validate_v3_record_identity_evidence(
    value: dict[str, Any], ledger: dict[str, str], payload: dict[str, Any] | None
) -> None:
    support = ledger.get("record_identity_support")
    if support not in {"VERBATIM_SHARED_IDENTIFIER", "DISTINCTIVE_MULTI_FACT_IDENTITY"}:
        return
    evidence, _ = _extract_quote_evidence(
        {"record_identity_support": value["record_identity_support"]}, payload, allow_fallback=False
    )
    by_side = {side: [item["quote"] for item in evidence if item["side"] == side] for side in ("A", "B")}
    minimum = 1 if support == "VERBATIM_SHARED_IDENTIFIER" else 2
    require(
        all(len(by_side[side]) >= minimum for side in ("A", "B")),
        "LOCAL_NDD_OUTPUT_INVALID",
        "positive record identity support requires field-specific aligned bilateral evidence",
        field="record_identity_support",
        minimum_quotes_per_side=minimum,
    )
    if support == "VERBATIM_SHARED_IDENTIFIER":
        keys_a = {_quote_match_key(quote)[0] for quote in by_side["A"]}
        keys_b = {_quote_match_key(quote)[0] for quote in by_side["B"]}
        require(
            bool(keys_a & keys_b),
            "LOCAL_NDD_OUTPUT_INVALID",
            "verbatim shared identifier requires a matching normalized quote on both sides",
            field="record_identity_support",
        )


def _adapt_v3(
    value: dict[str, Any],
    payload: dict[str, Any] | None,
    record_binding_critic: dict[str, Any] | None = None,
    record_binding_policy: str = "v1",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed: dict[str, Any] = {}
    for field in _V3_MODEL_FIELDS:
        score = _score(value, field)
        require(isinstance(score, str), "LOCAL_NDD_OUTPUT_INVALID", "categorical score must be text", field=field)
        parsed[field] = score.upper()
    semantic_ledger = _optional_v3_semantic_ledger(value)
    span_ledger_result = _optional_v3_span_ledger(value, payload)
    require(
        semantic_ledger is None or span_ledger_result is None,
        "LOCAL_NDD_OUTPUT_INVALID",
        "legacy semantic ledger and span ledger cannot be combined",
    )
    span_ledger = span_ledger_result[0] if span_ledger_result is not None else None
    span_citations = span_ledger_result[1] if span_ledger_result is not None else []
    span_citation_issue = span_ledger_result[2] if span_ledger_result is not None else None
    retained_review = (
        parse_retained_conflict(record_binding_critic, payload)
        if record_binding_policy in {"v6", "v7-exact", "v7", "v8", "v9"}
        else None
    )
    record_scope = (
        parse_record_scope(record_binding_critic, payload) if record_binding_policy in {"v7", "v8", "v9"} else None
    )
    boundary_review = parse_boundary_review(record_binding_critic, payload) if record_binding_policy == "v4" else None
    scoped_review = (
        parse_scoped_review(
            record_binding_critic,
            payload,
            translation_enabled=record_binding_policy != "v5-policy",
            legacy_binding=record_binding_policy != "v5-replay",
        )
        if record_binding_policy in {"v5", "v5-policy", "v5-replay"}
        else None
    )
    require(
        (boundary_review is None and scoped_review is None and retained_review is None) or span_ledger is not None,
        "LOCAL_NDD_OUTPUT_INVALID",
        "boundary arbitration requires the span ledger",
    )
    critic_result = (
        scoped_review.legacy_critic
        if scoped_review is not None
        else _optional_record_binding_critic(record_binding_critic, payload)
        if boundary_review is None
        else None
    )
    critic_citations = (
        scoped_review.evidence_ids
        if scoped_review is not None
        else boundary_review.evidence_ids
        if boundary_review is not None and not boundary_review.issues
        else critic_result[1]
        if critic_result is not None
        else []
    )
    if (
        retained_review is not None
        and retained_review.score not in {"NONE", "NOT_APPLICABLE", "UNRESOLVED"}
        and not retained_review.issues
    ):
        critic_citations = list(dict.fromkeys([*retained_review.citations, *critic_citations]))
    if record_scope is not None and not record_scope.issues:
        critic_citations = list(dict.fromkeys([*record_scope.citations, *critic_citations]))
    contract_events = []
    replacements: dict[str, str] = {}
    ledger_rule = None
    if span_ledger is not None:
        replacements, ledger_rule = _v3_span_ledger_target(
            parsed,
            span_ledger,
            payload,
            span_citation_issue,
            complete_message_translation=record_binding_policy in {"v6-route", "v6", "v7-exact", "v7", "v8", "v9"},
        )
        if (
            record_binding_policy in {"v7-exact", "v7", "v8", "v9"}
            and replacements.get("a_can_replace_b") == replacements.get("b_can_replace_a") == "YES"
            and complete_visible_equality(payload)
        ):
            replacements = {
                **replacements,
                "relation_type": "EXACT",
                "material_difference": "NONE",
                "primary_material_difference": "NONE",
            }
            critic_rule = "COMPLETE_VISIBLE_EXACT_IDENTITY_PROTECTED"
        elif scoped_review is not None:
            legacy_target, legacy_rule = _apply_asymmetric_record_binding_critic(
                replacements, span_ledger, critic_result
            )
            replacements, critic_rule = arbitrate_scoped_review(
                replacements,
                span_ledger,
                scoped_review,
                legacy_target=legacy_target,
                legacy_rule=legacy_rule,
                unresolved=_unresolved_span_target(),
            )
        elif boundary_review is not None:
            replacements, critic_rule = arbitrate_boundary_review(
                replacements, span_ledger, boundary_review, _unresolved_span_target()
            )
        elif record_binding_policy in {"v6", "v7-exact", "v7", "v8", "v9"}:
            main_target = replacements
            legacy_target, legacy_rule = _apply_asymmetric_record_binding_critic(
                replacements, span_ledger, critic_result
            )
            replacements, critic_rule = arbitrate_retained_conflict(
                replacements,
                span_ledger,
                retained_review,
                critic_verdict=critic_result[0],
                legacy_target=legacy_target,
                legacy_rule=legacy_rule,
                unresolved=_unresolved_span_target(),
            )
            if record_scope is not None:
                scope_arbitrator = (
                    arbitrate_record_scope_v9
                    if record_binding_policy == "v9"
                    else arbitrate_record_scope_v8
                    if record_binding_policy == "v8"
                    else arbitrate_record_scope
                )
                replacements, critic_rule = scope_arbitrator(
                    main_target,
                    record_scope,
                    retained_review,
                    critic_result,
                    legacy_target=replacements,
                    legacy_rule=critic_rule,
                    unresolved=_unresolved_span_target(),
                    **({"ledger": span_ledger} if record_binding_policy in {"v8", "v9"} else {}),
                )
        elif record_binding_policy in {"v3", "v6-route"}:
            replacements, critic_rule = _apply_asymmetric_record_binding_critic(
                replacements, span_ledger, critic_result
            )
        elif record_binding_policy == "v2":
            replacements, critic_rule = _apply_calibrated_record_binding_critic(
                replacements, span_ledger, critic_result
            )
        else:
            require(
                record_binding_policy == "v1",
                "LOCAL_NDD_OUTPUT_INVALID",
                "unknown record-binding critic policy",
                record_binding_policy=record_binding_policy,
            )
            replacements, critic_rule = _apply_record_binding_critic(replacements, span_ledger, critic_result)
        if critic_rule is not None:
            ledger_rule = f"{ledger_rule}+{critic_rule}"
    elif semantic_ledger is not None:
        if "boundary_delta_class" not in semantic_ledger:
            _validate_v3_record_identity_evidence(value, semantic_ledger, payload)
        replacements, ledger_rule = _v3_semantic_ledger_target(parsed, semantic_ledger)
    else:
        directions = {parsed["a_can_replace_b"], parsed["b_can_replace_a"]}
        if directions == {"YES", "NO"}:
            replacements = {
                "relation_type": "CONTAINMENT",
                "material_difference": "MAJOR",
                "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
            }
        elif parsed["a_can_replace_b"] == parsed["b_can_replace_a"] == "NO":
            if parsed["relation_type"] not in {"VERSION_RELATED", "RELATED_NON_DUPLICATE", "UNRELATED"}:
                replacements["relation_type"] = "RELATED_NON_DUPLICATE"
            if parsed["material_difference"] != "MAJOR":
                replacements["material_difference"] = "MAJOR"
            if parsed["primary_material_difference"] == "NONE":
                replacements["primary_material_difference"] = "OTHER_MATERIAL"
        elif parsed["a_can_replace_b"] == parsed["b_can_replace_a"] == "YES":
            if parsed["relation_type"] in {"EXACT", "CANONICAL_EXACT"}:
                replacements["material_difference"] = "NONE"
                replacements["primary_material_difference"] = "NONE"
            else:
                if parsed["relation_type"] != "NEAR_SURFACE":
                    replacements["relation_type"] = "NEAR_SURFACE"
                    replacements["primary_material_difference"] = "OTHER_MATERIAL"
                if parsed["material_difference"] == "MAJOR":
                    replacements["material_difference"] = "MINOR"
                effective_material = replacements.get("material_difference", parsed["material_difference"])
                if effective_material == "NONE":
                    replacements["primary_material_difference"] = "NONE"
                elif parsed["primary_material_difference"] == "NONE":
                    replacements["primary_material_difference"] = "OTHER_MATERIAL"
    for field, replacement in replacements.items():
        source = parsed[field]
        if source == replacement:
            continue
        parsed[field] = replacement
        contract_events.append(
            {
                "action": (
                    "reconcile_v3_semantic_field_from_ledger"
                    if semantic_ledger is not None or span_ledger is not None
                    else "reconcile_v3_semantic_field_from_directions"
                ),
                "field": field,
                "source": source,
                "replacement": replacement,
                **({"ledger_rule": ledger_rule} if ledger_rule is not None else {}),
            }
        )
    parsed["same_duplicate_group"] = _derive_same_duplicate_group(parsed["a_can_replace_b"], parsed["b_can_replace_a"])
    if (
        payload is not None
        and payload.get("long_document_evidence", {}).get("truncated") is True
        and parsed["confidence_tier"] == "HIGH"
    ):
        parsed["confidence_tier"] = "MEDIUM"
        contract_events.append({"action": "cap_truncated_confidence_tier", "source": "HIGH", "replacement": "MEDIUM"})
    if parsed["same_duplicate_group"] == "UNRESOLVED" and parsed["confidence_tier"] != "LOW":
        contract_events.append(
            {
                "action": "force_unresolved_confidence_tier",
                "source": parsed["confidence_tier"],
                "replacement": "LOW",
            }
        )
        parsed["confidence_tier"] = "LOW"
    boundary_requires_judgment = (
        parsed["relation_type"] == "CONTAINMENT"
        or span_ledger is not None
        or (
            semantic_ledger is not None
            and semantic_ledger.get("boundary_delta_class") not in {None, "NONE_OR_FORMATTING", "UNRESOLVED"}
        )
        or parsed["primary_risk_factor"] != "NONE"
        or parsed["dominant_overlap_source"]
        in {
            "SHARED_PAGE_TEMPLATE",
            "SITE_CHROME",
            "COOKIE_CONSENT",
            "LEGAL_POLICY_TEMPLATE",
            "ERROR_AUTH_PAYWALL",
            "LOCAL_PASSAGE",
            "PARSER_ARTIFACT",
        }
    )
    if parsed["confidence_tier"] == "HIGH" and boundary_requires_judgment:
        parsed["confidence_tier"] = "MEDIUM"
        contract_events.append(
            {
                "action": "cap_boundary_confidence_tier",
                "source": "HIGH",
                "replacement": "MEDIUM",
            }
        )
    parsed["reason_codes"] = []
    if semantic_ledger is not None:
        ledger_reason_prefixes = {
            "content_profile_a": "CONTENT_PROFILE_A",
            "content_profile_b": "CONTENT_PROFILE_B",
            "shared_content_basis": "SHARED_CONTENT_BASIS",
            "hard_conflict": "HARD_CONFLICT",
            "decisive_difference_location": "DIFFERENCE_LOCATION",
            "record_alignment": "RECORD_ALIGNMENT",
            "non_main_difference": "NON_MAIN_DIFFERENCE",
            "translation_status": "TRANSLATION_STATUS",
            "record_identity_support": "RECORD_IDENTITY_SUPPORT",
            "overlap_scope": "OVERLAP_SCOPE",
            "surface_delta_type": "SURFACE_DELTA_TYPE",
            "boundary_delta_class": "BOUNDARY_DELTA_CLASS",
        }
        parsed["reason_codes"].extend(
            f"{ledger_reason_prefixes[field]}:{score}" for field, score in semantic_ledger.items()
        )
        parsed["reason_codes"].append(f"SEMANTIC_LEDGER_RULE:{ledger_rule}")
    if span_ledger is not None:
        span_reason_prefixes = {
            "span_content_profile_a": "SPAN_CONTENT_PROFILE_A",
            "span_content_profile_b": "SPAN_CONTENT_PROFILE_B",
            "span_shared_basis": "SPAN_SHARED_BASIS",
            "span_a_delta": "SPAN_A_DELTA",
            "span_b_delta": "SPAN_B_DELTA",
            "span_hard_conflict": "SPAN_HARD_CONFLICT",
            "span_translation_status": "SPAN_TRANSLATION_STATUS",
        }
        parsed["reason_codes"].extend(f"{span_reason_prefixes[field]}:{score}" for field, score in span_ledger.items())
        parsed["reason_codes"].append(f"SPAN_LEDGER_RULE:{ledger_rule}")
        if span_citation_issue is not None:
            parsed["reason_codes"].append(f"SPAN_CITATION_ISSUE:{span_citation_issue}")
        if scoped_review is not None:
            parsed["reason_codes"].extend(
                f"BOUNDARY_{field.upper()}:{score}" for field, score in scoped_review.scores.items()
            )
            parsed["reason_codes"].extend(
                f"BOUNDARY_CRITIC_ISSUE:{field}:{issue}"
                for field, issues in scoped_review.issues.items()
                for issue in issues
            )
        if retained_review is not None:
            parsed["reason_codes"].append(f"RETAINED_CONFLICT:{retained_review.score}")
            parsed["reason_codes"].extend(f"RETAINED_CONFLICT_ISSUE:{issue}" for issue in retained_review.issues)
        if record_scope is not None:
            parsed["reason_codes"].append(f"RECORD_SCOPE:{record_scope.score}")
            parsed["reason_codes"].extend(f"RECORD_SCOPE_ISSUE:{issue}" for issue in record_scope.issues)
        if boundary_review is not None:
            parsed["reason_codes"].extend(
                f"BOUNDARY_{field.upper()}:{score}" for field, score in boundary_review.scores.items()
            )
            parsed["reason_codes"].extend(f"BOUNDARY_CRITIC_ISSUE:{issue}" for issue in boundary_review.issues)
        if critic_result is not None:
            parsed["reason_codes"].append(f"SPAN_RECORD_BINDING_CRITIC:{critic_result[0]}")
            if critic_result[2] is not None:
                parsed["reason_codes"].append(f"SPAN_CRITIC_ISSUE:{critic_result[2]}")
    if parsed["primary_material_difference"] not in {"NONE", "UNRESOLVED"}:
        parsed["reason_codes"].append(f"MATERIAL_DELTA:{parsed['primary_material_difference']}")
    if parsed["dominant_overlap_source"] not in {"NONE", "UNRESOLVED"}:
        parsed["reason_codes"].append(f"OVERLAP_SOURCE:{parsed['dominant_overlap_source']}")
    if parsed["primary_risk_factor"] not in {"NONE", "UNRESOLVED"}:
        parsed["reason_codes"].append(f"PRIMARY_RISK:{parsed['primary_risk_factor']}")
    if span_ledger is not None and parsed["same_duplicate_group"] == "UNRESOLVED":
        parsed["reason_codes"].append("INSUFFICIENT_EVIDENCE")
        parsed["evidence"] = []
        evidence_events = []
    elif span_ledger is not None and parsed["relation_type"] == "EXACT":
        parsed["evidence"] = []
        evidence_events = []
    elif span_ledger is not None:
        evidence_citations = list(dict.fromkeys([*critic_citations, *span_citations]))
        parsed["evidence"], evidence_issue = _span_citation_evidence(payload, evidence_citations)
        require(
            evidence_issue is None,
            "LOCAL_NDD_OUTPUT_INVALID",
            "resolved span-ledger result lacks bilateral exact evidence",
            evidence_issue=evidence_issue,
        )
        evidence_events = [
            {
                "action": "derive_evidence_from_span_citations",
                "span_ids": evidence_citations,
            }
        ]
    elif parsed["same_duplicate_group"] == "UNRESOLVED":
        require(
            str(_score(value, "quote_evidence")).lower() == "insufficient",
            "LOCAL_NDD_OUTPUT_INVALID",
            "unresolved V3 results must mark quote evidence insufficient",
            field="quote_evidence",
        )
        parsed["reason_codes"].append("INSUFFICIENT_EVIDENCE")
        parsed["evidence"] = []
        evidence_events = []
    elif parsed["relation_type"] == "EXACT":
        require(
            str(_score(value, "quote_evidence")).lower() == "not_required",
            "LOCAL_NDD_OUTPUT_INVALID",
            "exact V3 results must mark quote evidence not required",
            field="quote_evidence",
        )
        parsed["evidence"] = []
        evidence_events = []
    else:
        require(
            str(_score(value, "quote_evidence")).lower() == "provided",
            "LOCAL_NDD_OUTPUT_INVALID",
            "resolved non-exact V3 results must provide exact evidence",
            field="quote_evidence",
        )
        parsed["evidence"], evidence_events = _extract_quote_evidence(value, payload, allow_fallback=False)
    return validate_judge_output(parsed), [
        {
            "action": "derive_same_duplicate_group",
            "source_fields": ["a_can_replace_b", "b_can_replace_a"],
        },
        *contract_events,
        *evidence_events,
    ]


def _adapt_ndd_judge_output_with_events(
    value: Any,
    schema_version: str,
    *,
    payload: dict[str, Any] | None,
    record_binding_critic: dict[str, Any] | None = None,
    record_binding_policy: str = "v1",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(isinstance(value, dict), "LOCAL_NDD_OUTPUT_INVALID", "NDD judge result must be an object")
    require(
        schema_version == JUDGE_SCHEMA,
        "LOCAL_NDD_OUTPUT_INVALID",
        "NDD adapter supports only the selected Judge output schema",
        schema_version=schema_version,
    )
    return _adapt_v3(value, payload, record_binding_critic, record_binding_policy)


def adapt_ndd_judge_output(
    value: Any,
    schema_version: str = JUDGE_SCHEMA,
    *,
    payload: dict[str, Any] | None = None,
    record_binding_critic: dict[str, Any] | None = None,
    record_binding_policy: str = "v1",
) -> dict[str, Any]:
    """Convert NDD score objects to the selected versioned judge contract."""

    parsed, _ = _adapt_ndd_judge_output_with_events(
        value,
        schema_version,
        payload=payload,
        record_binding_critic=record_binding_critic,
        record_binding_policy=record_binding_policy,
    )
    return parsed


def _safe_retry_feedback(error: Exception) -> dict[str, Any]:
    if isinstance(error, DedupEvaluationError):
        details = {
            key: item
            for key, item in error.issue.details.items()
            if key in {"actual_fields", "expected_fields", "extra_fields", "field", "missing_fields"}
        }
        if (
            error.issue.code == "JUDGE_CONSISTENCY_INVALID"
            and error.issue.message
            == "equivalent relations require bidirectional replacement without a major difference"
        ):
            details["required_consistency"] = (
                "EXACT, CANONICAL_EXACT, or NEAR_SURFACE requires "
                "a_can_replace_b=yes, b_can_replace_a=yes, and material_difference not major. "
                "For a faithful full translation, language difference alone never blocks replacement."
            )
        return {"code": error.issue.code, "message": error.issue.message, "details": details}
    return {
        "code": error.__class__.__name__,
        "message": "local NDD runtime failed before producing a valid batch",
    }


def safe_retry_feedback(error: Exception) -> dict[str, Any]:
    """Return credential-safe feedback suitable for a correction prompt."""
    return _safe_retry_feedback(error)
