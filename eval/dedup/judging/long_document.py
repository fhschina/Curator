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

"""Deterministic evidence-window selection for oversized judge payloads."""

from __future__ import annotations

import re
from typing import Any

from eval.dedup.config import JudgeConfig
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.pair_construction.retrieval.lexical import char_shingles


def _windows(text: str, counter: TokenCounter, config: JudgeConfig) -> list[dict[str, Any]]:
    token_ids, offsets = counter.encode_with_offsets(text)
    if not token_ids:
        return [{"start_token": 0, "end_token": 0, "start_char": 0, "end_char": 0, "text": ""}]
    step = config.window_tokens - config.window_overlap_tokens
    windows = []
    for start in range(0, len(token_ids), step):
        end = min(len(token_ids), start + config.window_tokens)
        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]
        windows.append(
            {
                "start_token": start,
                "end_token": end,
                "start_char": start_char,
                "end_char": end_char,
                "text": text[start_char:end_char],
            }
        )
        if end == len(token_ids):
            break
    return windows


def _window_terms(text: str) -> set[str]:
    normalized = " ".join(re.findall(r"\w+", text.lower()))
    return char_shingles(normalized, 5)


def prepare_long_document_evidence(
    text_a: str,
    text_b: str,
    *,
    counter: TokenCounter,
    config: JudgeConfig,
) -> dict[str, Any]:
    """Return full texts when they fit, otherwise deterministic aligned windows."""

    counts = counter.count_many([text_a, text_b])
    if sum(counts) <= config.max_visible_tokens:
        return {
            "truncated": False,
            "token_counts": {"A": counts[0], "B": counts[1]},
            "text_a": text_a,
            "text_b": text_b,
            "windows": [],
        }
    windows_a = _windows(text_a, counter, config)
    windows_b = _windows(text_b, counter, config)
    selected_a = {0, len(windows_a) - 1}
    selected_b = {0, len(windows_b) - 1}
    consumed = sum(
        windows[index]["end_token"] - windows[index]["start_token"]
        for windows, indices in ((windows_a, selected_a), (windows_b, selected_b))
        for index in indices
    )
    scored = []
    terms_a = [_window_terms(item["text"]) for item in windows_a]
    terms_b = [_window_terms(item["text"]) for item in windows_b]
    for index_a, left in enumerate(terms_a):
        for index_b, right in enumerate(terms_b):
            union = len(left | right)
            score = len(left & right) / union if union else 1.0
            scored.append((-score, index_a, index_b))
    for _, index_a, index_b in sorted(scored):
        additional = 0
        if index_a not in selected_a:
            additional += windows_a[index_a]["end_token"] - windows_a[index_a]["start_token"]
        if index_b not in selected_b:
            additional += windows_b[index_b]["end_token"] - windows_b[index_b]["start_token"]
        if additional == 0:
            continue
        if consumed + additional > config.max_visible_tokens:
            continue
        selected_a.add(index_a)
        selected_b.add(index_b)
        consumed += additional
    visible = []
    for side, windows, indices in (("A", windows_a, selected_a), ("B", windows_b, selected_b)):
        for index in sorted(indices):
            visible.append({"side": side, **windows[index]})
    return {
        "truncated": True,
        "token_counts": {"A": counts[0], "B": counts[1]},
        "text_a": None,
        "text_b": None,
        "windows": visible,
    }
