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

from eval.dedup.analysis.metrics import wilson_interval
from eval.dedup.pair_construction.anchors import redistribute_group_quotas
from eval.dedup.pair_construction.outcomes import canonicalize_url_v0
from eval.dedup.pair_construction.retrieval.selection import _per_anchor_selection


def test_anchor_quota_redistribution() -> None:
    targets = {"size_2": 3, "size_3_5": 3, "size_6_20": 3, "size_21_plus": 3}
    capacities = {"size_2": 1, "size_3_5": 10, "size_6_20": 10, "size_21_plus": 10}
    allocation = redistribute_group_quotas(targets, capacities)
    assert sum(allocation.values()) == 12
    assert allocation["size_2"] == 1
    assert all(allocation[key] <= capacities[key] for key in capacities)


def test_url_normalization_is_conservative() -> None:
    hostname, canonical, ok = canonicalize_url_v0("HTTPS://Example.COM:443/A?B=Mixed#fragment")
    assert ok
    assert hostname == "example.com"
    assert canonical == "https://example.com/A?B=Mixed"
    assert canonicalize_url_v0("not a url") == (None, None, False)
    assert canonicalize_url_v0(None) == (None, None, True)


def test_wilson_interval() -> None:
    low, high = wilson_interval(90, 100)
    assert low == pytest.approx(0.8256, abs=1e-3)
    assert high == pytest.approx(0.9448, abs=1e-3)
    assert wilson_interval(0, 0) == (None, None)


def test_cross_group_quota_priorities_are_channel_interleaved() -> None:
    records = []
    for category, count in (("lexical_only", 4), ("semantic_only", 4), ("both", 2)):
        for rank in range(1, count + 1):
            candidate_id = len(records) + 100
            records.append(
                {
                    "anchor_id": 7,
                    "candidate_id": candidate_id,
                    "canonical_pair_id": f"pair-{candidate_id}",
                    "retriever_bitmask": category,
                    "lexical_rank": rank if category != "semantic_only" else None,
                    "semantic_rank": rank if category != "lexical_only" else None,
                    "jaccard": 1.0 / rank,
                    "cosine": 1.0 / rank if category != "lexical_only" else None,
                    "semantic_high_lexical_low": category == "semantic_only",
                    "best_normalized_rank": rank / 50,
                }
            )

    selected, event_rules = _per_anchor_selection(list(reversed(records)), top_k=50)
    by_pair_id = {row["canonical_pair_id"]: row for row in records}
    ordered_categories = [
        by_pair_id[pair_id]["retriever_bitmask"]
        for pair_id in sorted(
            selected,
            key=lambda pair_id: event_rules[(7, by_pair_id[pair_id]["candidate_id"])][1],
        )
    ]

    assert ordered_categories == [
        "lexical_only",
        "semantic_only",
        "both",
        "lexical_only",
        "semantic_only",
        "both",
        "lexical_only",
        "semantic_only",
        "lexical_only",
        "semantic_only",
    ]
