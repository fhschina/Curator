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

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eval.dedup.analysis.constraint_graph import build_constraint_graph
from eval.dedup.contracts import cp1_pair


def test_cannot_link_blocks_transitive_union(tmp_path: Path) -> None:
    pairs = [(1, 2, "YES"), (2, 3, "YES"), (1, 3, "NO")]
    candidate_rows = []
    result_rows = []
    for index, (left, right, answer) in enumerate(pairs):
        pair = cp1_pair(left, right)
        candidate_rows.append(
            {
                "canonical_pair_id": pair.canonical_pair_id,
                "doc_id_low": int(pair.doc_id_low),
                "doc_id_high": int(pair.doc_id_high),
            }
        )
        result_rows.append(
            {
                "canonical_pair_id": pair.canonical_pair_id,
                "judge_result_id": f"result-{index}",
                "judge_payload_hash": f"payload-{index}",
                "same_duplicate_group": answer,
                "confidence": 0.99,
            }
        )
    candidates = tmp_path / "candidate.parquet"
    results = tmp_path / "results.jsonl"
    pq.write_table(pa.Table.from_pylist(candidate_rows), candidates)
    results.write_text("".join(json.dumps(row) + "\n" for row in result_rows))
    counts = build_constraint_graph(
        results,
        candidates,
        must_links_destination=tmp_path / "must.parquet",
        cannot_links_destination=tmp_path / "cannot.parquet",
        components_destination=tmp_path / "components.parquet",
        conflicts_destination=tmp_path / "conflicts.parquet",
    )
    assert counts["must_links"] == 1
    assert counts["conflicts"] == 1
