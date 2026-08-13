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

"""Step 7: build an auditable partial judged constraint graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.dedup.contracts import DuplicateAnswer, stable_record_id


class UnionFind:
    def __init__(self, nodes: set[int]) -> None:
        self.parent = {node: node for node in nodes}
        self.members = {node: {node} for node in nodes}

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: int, right: int) -> int:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return root_left
        if len(self.members[root_left]) < len(self.members[root_right]):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.members[root_left].update(self.members.pop(root_right))
        return root_left


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_constraint_graph(
    judge_results_path: Path,
    candidate_pairs_path: Path,
    *,
    must_links_destination: Path,
    cannot_links_destination: Path,
    components_destination: Path,
    conflicts_destination: Path,
    confidence_threshold: float = 0.8,
) -> dict[str, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for constraint graph construction"
        raise RuntimeError(msg) from exc
    candidates = {row["canonical_pair_id"]: row for row in pq.read_table(candidate_pairs_path).to_pylist()}
    results = _read_jsonl(judge_results_path)
    committed = [row for row in results if float(row["confidence"]) >= confidence_threshold]
    must_rows = []
    cannot_rows = []
    nodes: set[int] = set()
    for result in committed:
        candidate = candidates[result["canonical_pair_id"]]
        left = int(candidate["doc_id_low"])
        right = int(candidate["doc_id_high"])
        nodes.update((left, right))
        edge = {
            "canonical_pair_id": result["canonical_pair_id"],
            "doc_id_low": left,
            "doc_id_high": right,
            "judge_result_id": result["judge_result_id"],
            "judge_payload_hash": result["judge_payload_hash"],
            "confidence": result["confidence"],
        }
        if result["same_duplicate_group"] == DuplicateAnswer.YES:
            must_rows.append(edge)
        elif result["same_duplicate_group"] == DuplicateAnswer.NO:
            cannot_rows.append(edge)
    cannot_pairs = {(row["doc_id_low"], row["doc_id_high"]): row for row in cannot_rows}
    union_find = UnionFind(nodes)
    conflicts = []
    accepted_must = []
    for edge in sorted(must_rows, key=lambda row: (-row["confidence"], row["canonical_pair_id"])):
        root_left = union_find.find(edge["doc_id_low"])
        root_right = union_find.find(edge["doc_id_high"])
        cross_conflicts = []
        if root_left != root_right:
            for left in union_find.members[root_left]:
                for right in union_find.members[root_right]:
                    key = tuple(sorted((left, right)))
                    if key in cannot_pairs:
                        cross_conflicts.append(cannot_pairs[key])
        if cross_conflicts:
            conflicts.append(
                {
                    "conflict_id": stable_record_id("constraint-conflict-v1", edge["canonical_pair_id"]),
                    "proposed_must_link_pair_id": edge["canonical_pair_id"],
                    "proposed_judge_result_id": edge["judge_result_id"],
                    "cannot_link_pair_ids": [item["canonical_pair_id"] for item in cross_conflicts],
                    "status": "human_qa_required",
                }
            )
            continue
        union_find.union(edge["doc_id_low"], edge["doc_id_high"])
        accepted_must.append(edge)
    component_rows = []
    roots = {node: union_find.find(node) for node in nodes}
    component_min = {
        root: min(node for node, candidate_root in roots.items() if candidate_root == root)
        for root in set(roots.values())
    }
    for node in sorted(nodes):
        component_rows.append({"doc_id": node, "partial_reference_component_id": component_min[roots[node]]})

    def write(rows: list[dict[str, Any]], destination: Path, schema: Any) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=schema) if rows else pa.Table.from_batches([], schema=schema)
        pq.write_table(table, destination, compression="zstd")

    edge_schema = pa.schema(
        [
            ("canonical_pair_id", pa.string()),
            ("doc_id_low", pa.int64()),
            ("doc_id_high", pa.int64()),
            ("judge_result_id", pa.string()),
            ("judge_payload_hash", pa.string()),
            ("confidence", pa.float64()),
        ]
    )
    write(accepted_must, must_links_destination, edge_schema)
    write(cannot_rows, cannot_links_destination, edge_schema)
    write(
        component_rows,
        components_destination,
        pa.schema([("doc_id", pa.int64()), ("partial_reference_component_id", pa.int64())]),
    )
    write(
        conflicts,
        conflicts_destination,
        pa.schema(
            [
                ("conflict_id", pa.string()),
                ("proposed_must_link_pair_id", pa.string()),
                ("proposed_judge_result_id", pa.string()),
                ("cannot_link_pair_ids", pa.list_(pa.string())),
                ("status", pa.string()),
            ]
        ),
    )
    unresolved = len(results) - len(committed)
    return {
        "nodes": len(nodes),
        "must_links": len(accepted_must),
        "cannot_links": len(cannot_rows),
        "components": len(set(component_min.values())),
        "conflicts": len(conflicts),
        "uncommitted_results": unresolved,
    }
