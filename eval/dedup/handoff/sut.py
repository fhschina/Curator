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

"""Read and validate the frozen fuzzy-dedup SUT outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.dedup.config import EvaluationConfig
from eval.dedup.validation import require


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "numpy and pyarrow are required to validate SUT artifacts"
        raise RuntimeError(msg) from exc
    return np, pq


@dataclass(frozen=True, slots=True)
class SutArrays:
    grouped_doc_ids: Any
    grouped_group_ids: Any
    group_ids: Any
    group_sizes: Any
    group_keepers: Any
    removal_ids: Any

    @property
    def grouped_document_count(self) -> int:
        return len(self.grouped_doc_ids)

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    @property
    def removal_count(self) -> int:
        return len(self.removal_ids)

    def lookup(self, doc_ids: Any) -> tuple[Any, Any, Any, Any]:
        """Return raw group, group size, keeper and removal mask for document IDs."""

        import numpy as np

        positions = np.searchsorted(self.grouped_doc_ids, doc_ids)
        in_bounds = positions < len(self.grouped_doc_ids)
        found = np.zeros(len(doc_ids), dtype=bool)
        found[in_bounds] = self.grouped_doc_ids[positions[in_bounds]] == doc_ids[in_bounds]
        raw_group = np.full(len(doc_ids), -1, dtype=np.int64)
        raw_group[found] = self.grouped_group_ids[positions[found]]
        group_positions = np.searchsorted(self.group_ids, raw_group[found])
        sizes = np.ones(len(doc_ids), dtype=np.int64)
        keepers = doc_ids.copy()
        sizes[found] = self.group_sizes[group_positions]
        keepers[found] = self.group_keepers[group_positions]
        removal_positions = np.searchsorted(self.removal_ids, doc_ids)
        is_removal = np.zeros(len(doc_ids), dtype=bool)
        removal_in_bounds = removal_positions < len(self.removal_ids)
        is_removal[removal_in_bounds] = (
            self.removal_ids[removal_positions[removal_in_bounds]] == doc_ids[removal_in_bounds]
        )
        return raw_group, sizes, keepers, is_removal


def load_sut_arrays(config: EvaluationConfig, *, groups_path: Path, removals_path: Path) -> SutArrays:
    np, pq = _dependencies()
    require(
        groups_path.is_file(), "SUT_GROUPS_NOT_FOUND", "duplicate groups artifact is missing", path=str(groups_path)
    )
    require(
        removals_path.is_file(), "SUT_REMOVALS_NOT_FOUND", "removal IDs artifact is missing", path=str(removals_path)
    )
    import pyarrow as pa

    groups_table = pq.read_table(groups_path)
    removals_table = pq.read_table(removals_path)
    require(
        config.retrieval.backend == "fixture_cpu"
        or (
            {"_curator_dedup_id", "_duplicate_group_id"}.issubset(groups_table.column_names)
            and groups_table.schema.field("_curator_dedup_id").type == pa.int64()
            and groups_table.schema.field("_duplicate_group_id").type == pa.int64()
            and not groups_table.schema.field("_curator_dedup_id").nullable
            and not groups_table.schema.field("_duplicate_group_id").nullable
        ),
        "SUT_GROUP_SCHEMA_MISMATCH",
        "duplicate groups IDs must be non-nullable int64",
    )
    require(
        config.retrieval.backend == "fixture_cpu"
        or (
            "_curator_dedup_id" in removals_table.column_names
            and removals_table.schema.field("_curator_dedup_id").type == pa.int64()
            and not removals_table.schema.field("_curator_dedup_id").nullable
        ),
        "SUT_REMOVAL_SCHEMA_MISMATCH",
        "removal IDs must be non-nullable int64",
    )
    required_group_fields = {"_curator_dedup_id", "_duplicate_group_id"}
    require(
        required_group_fields.issubset(groups_table.column_names),
        "SUT_GROUP_SCHEMA_MISMATCH",
        "duplicate groups artifact is missing required fields",
    )
    require(
        "_curator_dedup_id" in removals_table.column_names,
        "SUT_REMOVAL_SCHEMA_MISMATCH",
        "removal artifact is missing _curator_dedup_id",
    )
    doc_ids = groups_table["_curator_dedup_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    group_ids_for_docs = (
        groups_table["_duplicate_group_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    )
    removal_ids = removals_table["_curator_dedup_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    require(
        len(doc_ids) == config.dataset.expected_grouped_documents,
        "SUT_GROUP_COUNT_MISMATCH",
        "grouped document count is invalid",
    )
    require(
        len(removal_ids) == config.dataset.expected_removals, "SUT_REMOVAL_COUNT_MISMATCH", "removal count is invalid"
    )
    require(
        len(np.unique(doc_ids)) == len(doc_ids),
        "DUPLICATE_SUT_DOC_ID",
        "a document appears in more than one SUT group",
    )
    require(len(np.unique(removal_ids)) == len(removal_ids), "DUPLICATE_REMOVAL_ID", "removal IDs are not unique")
    require(
        doc_ids.min() >= 0 and doc_ids.max() < config.dataset.expected_rows,
        "SUT_ID_OUT_OF_RANGE",
        "grouped ID is outside corpus",
    )
    require(
        removal_ids.min() >= 0 and removal_ids.max() < config.dataset.expected_rows,
        "SUT_ID_OUT_OF_RANGE",
        "removal ID is outside corpus",
    )
    doc_sort = np.argsort(doc_ids, kind="stable")
    sorted_docs = doc_ids[doc_sort]
    sorted_doc_groups = group_ids_for_docs[doc_sort]
    removal_ids = np.sort(removal_ids)
    removal_positions = np.searchsorted(sorted_docs, removal_ids)
    require(
        np.all(removal_positions < len(sorted_docs)) and np.array_equal(sorted_docs[removal_positions], removal_ids),
        "REMOVAL_JOIN_INCOMPLETE",
        "one or more removal IDs are absent from duplicate groups",
    )
    unique_groups, inverse, group_sizes = np.unique(group_ids_for_docs, return_inverse=True, return_counts=True)
    removed_mask = np.isin(doc_ids, removal_ids, assume_unique=True)
    removed_per_group = np.bincount(inverse[removed_mask], minlength=len(unique_groups))
    require(
        bool(np.all(group_sizes >= 2)),
        "INVALID_DUPLICATE_GROUP_SIZE",
        "every fuzzy duplicate group must contain at least two documents",
    )
    keeper_counts = group_sizes - removed_per_group
    invalid = unique_groups[keeper_counts != 1]
    require(
        len(invalid) == 0,
        "AMBIGUOUS_GROUP_KEEPER",
        "every non-singleton group must contain exactly one logical keeper",
        invalid_group_count=len(invalid),
        sample=invalid[:20].tolist(),
    )
    keeper_groups = group_ids_for_docs[~removed_mask]
    keeper_docs = doc_ids[~removed_mask]
    keeper_order = np.argsort(keeper_groups)
    group_keepers = keeper_docs[keeper_order]
    require(
        len(unique_groups) == config.dataset.expected_groups,
        "SUT_GROUP_CARDINALITY_MISMATCH",
        "group count is invalid",
    )
    return SutArrays(
        grouped_doc_ids=sorted_docs,
        grouped_group_ids=sorted_doc_groups,
        group_ids=unique_groups,
        group_sizes=group_sizes.astype(np.int64, copy=False),
        group_keepers=group_keepers.astype(np.int64, copy=False),
        removal_ids=removal_ids,
    )
