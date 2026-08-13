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

"""Step 4: reproducible group-aware anchor sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.dedup.config import ProfileConfig
from eval.dedup.validation import require

GROUP_BUCKETS = ("size_2", "size_3_5", "size_6_20", "size_21_plus")


def group_size_bucket(size: int) -> str:
    if size == 2:
        return "size_2"
    if size <= 5:
        return "size_3_5"
    if size <= 20:
        return "size_6_20"
    return "size_21_plus"


def redistribute_group_quotas(targets: dict[str, int], capacities: dict[str, int]) -> dict[str, int]:
    """Redistribute sparse grouped quotas proportionally with deterministic remainders."""

    allocation = {bucket: min(targets[bucket], capacities[bucket]) for bucket in GROUP_BUCKETS}
    deficit = sum(targets.values()) - sum(allocation.values())
    while deficit:
        available = {bucket: capacities[bucket] - allocation[bucket] for bucket in GROUP_BUCKETS}
        total_available = sum(max(0, value) for value in available.values())
        if total_available <= 0:
            break
        exact = {bucket: deficit * max(0, available[bucket]) / total_available for bucket in GROUP_BUCKETS}
        added = 0
        for bucket in GROUP_BUCKETS:
            increment = min(available[bucket], int(exact[bucket]))
            allocation[bucket] += increment
            added += increment
        remaining = deficit - added
        for bucket in sorted(GROUP_BUCKETS, key=lambda item: (-(exact[item] % 1), item)):
            if remaining == 0:
                break
            if allocation[bucket] < capacities[bucket]:
                allocation[bucket] += 1
                remaining -= 1
        if remaining == deficit:
            break
        deficit = remaining
    return allocation


def sample_anchors(
    outcomes_path: Path,
    *,
    profile: ProfileConfig,
    anchor_seed: int,
    destination: Path,
) -> dict[str, int]:
    """Write the frozen ``anchors.parquet`` sample without loading text metadata for all 10M rows."""

    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "numpy and pyarrow are required for anchor sampling"
        raise RuntimeError(msg) from exc
    selection_columns = ["doc_id", "predicted_group_id", "predicted_group_size"]
    table = pq.read_table(outcomes_path, columns=selection_columns)
    doc_ids = table["doc_id"].to_numpy(zero_copy_only=False)
    group_ids = table["predicted_group_id"].to_numpy(zero_copy_only=False)
    group_sizes = table["predicted_group_size"].to_numpy(zero_copy_only=False)
    rng = np.random.default_rng(anchor_seed)
    singleton_ids = doc_ids[group_ids == -1]
    singleton_target = profile.anchor_quotas["singleton"]
    require(len(singleton_ids) >= singleton_target, "SPARSE_ANCHOR_STRATUM", "singleton anchor quota cannot be filled")
    selected_singletons = rng.choice(singleton_ids, size=singleton_target, replace=False)

    grouped_mask = group_ids != -1
    grouped_groups = group_ids[grouped_mask]
    grouped_sizes = group_sizes[grouped_mask]
    unique_groups, first_indices = np.unique(grouped_groups, return_index=True)
    unique_sizes = grouped_sizes[first_indices]
    bucket_groups = {
        bucket: unique_groups[[group_size_bucket(int(size)) == bucket for size in unique_sizes]]
        for bucket in GROUP_BUCKETS
    }
    capacities = {bucket: len(values) for bucket, values in bucket_groups.items()}
    targets = {bucket: profile.anchor_quotas[bucket] for bucket in GROUP_BUCKETS}
    allocation = redistribute_group_quotas(targets, capacities)
    require(
        sum(allocation.values()) == sum(targets.values()),
        "SPARSE_ANCHOR_STRATUM",
        "grouped anchor quotas cannot be filled after redistribution",
        capacities=capacities,
        requested=targets,
    )
    selected_specs: list[tuple[int, str, float]] = [
        (int(doc_id), "singleton", singleton_target / len(singleton_ids)) for doc_id in selected_singletons.tolist()
    ]
    grouped_doc_ids = doc_ids[grouped_mask]
    for bucket in GROUP_BUCKETS:
        chosen_groups = rng.choice(bucket_groups[bucket], size=allocation[bucket], replace=False)
        for group_id in chosen_groups.tolist():
            members = grouped_doc_ids[grouped_groups == group_id]
            member = int(rng.choice(members))
            selected_specs.append((member, bucket, allocation[bucket] / capacities[bucket] / len(members)))
    selected_ids = [item[0] for item in selected_specs]
    detail_columns = [
        "evaluation_run_id",
        "sut_run_id",
        "doc_id",
        "predicted_group_id",
        "predicted_cluster_key",
        "predicted_group_size",
        "action",
        "language",
        "length_bucket",
        "hostname",
    ]
    details = pq.read_table(outcomes_path, columns=detail_columns, filters=[("doc_id", "in", selected_ids)])
    row_by_doc = {int(row["doc_id"]): row for row in details.to_pylist()}
    require(len(row_by_doc) == len(selected_ids), "ANCHOR_JOIN_INCOMPLETE", "selected anchors could not be reloaded")
    selected_rows: list[dict[str, Any]] = []
    for doc_id, bucket, probability in selected_specs:
        selected_rows.append(
            {
                **row_by_doc[doc_id],
                "anchor_id": doc_id,
                "anchor_stratum": bucket,
                "group_size_bucket": bucket,
                "anchor_selection_probability": probability,
                "anchor_seed": anchor_seed,
            }
        )
    require(len(selected_rows) == profile.anchor_count, "ANCHOR_COUNT_MISMATCH", "anchor output count is invalid")
    require(
        len({row["anchor_id"] for row in selected_rows}) == len(selected_rows),
        "DUPLICATE_ANCHOR",
        "anchor IDs are not unique",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected_rows), destination, compression="zstd")
    return {"rows": len(selected_rows), **{f"quota_{key}": value for key, value in allocation.items()}}
