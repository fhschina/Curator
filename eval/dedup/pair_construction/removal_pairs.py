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

"""Step 5a: sample actual keeper-to-removed SUT decisions."""

from __future__ import annotations

from pathlib import Path

from eval.dedup.config import ProfileConfig
from eval.dedup.contracts import Action, Track, stable_record_id
from eval.dedup.validation import require


def sample_removal_pairs(
    outcomes_path: Path,
    *,
    profile: ProfileConfig,
    pair_seed: int,
    destination: Path,
) -> dict[str, int]:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "numpy and pyarrow are required for removal-pair sampling"
        raise RuntimeError(msg) from exc
    columns = [
        "evaluation_run_id",
        "sut_run_id",
        "doc_id",
        "predicted_cluster_key",
        "predicted_group_size",
        "action",
        "final_keeper_id",
        "language",
        "length_bucket",
        "token_count",
        "hostname",
        "canonical_url_v0",
    ]
    all_rows = pq.read_table(outcomes_path, columns=columns)
    removals = all_rows.filter(pc.equal(all_rows["action"], Action.REMOVE))
    frame_size = removals.num_rows
    sample_size = min(profile.removal_pair_budget, frame_size)
    rng = np.random.default_rng(pair_seed)
    selected_indices = np.sort(rng.choice(frame_size, size=sample_size, replace=False))
    selected = removals.take(pa.array(selected_indices))
    output_rows = []
    for row in selected.to_pylist():
        keeper_id = int(row["final_keeper_id"])
        removed_id = int(row["doc_id"])
        require(keeper_id != removed_id, "INVALID_REMOVAL_PAIR", "removed document points to itself as keeper")
        output_rows.append(
            {
                "evaluation_run_id": row["evaluation_run_id"],
                "sut_run_id": row["sut_run_id"],
                "track": Track.REMOVAL,
                "stream": "main",
                "pair_role": "removal_decision",
                "keeper_doc_id": keeper_id,
                "removed_doc_id": removed_id,
                "doc_i": keeper_id,
                "doc_j": removed_id,
                "predicted_cluster_key_i": row["predicted_cluster_key"],
                "predicted_cluster_key_j": row["predicted_cluster_key"],
                "selection_stratum": "main",
                "selection_rule": "uniform_without_replacement",
                "frame_size": frame_size,
                "selection_probability": sample_size / frame_size,
                "pair_seed": pair_seed,
                "provenance_record_id": stable_record_id("removal-main-v1", row["evaluation_run_id"], removed_id),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), destination, compression="zstd")
    return {"frame_size": frame_size, "rows": sample_size}
