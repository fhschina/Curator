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

"""Shared, stateless record preparation and application for judge critics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from .coverage import CoverageCritic


@dataclass
class CriticPrepareStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    critic: CoverageCritic

    def __post_init__(self) -> None:
        self.name = f"prepare_{self.critic.name}"
        self.resources = Resources(cpus=1, gpus=0)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", "text_a", "text_b", "semantic_diff", "truncated", self.critic.source_judge]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(self.critic.prepared_columns)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        collisions = set(batch.get_columns()) & {*self.critic.output_columns, *self.critic.temporary_columns}
        if collisions:
            message = f"Critic {self.critic.name!r} would overwrite input columns: {sorted(collisions)}"
            raise ValueError(message)
        frame = batch.to_pandas().copy()
        # Arrow restores nested lists, nullable integers and skipped cells to Python values.
        updates = [self.critic.prepare(record) for record in batch.to_pyarrow().to_pylist()]
        for column in self.critic.prepared_columns:
            frame[column] = pd.Series([update[column] for update in updates], index=frame.index, dtype=object)
        # Materialize a uniform struct schema before NDD reads nested records through DuckDB.
        data = pa.Table.from_pandas(frame, preserve_index=False)
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=data,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class CriticApplyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    critic: CoverageCritic

    def __post_init__(self) -> None:
        self.name = f"apply_{self.critic.name}"
        self.resources = Resources(cpus=1, gpus=0)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pair_id", *self.critic.prepared_columns, "coverage_review"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(self.critic.output_columns)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        frame = batch.to_pandas().copy()
        updates = [self.critic.apply(record) for record in batch.to_pyarrow().to_pylist()]
        for column in self.critic.applied_columns:
            frame[column] = pd.Series([update[column] for update in updates], index=frame.index, dtype=object)
        data = pa.Table.from_pandas(frame.drop(columns=list(self.critic.temporary_columns)), preserve_index=False)
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=data,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
