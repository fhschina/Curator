# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, stateless record preparation and application for judge critics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.eval.llm_judge.critics.base import Critic


def _update_batch(
    batch: DocumentBatch,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
    columns: tuple[str, ...],
    drop: tuple[str, ...] = (),
) -> DocumentBatch:
    frame = batch.to_pandas().copy()
    # Arrow restores nested lists, nullable integers and skipped cells to Python values.
    updates = [transform(record) for record in batch.to_pyarrow().to_pylist()]
    for column in columns:
        frame[column] = pd.Series([update[column] for update in updates], index=frame.index, dtype=object)
    # Materialize a uniform struct schema before NDD reads nested records through DuckDB.
    data = pa.Table.from_pandas(frame.drop(columns=list(drop)), preserve_index=False)
    return DocumentBatch(
        dataset_name=batch.dataset_name,
        data=data,
        _metadata=batch._metadata,
        _stage_perf=batch._stage_perf,
    )


@dataclass
class CriticPrepareStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    critic: Critic

    def __post_init__(self) -> None:
        self.name = f"prepare_{self.critic.name}"
        self.resources = Resources(cpus=1, gpus=0)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(self.critic.prepared_columns)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        collisions = set(batch.get_columns()) & {*self.critic.output_columns, *self.critic.temporary_columns}
        if collisions:
            message = f"Critic {self.critic.name!r} would overwrite input columns: {sorted(collisions)}"
            raise ValueError(message)
        return _update_batch(batch, self.critic.prepare, self.critic.prepared_columns)


@dataclass
class CriticApplyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    critic: Critic

    def __post_init__(self) -> None:
        self.name = f"apply_{self.critic.name}"
        self.resources = Resources(cpus=1, gpus=0)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(self.critic.prepared_columns)

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(self.critic.output_columns)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        return _update_batch(batch, self.critic.apply, self.critic.applied_columns, self.critic.temporary_columns)
