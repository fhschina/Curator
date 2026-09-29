# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_curator.tasks import DocumentBatch
from tutorials.eval.dedup.critics.coverage import CoverageCritic
from tutorials.eval.dedup.critics.stages import CriticApplyStage, CriticPrepareStage


def test_stages_preserve_records_and_metadata(pair: dict[str, Any], review: dict[str, Any]) -> None:
    critic = CoverageCritic("pair_semantic_judgment")
    batch = DocumentBatch(data=pd.DataFrame([pair]), dataset_name="pairs", _metadata={"source": "fixture"})
    prepared = CriticPrepareStage(critic).process(batch)
    prepared.data = prepared.to_pandas()
    prepared.data["coverage_review"] = [review]
    result = CriticApplyStage(critic).process(prepared)
    row = result.to_pyarrow().to_pylist()[0]
    for key, value in pair.items():
        # Arrow may materialize absent keys as null in heterogeneous span structs.
        if key != "semantic_diff":
            assert row[key] == value
    assert row["coverage_action"] == "KEEP_MAIN"
    assert result.dataset_name == batch.dataset_name
    assert result._metadata == batch._metadata
    assert result._stage_perf is batch._stage_perf
    assert not set(critic.temporary_columns) & set(result.get_columns())
    with pytest.raises(ValueError, match="overwrite"):
        CriticPrepareStage(critic).process(result)


def test_apply_does_not_treat_failed_generation_as_skip(pair: dict[str, Any]) -> None:
    critic = CoverageCritic("pair_semantic_judgment")
    prepared = CriticPrepareStage(critic).process(DocumentBatch(data=pd.DataFrame([pair]), dataset_name="pairs"))
    prepared.data = prepared.to_pandas()
    prepared.data["coverage_review"] = None
    with pytest.raises(ValueError, match="pair-1"):
        CriticApplyStage(critic).process(prepared)
