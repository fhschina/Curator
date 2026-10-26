# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pandas as pd
import pytest
from policy_cases import load_module

from nemo_curator.tasks import DocumentBatch


@pytest.mark.parametrize("missing", [None, float("nan"), "", "   "])
def test_missing_text_does_not_crash_and_is_never_complete(missing: object):
    module = load_module("4_span_alignment.py")
    flags = module.TokenizerStage()._tokenize_pair(missing, "visible", max_chars=6000)
    assert flags["truncated_a"]
    batch = DocumentBatch(
        dataset_name="missing",
        data=pd.DataFrame([{"pair_id": "missing", "text_a": missing, "text_b": "visible"}], dtype=object),
    )
    for stage in (module.TokenizerStage(), module.SpanAlignmentStage(), module.SpanChunkingStage()):
        batch = stage.process(batch)
    row = batch.to_pandas().to_dict("records")[0]
    assert row["truncated"]
    assert row["semantic_diff"]["truncated_a"]
    if isinstance(missing, float):
        assert math.isnan(row["text_a"])
