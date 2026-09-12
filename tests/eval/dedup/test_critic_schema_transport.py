# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import critic_retention_v3 as critic
from eval.dedup.judging import critic_schema_transport as subject
from eval.dedup.validation import DedupEvaluationError


@pytest.mark.parametrize("mode", subject.MODES)
def test_projection_preserves_required_types_enums_and_original(mode):
    original = critic.response_schema()
    saved = deepcopy(original)
    projected = subject.project_schema(original, mode)
    assert original == saved
    assert projected["required"] == original["required"]
    assert projected["additionalProperties"] is False
    assert projected["properties"]["conflict"] == original["properties"]["conflict"]
    assert projected["properties"]["shared_anchor_ids"] == {"type": "array", "items": {"type": "string"}}
    assert projected["properties"]["a_context_span_id"] == (
        {"type": "string", "minLength": 1} if mode == "without_unique_items" else {"type": "string"}
    )


def test_unknown_projection_rejected():
    with pytest.raises(DedupEvaluationError):
        subject.project_schema(critic.response_schema(), "relax_everything")
