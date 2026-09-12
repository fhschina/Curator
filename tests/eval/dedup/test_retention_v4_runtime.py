# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging import retention_v4_runtime as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_retention_v4 import payload


def test_registered_experiment_binds_prompt_rubric_schema_and_adapter_without_default_switch():
    spec = subject.specification()
    assert spec["config"]["model_response_contract"] == subject.adapter.CONTRACT
    assert not spec["release_eligible"]
    with pytest.raises(DedupEvaluationError, match="RETENTION_V4_VERSION"):
        subject.specification("v0.6.2.33-exp1")


def test_main_and_critic_render_same_original_evidence_and_policy_without_labels_or_main_predictions():
    p = payload("完整政策。\n下一页", "完整政策。")
    p["payload_schema_version"] = "judge-visible-payload-v3"
    before = deepcopy(p)
    main = subject.messages(p)
    critic = subject.messages(p, stage="critic")
    assert main[1] == critic[1]
    assert "No same-specific-record gate" in main[0]["content"]
    assert "only remove" in critic[0]["content"]
    assert "完整政策" in main[1]["content"]
    assert "下一页" in main[1]["content"]
    assert p == before
    p["human_label"] = "YES"
    with pytest.raises(DedupEvaluationError):
        subject.messages(p)
