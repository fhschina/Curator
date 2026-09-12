# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.judging import composite_runtime as subject
from eval.dedup.judging.composite_coverage import CONTRACT, composite_schema
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.validation import DedupEvaluationError, sha256_json
from tests.eval.dedup.test_composite_coverage import certificate


def test_main_and_critic_use_identical_new_policy_contract_model_and_rubric():
    config, _ = subject.load_composite_config(subject.CONFIG)
    builders = {
        stage: subject.build_composite_builder(subject.CONFIG, stage=stage, endpoint="http://127.0.0.1:1/v1")
        for stage in ("main", "critic")
    }
    assert builders["main"][0].model_configs == builders["critic"][0].model_configs
    assert builders["main"][1] == builders["critic"][1]
    policy = (subject.CONFIG.parent / config["policy_path"]).read_text()
    for builder, _ in builders.values():
        column = builder.get_column_config(COLUMN)
        assert column.output_format == composite_schema()
        assert column.system_prompt.count(policy) == 1
        assert json.dumps(config["rubric"], ensure_ascii=False) in column.prompt
        assert "SAME_SUBSTANTIVE_RECORD" not in column.system_prompt + column.prompt
        assert "DISTINCT_OR_UNBOUND_RECORDS" not in column.system_prompt + column.prompt
        assert CONTRACT in json.dumps(column.output_format)
        assert len(builder.get_column_configs()) == 1


@pytest.mark.parametrize("stage", ["main", "critic"])
def test_native_request_renders_originals_without_row_policy_or_label_injection(stage):
    main, payload, _, _ = certificate()
    packet = {
        "canonical_pair_id": "p",
        "payload": payload,
        "judge_payload_hash": sha256_json(payload),
        "repair_feedback": "",
        "semantic_policy": "LEAKED_OVERRIDE",
        "human_label": "SECRET_GOLD",
    }
    if stage == "critic":
        packet["proposed_main_text"] = subject.encode_main_review(main)
    messages = subject.composite_renderer(stage=stage)(packet)
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "LEAKED_OVERRIDE" not in rendered
    assert "SECRET_GOLD" not in rendered
    for side in ("a", "b"):
        assert payload[f"document_{side}"]["text"] in messages[1]["content"]
    schema = json.loads(messages[1]["content"].rsplit("<response_schema>", 1)[1].split("</response_schema>", 1)[0])
    assert schema == composite_schema()
    if stage == "critic":
        assert packet["proposed_main_text"] in messages[1]["content"]
    else:
        assert "<proposed_main_claim>" not in messages[1]["content"]


def test_critic_main_claim_is_explicitly_bound_and_cannot_be_a_legacy_response():
    main, payload, legacy, _ = certificate()
    row = {"payload": payload, "proposed_main_text": subject.encode_main_review(main)}
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_MAIN_LEAK"):
        subject.composite_record(row, "main")
    row["proposed_main_text"] = subject.encode_main_review(legacy)
    with pytest.raises(DedupEvaluationError, match="COMPOSITE_CONTRACT_INVALID"):
        subject.composite_record(row, "critic")
