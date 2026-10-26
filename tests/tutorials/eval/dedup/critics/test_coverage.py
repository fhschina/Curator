# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import os

import data_designer.config as dd
import pandas as pd
import pytest
from critics.coverage import CoverageApplyStage, CoveragePrepareStage, CoverageReview, coverage_column
from critics.spans import SpanPacket
from policy_cases import aligned, controlled

from nemo_curator.tasks import DocumentBatch


def test_native_structured_schema_and_skip_are_real_dependency_objects():
    column = coverage_column("judge")
    assert isinstance(column, dd.LLMStructuredColumnConfig)
    assert isinstance(column.skip, dd.SkipConfig)
    assert column.output_format == CoverageReview.model_json_schema()


def test_wrong_side_and_unknown_span_cannot_prove_a_veto():
    row = aligned(controlled()[:1])[0]
    packet = SpanPacket.from_record(row)
    a = next(k for k, v in packet.spans.items() if v.kind == "A_ONLY")
    b = next(k for k, v in packet.spans.items() if v.kind == "B_ONLY")
    review = CoverageReview(
        a_loss_span_id=b,
        b_loss_span_id="",
        a_context_span_id=a,
        b_context_span_id=b,
        conflict="NONE",
        overlap_basis="RETAINED_CONTENT",
        shared_anchor_ids=[],
        explanation="Different",
    )
    with pytest.raises(ValueError, match="wrong side"):
        CoverageApplyStage._check_proof(review, packet)
    with pytest.raises(ValueError, match="unknown span"):
        CoverageApplyStage._check_proof(review.model_copy(update={"a_loss_span_id": "A999"}), packet)


def test_coverage_snapshot_survives_later_final_changes():
    row = aligned(controlled()[:1])[0]
    value = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
    }
    row["main"] = {k: {"score": v} for k, v in value.items()}
    batch = CoveragePrepareStage("main").process(DocumentBatch(dataset_name="test", data=pd.DataFrame([row])))
    data = batch.to_pyarrow().to_pylist()
    data[0]["coverage_review"] = None
    batch = CoverageApplyStage("main").process(DocumentBatch(dataset_name="test", data=pd.DataFrame(data)))
    result = batch.to_pyarrow().to_pylist()[0]
    frozen = copy.deepcopy(result["coverage_decision"])
    result["final_decision"]["a_can_replace_b"] = "no"
    assert result["coverage_decision"] == frozen
    assert result["coverage_action"] == "UNVALIDATED_KEEP_MAIN"


@pytest.mark.gpu
def test_live_coverage_distinguishes_empty_navigation_from_facts_and_obligations():
    endpoint = os.environ.get("RETENTION_QWEN_ENDPOINT")
    if not endpoint:
        pytest.skip("Set RETENTION_QWEN_ENDPOINT to the owned local Qwen GPU service")
    from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage

    article = "The museum opens at 09:00 on Monday. Admission costs 12 dollars."
    policy = "This site uses cookies to remember display preferences. You can disable them in Settings."
    sources = [
        {
            "pair_id": "empty_heading",
            "text_a": article + "\nArtists/Participants\nAll years\nChoose a city",
            "text_b": article,
        },
        {
            "pair_id": "populated_record",
            "text_a": article + " Participants: Ada Roy contributed three ceramic works in 2024.",
            "text_b": article,
        },
        {
            "pair_id": "new_obligation",
            "text_a": policy + " Explicit consent is required before analytics cookies are set.",
            "text_b": policy,
        },
    ]
    value = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
    }
    rows = [{**r, "main": {key: {"score": score} for key, score in value.items()}} for r in aligned(sources)]
    batch = CoveragePrepareStage("main").process(
        DocumentBatch(dataset_name="coverage_boundary", data=pd.DataFrame(rows))
    )
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="judge",
                model="Qwen/Qwen3.8-27B-FP8",
                provider="local",
                skip_health_check=True,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.2,
                    max_tokens=6144,
                    timeout=600,
                    max_parallel_requests=8,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ),
            )
        ]
    )
    builder.add_column(coverage_column("judge"))
    batch = DataDesignerStage(
        config_builder=builder,
        model_providers=[dd.ModelProvider(name="local", endpoint=endpoint, api_key="unused")],
        keep_failed_rows=True,
        row_id_column="pair_id",
    ).process(batch)
    results = CoverageApplyStage("main").process(batch).to_pyarrow().to_pylist()
    assert {r["pair_id"] for r in results} == {r["pair_id"] for r in sources}
    for row in results:
        assert "UNVALIDATED" not in row["coverage_action"]
        assert row["final_decision"]["a_can_replace_b"] == "yes"
        assert row["final_decision"]["b_can_replace_a"] == ("yes" if row["pair_id"] == "empty_heading" else "no")
