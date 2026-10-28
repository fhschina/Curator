# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import data_designer.config as dd
import pandas as pd
import pytest
from critics.models import Decision
from critics.spans import SpanPacket
from critics.subject import (
    SubjectApplyStage,
    SubjectReview,
    SubjectVerification,
    SubjectVerifierApplyStage,
    subject_column,
    verifier_column,
)
from policy_cases import aligned

from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage
from nemo_curator.tasks import DocumentBatch


def fixed_proposal():
    row = aligned(
        [
            {
                "pair_id": "fixed-subject-proof",
                "text_a": "Orion Corp is liable for damage.",
                "text_b": "Lyra Corp is liable for damage.",
            }
        ]
    )[0]
    packet = SpanPacket.from_record(row)
    own = {side: next(k for k, v in packet.spans.items() if v.kind == side + "_ONLY") for side in ("A", "B")}
    shared = next(k for k, v in packet.spans.items() if v.kind == "SHARED")
    review = SubjectReview(
        a_subject_span_id=own["A"],
        b_subject_span_id=own["B"],
        a_predicate_span_id=shared,
        b_predicate_span_id=shared,
        binding_type="LIABILITY_PARTY",
        target_relation="DIFFERENT",
        explanation="Different named liable parties",
    )
    return row, packet, review


def test_native_subject_schemas_and_skip_use_actual_dependency():
    for column, schema in [(subject_column("judge"), SubjectReview), (verifier_column("judge"), SubjectVerification)]:
        assert isinstance(column.skip, dd.SkipConfig)
        assert column.output_format == schema.model_json_schema()


def test_subject_cannot_extend_its_veto_authority_to_record_subjects():
    _, packet, review = fixed_proposal()
    proposal, _, reason = SubjectApplyStage._propose(
        review.model_copy(update={"binding_type": "RECORD_SUBJECT"}), packet
    )
    assert proposal is None
    assert reason == "SUBJECT_OBJECTION_OUTSIDE_SPECIALIST_AUTHORITY"


@pytest.mark.gpu
def test_live_fixed_subject_verification():
    endpoint = os.environ.get("RETENTION_QWEN_ENDPOINT")
    if not endpoint:
        pytest.skip("Set RETENTION_QWEN_ENDPOINT to the owned local Qwen service")
    row, packet, review = fixed_proposal()
    candidate, _, _ = SubjectApplyStage._propose(review, packet)
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="judge",
                model="Qwen/Qwen3.8-27B-FP8",
                provider="local",
                skip_health_check=True,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    max_tokens=6144,
                    temperature=0.2,
                    timeout=600,
                    max_parallel_requests=8,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ),
            )
        ]
    )
    builder.add_column(verifier_column("judge"))
    stage = DataDesignerStage(
        config_builder=builder,
        model_providers=[dd.ModelProvider(name="local", endpoint=endpoint, api_key="unused")],
        keep_failed_rows=True,
        row_id_column="pair_id",
    )
    row.update(subject_candidate=candidate, subject_verifier_should_run=True)
    result = (
        stage.process(DocumentBatch(dataset_name="fixed_proof", data=pd.DataFrame([row]))).to_pyarrow().to_pylist()[0]
    )
    answer = SubjectVerification.model_validate(result["subject_verifier_review"])
    assert answer.comparison == "SUPPORTED_DIFFERENT_NAMED_TARGETS"
    decision = Decision(
        a_can_replace_b="yes",
        b_can_replace_a="yes",
        relation_type="near_surface",
        material_difference="none",
        primary_material_difference="none",
        dominant_overlap_source="legal_policy_template",
        primary_risk_factor="none",
        confidence_tier="medium",
    )
    result.update(
        final_decision=decision.model_dump(),
        subject_action="PENDING_VERIFICATION",
        subject_reason="VERIFY_FIXED_SUBJECT_VETO",
    )
    action, _, final = SubjectVerifierApplyStage._apply(result)
    assert action == "REJECT_BOTH"
    assert final.a_can_replace_b == final.b_can_replace_a == "no"
