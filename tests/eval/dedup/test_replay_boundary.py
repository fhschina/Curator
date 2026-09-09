# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.dedup.analysis.replay_boundary import replay_run
from eval.dedup.contracts import canonical_json_bytes
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.validation import DedupEvaluationError


def test_replay_rejects_a_changed_source_result_before_reading_raw_payloads(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "judge_results.jsonl").write_text("{}\n")
    (tmp_path / "run_complete.json").write_text(json.dumps({"results_sha256": "incorrect"}))
    (tmp_path / "run_manifest.json").write_text("{}")
    with pytest.raises(DedupEvaluationError, match="REPLAY_RESULT_CHANGED"):
        replay_run(tmp_path, tmp_path / "labels.csv")


def test_replay_uses_pinned_raw_scores_and_original_integer_offset_payload(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "runtime" / "block" / "attempt_01" / "output"
    data.mkdir()
    output.mkdir(parents=True)
    values = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "exact",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "none",
        "confidence_tier": "medium",
        "span_content_profile_a": "substantive_main",
        "span_content_profile_b": "substantive_main",
        "span_shared_basis": "verified_substantive_record",
        "span_a_delta": "none",
        "span_b_delta": "none",
        "span_hard_conflict": "none",
        "span_translation_status": "not_translation",
    }
    main = {
        field: {"score": score, "reasoning": "S001 is the complete identical record."}
        for field, score in values.items()
    }
    critic = {
        field: {"score": score, "reasoning": "S001 is the complete identical record."}
        for field, score in {
            "non_main_delta_subtype": "not_applicable",
            "translation_delta_direction": "not_translation",
            "record_binding_verdict": "semantic_equivalence",
        }.items()
    }
    payload = {
        "document_a": {"text": "record"},
        "document_b": {"text": "record"},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": {
            "status": "COMPLETE",
            "span_counts": {"SHARED": 1, "A_ONLY": 0, "B_ONLY": 0},
            "spans": [
                {
                    "span_id": "S001",
                    "kind": "SHARED",
                    "a_start_char": 0,
                    "a_end_char": 6,
                    "a_text": "record",
                    "b_start_char": 0,
                    "b_end_char": 6,
                    "b_text": "record",
                }
            ],
        },
    }
    result = adapt_ndd_judge_output(
        main, "dedup-judge-output-v3", payload=payload, record_binding_critic=critic, record_binding_policy="v4"
    )
    result.update(
        {
            "canonical_pair_id": "pair",
            "provider_response_sha256": hashlib.sha256(
                canonical_json_bytes({"semantic_judge": main, "record_binding_critic": critic})
            ).hexdigest(),
        }
    )
    raw_path = output / "raw.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "canonical_pair_id": "pair",
                "qwen_dedup_semantic_judge": main,
                "qwen_dedup_record_binding_critic": critic,
            }
        )
        + "\n"
    )
    results_path, payload_path = data / "judge_results.jsonl", data / "judge_payloads.jsonl"
    results_path.write_text(json.dumps(result) + "\n")
    payload_path.write_text(json.dumps({"canonical_pair_id": "pair", "payload": payload}) + "\n")
    (tmp_path / "run_complete.json").write_text(
        json.dumps({"results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest()})
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "judge_payloads_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                "settings": {"prompt_version": "dedup-judge-hs-v0.6.2.10"},
            }
        )
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "canonical_pair_id,human_same_duplicate_group,human_a_can_replace_b,human_b_can_replace_a,human_relation_type,human_material_difference\npair,YES,YES,YES,EXACT,NONE\n"
    )
    replay = replay_run(tmp_path, labels)
    assert replay["eligible_for_promotion"] is False
    assert replay["replay"]["unweighted"]["primary_decision_exact"] == 1
    assert replay["replay_results"][0]["same_duplicate_group"] == "YES"
    critic["record_binding_verdict"]["reasoning"] = "Changed source reasoning."
    raw_path.write_text(
        json.dumps(
            {
                "canonical_pair_id": "pair",
                "qwen_dedup_semantic_judge": main,
                "qwen_dedup_record_binding_critic": critic,
            }
        )
        + "\n"
    )
    with pytest.raises(DedupEvaluationError, match="REPLAY_PROVIDER_CHANGED"):
        replay_run(tmp_path, labels)
