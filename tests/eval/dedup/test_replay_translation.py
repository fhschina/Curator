# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.dedup.analysis.replay_translation import replay_run
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json


def _source(root: Path) -> Path:
    data, raw = root / "data", root / "runtime/block/attempt_01/output"
    data.mkdir()
    raw.mkdir(parents=True)
    texts = {"A": "We use necessary cookies.", "B": "Nous utilisons des cookies nécessaires."}
    payload = {
        "document_a": {"text": texts["A"]},
        "document_b": {"text": texts["B"]},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": {
            "status": "COMPLETE",
            "span_counts": {"A_ONLY": 1, "B_ONLY": 1, "SHARED": 0},
            "spans": [
                {
                    "span_id": f"{side}001",
                    "kind": f"{side}_ONLY",
                    "side": side,
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                }
                for side, text in texts.items()
            ],
        },
    }
    values = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "cookie_consent",
        "primary_risk_factor": "translation_equivalence",
        "confidence_tier": "medium",
        "span_content_profile_a": "non_main_only",
        "span_content_profile_b": "non_main_only",
        "span_shared_basis": "verified_equivalent_non_main_message",
        "span_a_delta": "semantically_covered",
        "span_b_delta": "semantically_covered",
        "span_hard_conflict": "none",
        "span_translation_status": "complete_faithful",
    }
    main = {
        field: {"score": score, "reasoning": "A001 B001 preserve the same complete message."}
        for field, score in values.items()
    }
    critic = {"record_binding_verdict": {"score": "benign_non_record_delta", "reasoning": "A001 B001 are equivalent."}}
    result = adapt_ndd_judge_output(
        main, "dedup-judge-output-v3", payload=payload, record_binding_critic=critic, record_binding_policy="v3"
    )
    result.update(
        {
            "canonical_pair_id": "pair",
            "judge_contract_digest": "contract",
            "provider_response_sha256": sha256_json({"semantic_judge": main, "record_binding_critic": critic}),
        }
    )
    (data / "judge_payloads.jsonl").write_text(json.dumps({"canonical_pair_id": "pair", "payload": payload}) + "\n")
    (data / "judge_results.jsonl").write_text(json.dumps(result) + "\n")
    saved = {
        "canonical_pair_id": "pair",
        "qwen_dedup_semantic_judge": main,
        "qwen_dedup_record_binding_critic": critic,
    }
    (raw / "raw.jsonl").write_text(json.dumps(saved) + "\n")
    # A later unmatched attempt must not replace the digest-pinned response.
    saved["qwen_dedup_record_binding_critic"] = {
        "record_binding_verdict": {"score": "unresolved", "reasoning": "no proof"}
    }
    (raw / "z_retry.jsonl").write_text(json.dumps(saved) + "\n")
    labels = root / "labels.csv"
    labels.write_text(
        "review_id,canonical_pair_id,sample_weight,human_same_duplicate_group,human_a_can_replace_b,human_b_can_replace_a,human_relation_type,human_material_difference,human_reason_code\nH1,pair,3,YES,YES,YES,NEAR_SURFACE,NONE,translation\n"
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_pair_count": 1,
                "judge_contract_digest": "contract",
                "judge_payloads_sha256": sha256_file(data / "judge_payloads.jsonl"),
                "settings": {"prompt_version": "dedup-judge-hs-v0.6.2.9"},
                "pair_selection": {"sha256": sha256_file(labels)},
            }
        )
    )
    (root / "run_complete.json").write_text(
        json.dumps(
            {"requested": 1, "valid": 1, "errors": 0, "results_sha256": sha256_file(data / "judge_results.jsonl")}
        )
    )
    return labels


def test_translation_replay_uses_real_pinned_scores_and_changes_only_new_policy(tmp_path: Path) -> None:
    labels = _source(tmp_path)
    result_hash = sha256_file(tmp_path / "data/judge_results.jsonl")
    summary = replay_run(tmp_path, labels)
    assert summary["historical_public_replay_exact"] == 1
    assert summary["original"]["weighted"]["duplicate_recall"] == 0
    assert summary["replay"]["weighted"]["duplicate_recall"] == 1
    assert summary["changed_rows"][0]["replay_error"] == "CORRECT"
    assert summary["eligible_for_promotion"] is False
    assert sha256_file(tmp_path / "data/judge_results.jsonl") == result_hash


@pytest.mark.parametrize("changed", ["labels", "raw", "results", "payload", "public_replay"])
def test_translation_replay_rejects_changed_sources_and_nonreproducible_history(tmp_path: Path, changed: str) -> None:
    labels = _source(tmp_path)
    if changed == "labels":
        labels.write_text(labels.read_text() + "\n")
    elif changed == "raw":
        (tmp_path / "runtime/block/attempt_01/output/raw.jsonl").write_text("{}\n")
    elif changed == "public_replay":
        path = tmp_path / "data/judge_results.jsonl"
        result = json.loads(path.read_text())
        result["reason_codes"].append("ALTERED")
        path.write_text(json.dumps(result) + "\n")
        complete_path = tmp_path / "run_complete.json"
        complete = json.loads(complete_path.read_text())
        complete["results_sha256"] = sha256_file(path)
        complete_path.write_text(json.dumps(complete))
    else:
        path = tmp_path / "data" / ("judge_results.jsonl" if changed == "results" else "judge_payloads.jsonl")
        path.write_text(path.read_text() + "\n")
    with pytest.raises(DedupEvaluationError):
        replay_run(tmp_path, labels)
