# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eval.dedup.analysis.comparison import build_pair_comparisons


def test_v3_comparison_keeps_confidence_tier_and_absent_legacy_diagnostics(tmp_path: Path) -> None:
    candidate = {
        "canonical_pair_id": "cp1_fixture",
        "canonical_pair_id_version": "cp1",
        "judge_payload_hash": "payload",
        "doc_id_low": 1,
        "doc_id_high": 2,
        "presented_doc_a": 1,
        "presented_doc_b": 2,
        "token_count_low": 10,
        "token_count_high": 20,
        "language_low": "ENGLISH",
        "language_high": "ENGLISH",
        "hostname_low": "a.example",
        "hostname_high": "b.example",
    }
    provenance = {
        "canonical_pair_id": "cp1_fixture",
        "track": "5b",
        "retriever_bitmask": "lexical_only",
    }
    outcomes = [
        {
            "doc_id": doc_id,
            "predicted_cluster_key": f"group-{doc_id}",
            "predicted_group_size": 1,
            "action": "KEEP",
            "final_keeper_id": doc_id,
        }
        for doc_id in (1, 2)
    ]
    result = {
        "canonical_pair_id": "cp1_fixture",
        "canonical_pair_id_version": "cp1",
        "judge_payload_hash": "payload",
        "same_duplicate_group": "NO",
        "a_can_replace_b": "NO",
        "b_can_replace_a": "NO",
        "relation_type": "UNRELATED",
        "material_difference": "MAJOR",
        "primary_material_difference": "DOCUMENT_IDENTITY_CHANGE",
        "dominant_overlap_source": "NONE",
        "primary_risk_factor": "NONE",
        "confidence_tier": "HIGH",
        "attempts": 1,
    }
    candidate_path = tmp_path / "candidate.parquet"
    provenance_path = tmp_path / "provenance.parquet"
    outcomes_path = tmp_path / "outcomes.parquet"
    results_path = tmp_path / "results.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    destination = tmp_path / "comparison.parquet"
    pq.write_table(pa.Table.from_pylist([candidate]), candidate_path)
    pq.write_table(pa.Table.from_pylist([provenance]), provenance_path)
    pq.write_table(pa.Table.from_pylist(outcomes), outcomes_path)
    results_path.write_text(json.dumps(result) + "\n")
    errors_path.write_text("")

    counts = build_pair_comparisons(
        candidate_pairs_path=candidate_path,
        pair_provenance_path=provenance_path,
        outcomes_path=outcomes_path,
        judge_results_path=results_path,
        judge_errors_path=errors_path,
        destination=destination,
    )
    row = pq.read_table(destination).to_pylist()[0]

    assert counts["rows"] == 1
    assert row["confidence_tier"] == "HIGH"
    assert row["confidence"] is None
    assert row["expected_minhash_action"] is None
