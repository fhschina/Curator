# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step 8: restore hidden SUT context and classify track-specific outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.dedup.contracts import DuplicateAnswer, Track
from eval.dedup.validation import require


def _jsonl_by_pair(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            result[row["canonical_pair_id"]] = row
    return result


def build_pair_comparisons(
    *,
    candidate_pairs_path: Path,
    pair_provenance_path: Path,
    outcomes_path: Path,
    judge_results_path: Path,
    judge_errors_path: Path,
    destination: Path,
) -> dict[str, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for pair comparison"
        raise RuntimeError(msg) from exc
    candidates = pq.read_table(candidate_pairs_path).to_pylist()
    provenance = pq.read_table(pair_provenance_path).to_pylist()
    provenance_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in provenance:
        provenance_by_pair.setdefault(row["canonical_pair_id"], []).append(row)
    endpoints = sorted(
        {int(row["doc_id_low"]) for row in candidates} | {int(row["doc_id_high"]) for row in candidates}
    )
    outcome_table = pq.read_table(
        outcomes_path,
        columns=["doc_id", "predicted_cluster_key", "predicted_group_size", "action", "final_keeper_id"],
        filters=[("doc_id", "in", endpoints)],
    )
    outcomes = {int(row["doc_id"]): row for row in outcome_table.to_pylist()}
    require(len(outcomes) == len(endpoints), "COMPARISON_ENDPOINT_JOIN_INCOMPLETE", "comparison endpoint missing")
    results = _jsonl_by_pair(judge_results_path)
    errors = _jsonl_by_pair(judge_errors_path)
    output = []
    for pair in candidates:
        pair_id = pair["canonical_pair_id"]
        events = provenance_by_pair.get(pair_id, [])
        require(events, "PAIR_PROVENANCE_FK_MISMATCH", "candidate has no provenance", canonical_pair_id=pair_id)
        result = results.get(pair_id)
        error = errors.get(pair_id)
        require(
            (result is None) != (error is None),
            "JUDGE_JOIN_INCOMPLETE",
            "pair must have exactly one result or error",
            canonical_pair_id=pair_id,
        )
        if result is not None:
            require(
                result["judge_payload_hash"] == pair["judge_payload_hash"],
                "JUDGE_PAYLOAD_HASH_MISMATCH",
                "judge join hash differs",
            )
            require(
                result["canonical_pair_id_version"] == pair["canonical_pair_id_version"],
                "PAIR_ID_VERSION_MISMATCH",
                "judge join version differs",
            )
        low = int(pair["doc_id_low"])
        high = int(pair["doc_id_high"])
        predicted_same = outcomes[low]["predicted_cluster_key"] == outcomes[high]["predicted_cluster_key"]
        removal_events = [event for event in events if event["track"] == Track.REMOVAL]
        cross_events = [event for event in events if event["track"] == Track.CROSS_GROUP]
        keeper_can_replace_removed = None
        removal_outcome = None
        keeper_id = None
        removed_id = None
        if removal_events:
            directions = {
                (int(event["ordered_keeper_id"]), int(event["ordered_removed_id"])) for event in removal_events
            }
            require(len(directions) == 1, "AMBIGUOUS_REMOVAL_DIRECTION", "pair has conflicting keeper directions")
            keeper_id, removed_id = next(iter(directions))
            if result is not None:
                keeper_can_replace_removed = (
                    result["a_can_replace_b"]
                    if int(pair["presented_doc_a"]) == keeper_id
                    else result["b_can_replace_a"]
                )
                removal_outcome = {
                    DuplicateAnswer.YES: "safe_removal",
                    DuplicateAnswer.NO: "wrong_removal",
                    DuplicateAnswer.UNRESOLVED: "unresolved",
                }[keeper_can_replace_removed]
            else:
                removal_outcome = "judge_error"
        cross_outcome = None
        if cross_events:
            if result is None:
                cross_outcome = "judge_error"
            else:
                cross_outcome = {
                    DuplicateAnswer.YES: "discovered_candidate_fn",
                    DuplicateAnswer.NO: "hard_negative",
                    DuplicateAnswer.UNRESOLVED: "unresolved",
                }[result["same_duplicate_group"]]
        sources = sorted({event.get("retriever_bitmask") for event in cross_events if event.get("retriever_bitmask")})
        if "both" in sources or ({"lexical_only", "semantic_only"} <= set(sources)):
            retriever_category = "both_or_overlap"
        elif sources == ["lexical_only"]:
            retriever_category = "lexical_only"
        elif sources == ["semantic_only"]:
            retriever_category = "semantic_only"
        else:
            retriever_category = None
        removal_frame_sizes = {int(event["frame_size"]) for event in removal_events}
        removal_probabilities = {float(event["selection_probability"]) for event in removal_events}
        require(
            len(removal_frame_sizes) <= 1 and len(removal_probabilities) <= 1,
            "REMOVAL_FRAME_CONFLICT",
            "pair has inconsistent removal sampling metadata",
        )
        low_tokens = int(pair["token_count_low"])
        high_tokens = int(pair["token_count_high"])
        max_tokens = max(low_tokens, high_tokens)
        token_length_ratio = min(low_tokens, high_tokens) / max_tokens if max_tokens else 1.0
        output.append(
            {
                **pair,
                "predicted_same_group": predicted_same,
                "predicted_group_size_low": outcomes[low]["predicted_group_size"],
                "predicted_group_size_high": outcomes[high]["predicted_group_size"],
                "same_language": bool(pair["language_low"] and pair["language_low"] == pair["language_high"]),
                "same_hostname": bool(pair["hostname_low"] and pair["hostname_low"] == pair["hostname_high"]),
                "token_length_ratio": token_length_ratio,
                "has_track_5a": bool(removal_events),
                "has_track_5b": bool(cross_events),
                "provenance_count": len(events),
                "keeper_doc_id": keeper_id,
                "removed_doc_id": removed_id,
                "keeper_can_replace_removed": keeper_can_replace_removed,
                "removal_outcome": removal_outcome,
                "cross_group_outcome": cross_outcome,
                "judge_status": "valid" if result is not None else "judge_error",
                "same_duplicate_group": result["same_duplicate_group"] if result else None,
                "relation_type": result["relation_type"] if result else None,
                "material_difference": result["material_difference"] if result else None,
                "fuzzy_scope": result["fuzzy_scope"] if result else None,
                "confidence": result["confidence"] if result else None,
                "judge_attempts": result["attempts"] if result else error["attempts"],
                "retriever_sources": sources,
                "retriever_category": retriever_category,
                "removal_sampling_frame_size": next(iter(removal_frame_sizes), None),
                "removal_selection_probability": next(iter(removal_probabilities), None),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output), destination, compression="zstd")
    return {
        "rows": len(output),
        "track_5a_pairs": sum(row["has_track_5a"] for row in output),
        "track_5b_pairs": sum(row["has_track_5b"] for row in output),
        "judge_errors": sum(row["judge_status"] == "judge_error" for row in output),
    }
