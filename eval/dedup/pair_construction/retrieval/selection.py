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

"""Step 5b pilot tuning, channel union, quotas, and deterministic refill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.dedup.config import EvaluationConfig, ProfileConfig
from eval.dedup.contracts import Track, cp1_pair, stable_record_id
from eval.dedup.handoff.corpus import load_documents_by_ids
from eval.dedup.pair_construction.anchors import GROUP_BUCKETS, group_size_bucket, redistribute_group_quotas
from eval.dedup.pair_construction.retrieval.lexical import (
    char_shingles,
    choose_lsh_configuration,
    lsh_candidates,
    pair_features,
    pair_features_from_shingles,
)
from eval.dedup.pair_construction.retrieval.semantic import exact_cosine_topk
from eval.dedup.validation import require, write_json_atomic


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "numpy and pyarrow are required for cross-group selection"
        raise RuntimeError(msg) from exc
    return np, pa, pq


def _outcome_arrays(outcomes_path: Path, expected_rows: int) -> tuple[Any, Any, Any]:
    np, _, pq = _dependencies()
    table = pq.read_table(outcomes_path, columns=["doc_id", "predicted_group_id", "predicted_group_size"])
    doc_ids = table["doc_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    require(
        len(doc_ids) == expected_rows and np.array_equal(doc_ids, np.arange(expected_rows, dtype=np.int64)),
        "OUTCOME_ID_ORDER_MISMATCH",
        "document outcomes must be in contiguous doc_id order for retrieval",
    )
    return (
        doc_ids,
        table["predicted_group_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
        table["predicted_group_size"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
    )


def _pilot_anchor_ids(
    doc_ids: Any,
    group_ids: Any,
    group_sizes: Any,
    *,
    seed: int,
    target: int,
) -> list[int]:
    np, _, _ = _dependencies()
    rng = np.random.default_rng(seed)
    if target < 10:
        return sorted(rng.choice(doc_ids, size=target, replace=False).astype(int).tolist())
    desired = {"singleton": target // 2, "size_2": target // 8, "size_3_5": target // 8, "size_6_20": target // 8}
    desired["size_21_plus"] = target - sum(desired.values())
    selected: list[int] = []
    singletons = doc_ids[group_ids == -1]
    take_singletons = min(desired["singleton"], len(singletons))
    selected.extend(rng.choice(singletons, size=take_singletons, replace=False).astype(int).tolist())
    grouped_mask = group_ids != -1
    grouped_groups = group_ids[grouped_mask]
    grouped_docs = doc_ids[grouped_mask]
    unique_groups, first = np.unique(grouped_groups, return_index=True)
    unique_sizes = group_sizes[grouped_mask][first]
    pools = {
        bucket: unique_groups[[group_size_bucket(int(size)) == bucket for size in unique_sizes]]
        for bucket in GROUP_BUCKETS
    }
    capacities = {bucket: len(pools[bucket]) for bucket in GROUP_BUCKETS}
    allocation = redistribute_group_quotas({bucket: desired[bucket] for bucket in GROUP_BUCKETS}, capacities)
    for bucket in GROUP_BUCKETS:
        for group_id in rng.choice(pools[bucket], size=allocation[bucket], replace=False).tolist():
            members = grouped_docs[grouped_groups == group_id]
            selected.append(int(rng.choice(members)))
    if len(selected) < target:
        remaining = np.setdiff1d(doc_ids, np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=target - len(selected), replace=False).astype(int).tolist())
    return sorted(selected)


def _filter_cross_group(raw: dict[int, set[int]], group_ids: Any) -> dict[int, set[int]]:
    filtered: dict[int, set[int]] = {}
    for anchor_id, candidates in raw.items():
        anchor_group = group_ids[anchor_id]
        filtered[anchor_id] = {
            candidate
            for candidate in candidates
            if candidate != anchor_id and (anchor_group == -1 or group_ids[candidate] != anchor_group)
        }
    return filtered


def _lexical_records(
    candidate_ids: dict[int, set[int]],
    *,
    corpus_manifest: dict[str, Any],
    feature_width: int,
    top_k: int,
) -> list[dict[str, Any]]:
    all_ids = set(candidate_ids)
    for values in candidate_ids.values():
        all_ids.update(values)
    documents = load_documents_by_ids(corpus_manifest, sorted(all_ids), columns=("text",))
    output: list[dict[str, Any]] = []
    for anchor_id, candidates in candidate_ids.items():
        anchor_text = documents[anchor_id]["text"]
        anchor_shingles = char_shingles(anchor_text, feature_width)
        rows = []
        for candidate_id in candidates:
            candidate_text = documents[candidate_id]["text"]
            features = pair_features_from_shingles(
                anchor_shingles,
                char_shingles(candidate_text, feature_width),
                left_text_length=len(anchor_text),
                right_text_length=len(candidate_text),
            )
            rows.append({"anchor_id": anchor_id, "candidate_id": candidate_id, **features})
        rows.sort(key=lambda row: (-row["jaccard"], -row["containment"], row["candidate_id"]))
        for rank, row in enumerate(rows[:top_k], start=1):
            output.append({**row, "lexical_rank": rank})
    return output


def _semantic_records(
    neighbors: dict[int, list[tuple[int, float, int]]],
    *,
    group_ids: Any,
    corpus_manifest: dict[str, Any],
    feature_width: int,
) -> list[dict[str, Any]]:
    filtered: list[tuple[int, int, float, int]] = []
    for anchor_id, rows in neighbors.items():
        anchor_group = group_ids[anchor_id]
        for candidate_id, score, rank in rows:
            if candidate_id == anchor_id or (anchor_group != -1 and group_ids[candidate_id] == anchor_group):
                continue
            filtered.append((anchor_id, candidate_id, score, rank))
    all_ids = {item for row in filtered for item in row[:2]}
    documents = load_documents_by_ids(corpus_manifest, sorted(all_ids), columns=("text",))
    output = []
    for anchor_id, candidate_id, score, rank in filtered:
        features = pair_features(
            documents[anchor_id]["text"],
            documents[candidate_id]["text"],
            ngram_width=feature_width,
        )
        output.append(
            {
                "anchor_id": anchor_id,
                "candidate_id": candidate_id,
                "cosine": score,
                "semantic_rank": rank,
                **features,
            }
        )
    return output


def _merge_channels(
    lexical: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    *,
    cosine_cutoff: float,
    jaccard_cutoff: float,
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for row in lexical:
        merged[(row["anchor_id"], row["candidate_id"])] = {
            **row,
            "cosine": None,
            "semantic_rank": None,
        }
    for row in semantic:
        key = (row["anchor_id"], row["candidate_id"])
        if key in merged:
            lexical_row = merged[key]
            lexical_row["cosine"] = row["cosine"]
            lexical_row["semantic_rank"] = row["semantic_rank"]
        else:
            merged[key] = {**row, "lexical_rank": None}
    output = []
    for row in merged.values():
        has_lexical = row["lexical_rank"] is not None
        has_semantic = row["semantic_rank"] is not None
        source = "both" if has_lexical and has_semantic else "lexical_only" if has_lexical else "semantic_only"
        normalized = min(
            row["lexical_rank"] / top_k if has_lexical else float("inf"),
            row["semantic_rank"] / top_k if has_semantic else float("inf"),
        )
        output.append(
            {
                **row,
                "retriever_bitmask": source,
                "semantic_high_lexical_low": bool(
                    has_semantic and row["cosine"] >= cosine_cutoff and row["jaccard"] < jaccard_cutoff
                ),
                "best_normalized_rank": normalized,
                "canonical_pair_id": cp1_pair(row["anchor_id"], row["candidate_id"]).canonical_pair_id,
            }
        )
    return output


def _per_anchor_selection(
    records: list[dict[str, Any]], *, top_k: int
) -> tuple[set[str], dict[tuple[int, int], tuple[str, int]]]:
    by_anchor: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        by_anchor.setdefault(row["anchor_id"], []).append(row)
    chosen_events: list[tuple[int, int, dict[str, Any], str]] = []
    event_rule: dict[tuple[int, int], tuple[str, int]] = {}
    for anchor_id, rows in by_anchor.items():
        lexical = sorted(
            (row for row in rows if row["retriever_bitmask"] == "lexical_only"),
            key=lambda row: (-row["jaccard"], row["canonical_pair_id"]),
        )
        semantic = sorted(
            (row for row in rows if row["retriever_bitmask"] == "semantic_only"),
            key=lambda row: (-int(row["semantic_high_lexical_low"]), -row["cosine"], row["canonical_pair_id"]),
        )
        both = sorted(
            (row for row in rows if row["retriever_bitmask"] == "both"),
            key=lambda row: (
                row["lexical_rank"] / top_k + row["semantic_rank"] / top_k,
                row["canonical_pair_id"],
            ),
        )
        selected: list[tuple[dict[str, Any], str]] = []
        for slot in range(4):
            if slot < len(lexical):
                selected.append((lexical[slot], "lexical_quota"))
            if slot < len(semantic):
                selected.append((semantic[slot], "semantic_quota"))
            if slot < 2 and slot < len(both):
                selected.append((both[slot], "both_quota"))
        selected_keys = {(row["anchor_id"], row["candidate_id"]) for row, _ in selected}
        refill = sorted(
            (row for row in rows if (row["anchor_id"], row["candidate_id"]) not in selected_keys),
            key=lambda row: (row["best_normalized_rank"], row["canonical_pair_id"]),
        )
        for item in refill[: 10 - len(selected)]:
            selected.append((item, "per_anchor_refill"))
        for priority, (row, rule) in enumerate(selected):
            chosen_events.append((priority, anchor_id, row, rule))
            event_rule[(anchor_id, row["candidate_id"])] = (rule, priority)
    chosen_events.sort(key=lambda item: (item[0], item[1], item[2]["canonical_pair_id"]))
    return {item[2]["canonical_pair_id"] for item in chosen_events}, event_rule


def retrieve_and_select_cross_group_pairs(
    config: EvaluationConfig,
    *,
    profile: ProfileConfig,
    corpus_manifest: dict[str, Any],
    outcomes_path: Path,
    anchors_path: Path,
    signature_path: Path,
    signature_manifest: dict[str, Any],
    destination: Path,
    retrieval_config_destination: Path,
) -> dict[str, int]:
    """Execute both retrieval channels and write selected provenance memberships."""

    np, pa, pq = _dependencies()
    doc_ids, group_ids, group_sizes = _outcome_arrays(outcomes_path, config.dataset.expected_rows)
    pilot_count = min(100, config.dataset.expected_rows)
    pilot_ids = _pilot_anchor_ids(doc_ids, group_ids, group_sizes, seed=config.seeds["pilot_seed"], target=pilot_count)
    (bands, rows_per_band), trials = choose_lsh_configuration(
        signature_path,
        config=config,
        pilot_anchor_ids=pilot_ids,
        predicted_group_ids=group_ids,
    )
    pilot_semantic = exact_cosine_topk(
        Path(corpus_manifest["embedding"]["path"]),
        rows=config.dataset.embedding_rows,
        dimensions=config.dataset.embedding_dimensions,
        anchor_ids=pilot_ids,
        predicted_group_ids=group_ids,
        top_k=config.retrieval.top_k,
        chunk_rows=config.retrieval.semantic_chunk_rows,
    )
    pilot_semantic_records = _semantic_records(
        pilot_semantic,
        group_ids=group_ids,
        corpus_manifest=corpus_manifest,
        feature_width=config.retrieval.feature_ngram_width,
    )
    require(pilot_semantic_records, "SEMANTIC_PILOT_EMPTY", "semantic pilot produced no cross-group candidates")
    cosine_cutoff = float(np.quantile([row["cosine"] for row in pilot_semantic_records], 0.9))
    jaccard_cutoff = float(np.median([row["jaccard"] for row in pilot_semantic_records]))

    anchor_table = pq.read_table(anchors_path, columns=["anchor_id", "evaluation_run_id", "sut_run_id"])
    anchor_ids = [int(value) for value in anchor_table["anchor_id"].to_pylist()]
    raw_lexical = lsh_candidates(
        signature_path,
        row_count=config.dataset.expected_rows,
        num_hashes=config.retrieval.num_hashes,
        anchor_ids=anchor_ids,
        bands=bands,
        rows_per_band=rows_per_band,
        chunk_rows=config.retrieval.signature_chunk_rows,
        max_candidates_per_anchor=config.retrieval.max_candidates_per_anchor,
    )
    filtered_lexical = _filter_cross_group(raw_lexical, group_ids)
    lexical_records = _lexical_records(
        filtered_lexical,
        corpus_manifest=corpus_manifest,
        feature_width=config.retrieval.feature_ngram_width,
        top_k=config.retrieval.top_k,
    )
    semantic_neighbors = exact_cosine_topk(
        Path(corpus_manifest["embedding"]["path"]),
        rows=config.dataset.embedding_rows,
        dimensions=config.dataset.embedding_dimensions,
        anchor_ids=anchor_ids,
        predicted_group_ids=group_ids,
        top_k=config.retrieval.top_k,
        chunk_rows=config.retrieval.semantic_chunk_rows,
    )
    semantic_records = _semantic_records(
        semantic_neighbors,
        group_ids=group_ids,
        corpus_manifest=corpus_manifest,
        feature_width=config.retrieval.feature_ngram_width,
    )
    records = _merge_channels(
        lexical_records,
        semantic_records,
        cosine_cutoff=cosine_cutoff,
        jaccard_cutoff=jaccard_cutoff,
        top_k=config.retrieval.top_k,
    )
    initially_selected, event_rules = _per_anchor_selection(records, top_k=config.retrieval.top_k)
    ordered_initial = sorted(
        initially_selected,
        key=lambda pair_id: min(
            (
                priority,
                row["anchor_id"],
                pair_id,
            )
            for row in records
            if row["canonical_pair_id"] == pair_id
            for _, priority in [event_rules.get((row["anchor_id"], row["candidate_id"]), ("", 99))]
        ),
    )
    selected_pair_ids = set(ordered_initial[: profile.cross_group_pair_budget])
    if len(selected_pair_ids) < profile.cross_group_pair_budget:
        refill = sorted(
            (row for row in records if row["canonical_pair_id"] not in selected_pair_ids),
            key=lambda row: (row["best_normalized_rank"], row["canonical_pair_id"], row["anchor_id"]),
        )
        for row in refill:
            selected_pair_ids.add(row["canonical_pair_id"])
            if len(selected_pair_ids) == profile.cross_group_pair_budget:
                break
    run_id = anchor_table["evaluation_run_id"][0].as_py()
    sut_run_id = anchor_table["sut_run_id"][0].as_py()
    output_rows = []
    for row in records:
        if row["canonical_pair_id"] not in selected_pair_ids:
            continue
        rule, priority = event_rules.get((row["anchor_id"], row["candidate_id"]), ("membership_preserved", None))
        output_rows.append(
            {
                "evaluation_run_id": run_id,
                "sut_run_id": sut_run_id,
                "track": Track.CROSS_GROUP,
                "stream": "cross_group_retrieval",
                "doc_i": row["anchor_id"],
                "doc_j": row["candidate_id"],
                "anchor_id": row["anchor_id"],
                "canonical_pair_id": row["canonical_pair_id"],
                "retriever_bitmask": row["retriever_bitmask"],
                "lexical_rank": row["lexical_rank"],
                "semantic_rank": row["semantic_rank"],
                "jaccard": row["jaccard"],
                "containment": row["containment"],
                "length_ratio": row["length_ratio"],
                "cosine": row["cosine"],
                "semantic_high_lexical_low": row["semantic_high_lexical_low"],
                "selection_stratum": row["retriever_bitmask"],
                "selection_rule": rule,
                "within_anchor_priority": priority,
                "frame_size": len(records),
                "selection_probability": None,
                "provenance_record_id": stable_record_id(
                    "cross-group-v1", run_id, row["anchor_id"], row["candidate_id"], row["retriever_bitmask"]
                ),
            }
        )
    require(selected_pair_ids, "EMPTY_CROSS_GROUP_SELECTION", "Step 5b selected no unique pairs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), destination, compression="zstd")
    write_json_atomic(
        retrieval_config_destination,
        {
            "schema_version": "retrieval-config-v1",
            "pilot_seed": config.seeds["pilot_seed"],
            "pilot_anchor_ids": pilot_ids,
            "lexical_trials": trials,
            "pilot_candidate_count_target": {
                "minimum": config.retrieval.pilot_target_min,
                "maximum": config.retrieval.pilot_target_max,
                "center": config.retrieval.pilot_target_center,
            },
            "candidate_safety_limit_per_anchor": config.retrieval.max_candidates_per_anchor,
            "selected_lsh": {"bands": bands, "rows_per_band": rows_per_band},
            "semantic_cosine_p90": cosine_cutoff,
            "semantic_jaccard_median": jaccard_cutoff,
            "minhash_contract_digest": signature_manifest["contract_digest"],
            "minhash_matrix_sha256": signature_manifest["matrix_sha256"],
            "embedding_sha256": config.dataset.embedding_sha256,
            "top_k": config.retrieval.top_k,
        },
    )
    return {
        "lexical_candidates": len(lexical_records),
        "semantic_candidates": len(semantic_records),
        "union_candidates": len(records),
        "unique_selected_pairs": len(selected_pair_ids),
        "provenance_rows": len(output_rows),
    }
