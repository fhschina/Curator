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

"""Step 5c: canonicalize the combined queue without losing provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.dedup.config import EvaluationConfig
from eval.dedup.contracts import CANONICAL_PAIR_ID_VERSION, Track, cp1_pair, stable_record_id
from eval.dedup.handoff.corpus import TokenCounter, load_documents_by_ids
from eval.dedup.judging.payload import assert_blind_payload, build_visible_payload
from eval.dedup.validation import require


def _dependencies() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for pair canonicalization"
        raise RuntimeError(msg) from exc
    return pa, pq


def _load_outcome_details(path: Path, doc_ids: list[int]) -> dict[int, dict[str, Any]]:
    _, pq = _dependencies()
    columns = [
        "doc_id",
        "predicted_cluster_key",
        "predicted_group_id",
        "action",
        "final_keeper_id",
        "language",
        "length_bucket",
        "token_count",
        "hostname",
        "canonical_url_v0",
        "url",
        "crawl_timestamp",
        "shard_index",
        "physical_row_index",
    ]
    table = pq.read_table(path, columns=columns, filters=[("doc_id", "in", doc_ids)])
    result = {int(row["doc_id"]): row for row in table.to_pylist()}
    require(len(result) == len(doc_ids), "PAIR_ENDPOINT_JOIN_INCOMPLETE", "pair endpoint is missing from outcomes")
    return result


def canonicalize_selected_pairs(
    config: EvaluationConfig,
    *,
    corpus_manifest: dict[str, Any],
    outcomes_path: Path,
    removal_pairs_path: Path,
    cross_group_pairs_path: Path,
    tokenizer: TokenCounter,
    candidate_destination: Path,
    provenance_destination: Path,
) -> dict[str, int]:
    """Write authoritative one-row-per-pair and one-row-per-event contracts."""

    pa, pq = _dependencies()
    removal_rows = pq.read_table(removal_pairs_path).to_pylist()
    cross_rows = pq.read_table(cross_group_pairs_path).to_pylist()
    events = [*removal_rows, *cross_rows]
    require(events, "EMPTY_JUDGE_QUEUE", "no Step 5 pair events were produced")
    endpoints = sorted({int(row["doc_i"]) for row in events} | {int(row["doc_j"]) for row in events})
    outcome_details = _load_outcome_details(outcomes_path, endpoints)
    documents = load_documents_by_ids(
        corpus_manifest,
        endpoints,
        columns=("text", "url", "timestamp", "language", "source_id", "warc_path", "warc_record_id"),
    )
    pair_events: dict[str, list[dict[str, Any]]] = {}
    pair_endpoints: dict[str, tuple[int, int]] = {}
    provenance_rows = []
    for event in events:
        doc_i = int(event["doc_i"])
        doc_j = int(event["doc_j"])
        pair = cp1_pair(doc_i, doc_j)
        low = int(pair.doc_id_low)
        high = int(pair.doc_id_high)
        require(
            pair.canonical_pair_id_version == config.canonical_pair_id_version,
            "PAIR_ID_VERSION_MISMATCH",
            "canonical pair version differs from frozen manifest",
        )
        existing = pair_endpoints.setdefault(pair.canonical_pair_id, (low, high))
        require(existing == (low, high), "PAIR_ID_COLLISION", "canonical pair ID maps to more than one endpoint tuple")
        pair_events.setdefault(pair.canonical_pair_id, []).append(event)
        row = {
            **event,
            "canonical_pair_id_version": CANONICAL_PAIR_ID_VERSION,
            "canonical_pair_id": pair.canonical_pair_id,
            "doc_id_low": low,
            "doc_id_high": high,
        }
        if event["track"] == Track.REMOVAL:
            row["ordered_keeper_id"] = int(event["keeper_doc_id"])
            row["ordered_removed_id"] = int(event["removed_doc_id"])
        else:
            row["ordered_keeper_id"] = None
            row["ordered_removed_id"] = None
        if not row.get("provenance_record_id"):
            row["provenance_record_id"] = stable_record_id(
                "pair-provenance-v1", event["evaluation_run_id"], pair.canonical_pair_id, len(provenance_rows)
            )
        provenance_rows.append(row)
    require(
        len({row["provenance_record_id"] for row in provenance_rows}) == len(provenance_rows),
        "DUPLICATE_PROVENANCE_ID",
        "provenance record IDs are not unique",
    )

    candidate_rows = []
    for pair_id in sorted(pair_endpoints):
        low, high = pair_endpoints[pair_id]
        order_value = int(stable_record_id("judge-order-v1", config.seeds["judge_order_seed"], pair_id), 16) & 1
        doc_a, doc_b = (low, high) if order_value == 0 else (high, low)
        payload, payload_hash = build_visible_payload(
            documents[doc_a], documents[doc_b], counter=tokenizer, config=config.judge
        )
        assert_blind_payload(payload)
        first_event = pair_events[pair_id][0]
        candidate_rows.append(
            {
                "evaluation_run_id": first_event["evaluation_run_id"],
                "sut_run_id": first_event["sut_run_id"],
                "canonical_pair_id_version": CANONICAL_PAIR_ID_VERSION,
                "canonical_pair_id": pair_id,
                "doc_id_low": low,
                "doc_id_high": high,
                "presented_doc_a": doc_a,
                "presented_doc_b": doc_b,
                "judge_order_seed": config.seeds["judge_order_seed"],
                "judge_payload_hash": payload_hash,
                "doc_low_shard_index": outcome_details[low]["shard_index"],
                "doc_low_physical_row_index": outcome_details[low]["physical_row_index"],
                "doc_high_shard_index": outcome_details[high]["shard_index"],
                "doc_high_physical_row_index": outcome_details[high]["physical_row_index"],
                "language_low": outcome_details[low]["language"],
                "language_high": outcome_details[high]["language"],
                "length_bucket_low": outcome_details[low]["length_bucket"],
                "length_bucket_high": outcome_details[high]["length_bucket"],
                "token_count_low": outcome_details[low]["token_count"],
                "token_count_high": outcome_details[high]["token_count"],
                "url_low": outcome_details[low]["url"],
                "url_high": outcome_details[high]["url"],
                "hostname_low": outcome_details[low]["hostname"],
                "hostname_high": outcome_details[high]["hostname"],
                "canonical_url_low": outcome_details[low]["canonical_url_v0"],
                "canonical_url_high": outcome_details[high]["canonical_url_v0"],
                "crawl_timestamp_low": outcome_details[low]["crawl_timestamp"],
                "crawl_timestamp_high": outcome_details[high]["crawl_timestamp"],
                "judge_status": "pending",
            }
        )
    require(
        all(pair_id in pair_events for pair_id in pair_endpoints),
        "PAIR_PROVENANCE_FK_MISMATCH",
        "candidate pair has no provenance event",
    )
    candidate_destination.parent.mkdir(parents=True, exist_ok=True)
    provenance_destination.parent.mkdir(parents=True, exist_ok=True)
    provenance_fields = list(dict.fromkeys(key for row in provenance_rows for key in row))
    normalized_provenance_rows = [{field: row.get(field) for field in provenance_fields} for row in provenance_rows]
    pq.write_table(pa.Table.from_pylist(candidate_rows), candidate_destination, compression="zstd")
    pq.write_table(pa.Table.from_pylist(normalized_provenance_rows), provenance_destination, compression="zstd")
    return {
        "unique_pairs": len(candidate_rows),
        "provenance_rows": len(provenance_rows),
        "multi_event_pairs": sum(len(rows) > 1 for rows in pair_events.values()),
    }
