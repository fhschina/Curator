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

"""Step 3: recover one authoritative predicted outcome per corpus document."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from eval.dedup.core.config import EvaluationConfig
from eval.dedup.core.contracts import DOCUMENT_OUTCOME_COLUMNS, Action
from eval.dedup.core.validation import require
from eval.dedup.handoff.corpus import TokenCounter, load_documents_by_ids
from eval.dedup.handoff.sut import load_sut_arrays


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "numpy and pyarrow are required to build document outcomes"
        raise RuntimeError(msg) from exc
    return np, (pa, pq)


def canonicalize_url_v0(url: str | None) -> tuple[str | None, str | None, bool]:
    """Return hostname, conservative canonical URL, and parse success."""

    if not url:
        return None, None, True
    try:
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.hostname:
            return None, None, False
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname.lower()
        port = parsed.port
        if port is None or (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            netloc = hostname
        else:
            netloc = f"{hostname}:{port}"
        if parsed.username is not None:
            userinfo = parsed.username
            if parsed.password is not None:
                userinfo += f":{parsed.password}"
            netloc = f"{userinfo}@{netloc}"
        return hostname, urlunsplit((scheme, netloc, parsed.path, parsed.query, "")), True
    except (TypeError, ValueError, UnicodeError):
        return None, None, False


def _length_bucket(count: int) -> str:
    if count <= 500:
        return "short"
    if count <= 2_000:
        return "medium"
    return "long"


def _relations(doc_ids: Any, sut: Any, evaluation_run_id: str, sut_run_id: str) -> dict[str, Any]:
    groups, sizes, keepers, removed = sut.lookup(doc_ids)
    return {
        "evaluation_run_id": [evaluation_run_id] * len(doc_ids),
        "sut_run_id": [sut_run_id] * len(doc_ids),
        "doc_id": doc_ids,
        "predicted_group_id": groups,
        "predicted_cluster_key": [
            f"singleton:{int(doc_id)}" if group == -1 else f"group:{sut_run_id}:{int(group)}"
            for doc_id, group in zip(doc_ids, groups, strict=True)
        ],
        "predicted_group_size": sizes,
        "action": [Action.REMOVE if flag else Action.KEEP for flag in removed],
        "final_keeper_id": keepers,
    }


def build_document_relations(
    config: EvaluationConfig,
    *,
    evaluation_manifest: dict[str, Any],
    corpus_manifest: dict[str, Any],
    sut_manifest: dict[str, Any],
    destination: Path,
) -> dict[str, int]:
    """Build the sampling frame from validated IDs and SUT relations, without text I/O."""
    np, (pa, pq) = _dependencies()
    sut = load_sut_arrays(
        config,
        groups_path=Path(sut_manifest["duplicate_groups"]["path"]),
        removals_path=Path(sut_manifest["removal_ids"]["path"]),
    )
    rows = config.dataset.expected_rows
    locations = (
        np.memmap(corpus_manifest["document_locations"], dtype=np.int64, mode="r", shape=(rows, 2))
        if corpus_manifest.get("explicit_id_column")
        else None
    )
    ends = np.asarray([shard.get("end_id", -1) for shard in corpus_manifest["shards"]])
    starts = np.asarray([shard.get("start_id", 0) for shard in corpus_manifest["shards"]])
    destination.parent.mkdir(parents=True, exist_ok=True)
    removed = singletons = 0
    writer = None
    try:
        for start in range(0, rows, 16384):
            ids = np.arange(start, min(start + 16384, rows), dtype=np.int64)
            values = _relations(ids, sut, evaluation_manifest["evaluation_run_id"], sut_manifest["sut_run_id"])
            if locations is not None:
                values["shard_index"] = pa.array(locations[ids, 0], type=pa.int32())
                values["physical_row_index"] = locations[ids, 1]
            else:
                indices = np.searchsorted(ends, ids)
                values["shard_index"] = pa.array(indices, type=pa.int32())
                values["physical_row_index"] = ids - starts[indices]
            removed += sum(action == Action.REMOVE for action in values["action"])
            singletons += int((values["predicted_group_id"] == -1).sum())
            table = pa.table(values)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    require(removed == config.dataset.expected_removals, "OUTCOME_REMOVAL_COUNT_MISMATCH", "removal count differs")
    require(
        singletons == config.dataset.expected_singletons, "OUTCOME_SINGLETON_COUNT_MISMATCH", "singleton count differs"
    )
    require(
        rows - removed == config.dataset.expected_retained, "OUTCOME_RETAINED_COUNT_MISMATCH", "retained count differs"
    )
    return {"rows": rows, "removals": removed, "singletons": singletons, "logical_retained": rows - removed}


def build_document_outcomes(
    config: EvaluationConfig,
    *,
    evaluation_manifest: dict[str, Any],
    corpus_manifest: dict[str, Any],
    sut_manifest: dict[str, Any],
    destination: Path,
    tokenizer: TokenCounter,
    doc_ids: Sequence[int],
) -> dict[str, Any]:
    """Enrich only selected pair endpoints with exact lengths and optional metadata."""
    np, (pa, pq) = _dependencies()
    sut = load_sut_arrays(
        config,
        groups_path=Path(sut_manifest["duplicate_groups"]["path"]),
        removals_path=Path(sut_manifest["removal_ids"]["path"]),
    )
    ids = sorted({int(doc_id) for doc_id in doc_ids})
    require(bool(ids), "EMPTY_ENDPOINTS", "selected pair endpoints must not be empty")
    input_columns = ("source_id", "warc_path", "warc_record_id", "url", "timestamp", "language", "text")
    strings = {
        "source_id",
        "warc_path",
        "warc_id",
        "url",
        "crawl_timestamp",
        "language",
        "hostname",
        "canonical_url_v0",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    documents = load_documents_by_ids(corpus_manifest, ids, columns=input_columns)
    writer = None
    url_nulls = url_failures = empty_texts = 0
    try:
        for start in range(0, len(ids), 256):
            batch_ids = ids[start : start + 256]
            rows = [documents[doc_id] for doc_id in batch_ids]
            texts = [row["text"] for row in rows]
            require(all(text is not None for text in texts), "NULL_TEXT", "selected text must not be null")
            counts = tokenizer.count_many(texts)
            values = _relations(
                np.asarray(batch_ids, dtype=np.int64),
                sut,
                evaluation_manifest["evaluation_run_id"],
                sut_manifest["sut_run_id"],
            )
            urls = [canonicalize_url_v0(row["url"]) for row in rows]
            empty_texts += sum(not text for text in texts)
            url_nulls += sum(not row["url"] for row in rows)
            url_failures += sum(bool(row["url"]) and not url[2] for row, url in zip(rows, urls, strict=True))
            values.update(
                {
                    "char_count": [len(text) for text in texts],
                    "token_count": counts,
                    "length_bucket": [_length_bucket(count) for count in counts],
                    **{name: [row[name] for row in rows] for name in ("source_id", "warc_path", "url", "language")},
                    "warc_id": [row["warc_record_id"] for row in rows],
                    "crawl_timestamp": [row["timestamp"] for row in rows],
                    "hostname": [url[0] for url in urls],
                    "canonical_url_v0": [url[1] for url in urls],
                    "shard_index": pa.array([row["shard_index"] for row in rows], type=pa.int32()),
                    "physical_row_index": [row["physical_row_index"] for row in rows],
                }
            )
            table = pa.table(
                {
                    name: pa.array(values[name], type=pa.string()) if name in strings else values[name]
                    for name in DOCUMENT_OUTCOME_COLUMNS
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return {
        "rows": len(ids),
        "scope": "selected_pair_endpoints",
        "url_nulls": url_nulls,
        "url_parse_failures": url_failures,
        "empty_texts": empty_texts,
    }
