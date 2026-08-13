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

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from eval.dedup.config import EvaluationConfig
from eval.dedup.contracts import DOCUMENT_OUTCOME_COLUMNS, Action
from eval.dedup.handoff.corpus import TokenCounter, iter_corpus_batches
from eval.dedup.handoff.sut import load_sut_arrays
from eval.dedup.validation import require


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


def build_document_outcomes(
    config: EvaluationConfig,
    *,
    evaluation_manifest: dict[str, Any],
    corpus_manifest: dict[str, Any],
    sut_manifest: dict[str, Any],
    destination: Path,
    tokenizer: TokenCounter,
) -> dict[str, int]:
    """Stream the entire corpus and materialize ``document_outcomes.parquet``."""

    np, (pa, pq) = _dependencies()
    sut = load_sut_arrays(
        config,
        groups_path=Path(sut_manifest["duplicate_groups"]["path"]),
        removals_path=Path(sut_manifest["removal_ids"]["path"]),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: Any | None = None
    total_rows = 0
    removal_count = 0
    singleton_count = 0
    url_null_count = 0
    url_failure_count = 0
    empty_text_count = 0
    input_columns = ("source_id", "warc_path", "warc_record_id", "url", "timestamp", "language", "text")
    for batch in iter_corpus_batches(corpus_manifest, columns=input_columns):
        values = batch.to_pydict()
        doc_ids = np.asarray(values["doc_id"], dtype=np.int64)
        raw_group, group_sizes, keepers, is_removal = sut.lookup(doc_ids)
        texts = values["text"]
        null_indexes = [index for index, text in enumerate(texts) if text is None]
        require(
            not null_indexes,
            "NULL_TEXT",
            "frozen corpus contains null text",
            first_doc_id=int(doc_ids[null_indexes[0]]) if null_indexes else None,
        )
        empty_text_count += sum(not text for text in texts)
        token_counts = tokenizer.count_many(texts)
        hostnames: list[str | None] = []
        canonical_urls: list[str | None] = []
        for url in values["url"]:
            hostname, canonical, ok = canonicalize_url_v0(url)
            hostnames.append(hostname)
            canonical_urls.append(canonical)
            url_null_count += int(not url)
            url_failure_count += int(bool(url) and not ok)
        cluster_keys = [
            f"singleton:{int(doc_id)}" if group_id == -1 else f"group:{sut_manifest['sut_run_id']}:{int(group_id)}"
            for doc_id, group_id in zip(doc_ids, raw_group, strict=True)
        ]
        output = {
            "evaluation_run_id": [evaluation_manifest["evaluation_run_id"]] * len(doc_ids),
            "sut_run_id": [sut_manifest["sut_run_id"]] * len(doc_ids),
            "doc_id": doc_ids,
            "predicted_group_id": raw_group,
            "predicted_cluster_key": cluster_keys,
            "predicted_group_size": group_sizes,
            "action": [Action.REMOVE if flag else Action.KEEP for flag in is_removal],
            "final_keeper_id": keepers,
            "char_count": [len(text) for text in texts],
            "token_count": token_counts,
            "length_bucket": [_length_bucket(count) for count in token_counts],
            "source_id": values["source_id"],
            "warc_path": values["warc_path"],
            "warc_id": values["warc_record_id"],
            "url": values["url"],
            "crawl_timestamp": values["timestamp"],
            "language": values["language"],
            "hostname": hostnames,
            "canonical_url_v0": canonical_urls,
            "shard_index": values["shard_index"],
            "physical_row_index": values["physical_row_index"],
        }
        table = pa.table(output)
        require(
            tuple(table.column_names) == DOCUMENT_OUTCOME_COLUMNS, "INTERNAL_SCHEMA_ERROR", "outcome columns changed"
        )
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
        writer.write_table(table)
        total_rows += len(doc_ids)
        removal_count += int(is_removal.sum())
        singleton_count += int((raw_group == -1).sum())
    if writer is not None:
        writer.close()
    retained_count = total_rows - removal_count
    require(
        total_rows == config.dataset.expected_rows,
        "OUTCOME_ROW_COUNT_MISMATCH",
        "Step 3 did not emit one row per document",
    )
    require(
        removal_count == config.dataset.expected_removals,
        "OUTCOME_REMOVAL_COUNT_MISMATCH",
        "Step 3 removal count differs",
    )
    require(
        singleton_count == config.dataset.expected_singletons,
        "OUTCOME_SINGLETON_COUNT_MISMATCH",
        "Step 3 singleton count differs",
    )
    require(
        retained_count == config.dataset.expected_retained,
        "OUTCOME_RETAINED_COUNT_MISMATCH",
        "Step 3 logical retained count differs",
    )
    return {
        "rows": total_rows,
        "removals": removal_count,
        "singletons": singleton_count,
        "logical_retained": retained_count,
        "url_nulls": url_null_count,
        "url_parse_failures": url_failure_count,
        "empty_texts": empty_text_count,
    }
