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

"""Create evaluation-owned wrappers around the read-only 10M handoff."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dedup.config import EvaluationConfig
from eval.dedup.handoff.sut import load_sut_arrays
from eval.dedup.validation import read_json, require, sha256_file, write_json_atomic


def _parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for dedup evaluation handoff validation"
        raise RuntimeError(msg) from exc
    return pq


def _resolved_shard_path(handoff_root: Path, source_key: str) -> Path:
    return handoff_root / "extracted_text" / "objects" / source_key / "text.parquet"


def register_corpus_handoff(config: EvaluationConfig, destination: Path) -> dict[str, Any]:
    """Validate the frozen shard manifest and write ``corpus_manifest.json``."""

    dense_path = config.handoff_root / "manifest" / "dense_10m_explicit.json"
    singleton_manifest_path = config.handoff_root / "singletons" / "dataset_manifest.json"
    embedding_path = config.handoff_root / "embeddings" / "embeddinggemma_300m.normalized.f32"
    dense = read_json(dense_path)
    singleton_manifest = read_json(singleton_manifest_path)
    shards = dense.get("shards")
    require(isinstance(shards, list) and shards, "INVALID_DENSE_MANIFEST", "dense manifest contains no shards")

    pq = _parquet()
    import pyarrow as pa

    resolved_shards: list[dict[str, Any]] = []
    expected_start = 0
    total_rows = 0
    required_columns = {
        "source_id",
        "warc_path",
        "warc_record_id",
        "warc_record_index",
        "url",
        "timestamp",
        "language",
        "text",
    }
    for shard_index, shard in enumerate(shards):
        require(
            isinstance(shard, dict),
            "INVALID_DENSE_MANIFEST",
            "shard record must be an object",
            shard_index=shard_index,
        )
        source_key = str(shard.get("source_key", ""))
        local_path = _resolved_shard_path(config.handoff_root, source_key)

        require(
            local_path.is_file(), "SHARD_NOT_FOUND", "manifest-selected text shard is missing", path=str(local_path)
        )
        start_id = int(shard["start_id"])
        end_id = int(shard["end_id"])
        rows = int(shard["rows"])

        require(
            start_id == expected_start,
            "NONCONTIGUOUS_ID_RANGE",
            "manifest shard ranges are not contiguous",
            shard_index=shard_index,
        )

        require(
            end_id - start_id + 1 == rows,
            "INVALID_SHARD_RANGE",
            "shard ID range does not match row count",
            shard_index=shard_index,
        )
        parquet_file = pq.ParquetFile(local_path)
        actual_rows = parquet_file.metadata.num_rows
        actual_columns = set(parquet_file.schema_arrow.names)
        schema = parquet_file.schema_arrow
        expected_types = {
            "source_id": pa.string(),
            "warc_path": pa.string(),
            "warc_record_id": pa.string(),
            "warc_record_index": pa.int64(),
            "url": pa.string(),
            "timestamp": pa.string(),
            "language": pa.string(),
            "text": pa.large_string(),
        }
        expected_non_nullable = {"source_id", "warc_path", "warc_record_index", "text"}

        require(
            actual_rows == rows,
            "SHARD_ROW_COUNT_MISMATCH",
            "Parquet row count differs from manifest",
            path=str(local_path),
        )

        require(
            required_columns.issubset(actual_columns),
            "SHARD_SCHEMA_MISMATCH",
            "Parquet shard is missing required columns",
            path=str(local_path),
            missing=sorted(required_columns - actual_columns),
        )
        require(
            config.retrieval.backend == "fixture_cpu"
            or all(schema.field(name).type == expected_type for name, expected_type in expected_types.items()),
            "SHARD_SCHEMA_MISMATCH",
            "Parquet shard field types differ from the frozen handoff schema",
            path=str(local_path),
        )
        require(
            config.retrieval.backend == "fixture_cpu"
            or all(not schema.field(name).nullable for name in expected_non_nullable),
            "SHARD_SCHEMA_MISMATCH",
            "required Parquet shard fields must be non-nullable",
            path=str(local_path),
        )
        metadata = schema.metadata or {}
        require(
            config.retrieval.backend == "fixture_cpu"
            or (
                metadata.get(b"dense_retrieval_output_schema") == b"2"
                and metadata.get(b"normalization_version") == b"common-crawl-extractor-output-v1"
            ),
            "SHARD_SCHEMA_METADATA_MISMATCH",
            "Parquet shard schema metadata differs from the frozen extractor contract",
            path=str(local_path),
        )
        if config.verify_checksums:
            actual_digest = sha256_file(local_path)

            require(
                actual_digest == shard["sha256"],
                "SHARD_CHECKSUM_MISMATCH",
                "manifest-selected text shard failed SHA-256 validation",
                path=str(local_path),
            )
        resolved_shards.append(
            {
                "shard_index": shard_index,
                "source_key": source_key,
                "upstream_path": shard.get("path"),
                "resolved_path": str(local_path),
                "start_id": start_id,
                "end_id": end_id,
                "rows": rows,
                "sha256": shard["sha256"],
            }
        )
        total_rows += rows
        expected_start = end_id + 1

    require(total_rows == config.dataset.expected_rows, "CORPUS_ROW_COUNT_MISMATCH", "frozen corpus count is invalid")
    require(
        expected_start == total_rows, "CORPUS_ID_DOMAIN_MISMATCH", "frozen ID domain is not zero-based and contiguous"
    )
    expected_bytes = config.dataset.embedding_rows * config.dataset.embedding_dimensions * 4
    require(
        embedding_path.is_file(),
        "EMBEDDING_NOT_FOUND",
        "normalized embedding matrix is missing",
        path=str(embedding_path),
    )
    require(
        embedding_path.stat().st_size == expected_bytes,
        "EMBEDDING_SIZE_MISMATCH",
        "embedding byte size does not match shape and dtype",
        expected_bytes=expected_bytes,
        actual_bytes=embedding_path.stat().st_size,
    )
    if config.verify_checksums:
        require(
            sha256_file(embedding_path) == config.dataset.embedding_sha256,
            "EMBEDDING_CHECKSUM_MISMATCH",
            "embedding matrix failed SHA-256 validation",
        )

    wrapper = {
        "schema_version": "corpus-manifest-v1",
        "dataset_version": config.dataset.dataset_version,
        "dataset_row_count": total_rows,
        "id_mapping_version": "manifest-start-id-plus-physical-row-v1",
        "id_field": "doc_id",
        "upstream_id_field": "_curator_dedup_id",
        "normalization_version": "common-crawl-extractor-output-v1",
        "exact_dedup_applied_upstream": True,
        "exact_dedup_in_evaluation_scope": False,
        "exact_dedup_provenance_optional": True,
        "dense_manifest_path": str(dense_path),
        "dense_manifest_sha256": sha256_file(dense_path),
        "singleton_manifest_path": str(singleton_manifest_path),
        "singleton_manifest_sha256": sha256_file(singleton_manifest_path),
        "singleton_part_count": int(singleton_manifest.get("part_count", 0)),
        "shards": resolved_shards,
        "embedding": {
            "path": str(embedding_path),
            "sha256": config.dataset.embedding_sha256,
            "model": "google/embeddinggemma-300m",
            "revision": None,
            "rows": config.dataset.embedding_rows,
            "dimensions": config.dataset.embedding_dimensions,
            "dtype": config.dataset.embedding_dtype,
            "l2_normalized": True,
            "row_mapping": "matrix row N maps to doc_id N",
        },
        "provenance_availability": {
            "exact_dedup_config": False,
            "embedding_run_manifest": False,
            "embedding_completion_marker": False,
        },
        "corpus_level_claims_authorized": bool(
            dense.get("scale_semantics", {}).get("corpus_level_claims_authorized", False)
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(destination, wrapper)
    return wrapper


def register_sut_handoff(
    config: EvaluationConfig, corpus_manifest: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Validate groups/removals and write ``sut_run_manifest.json``."""

    run_identity_path = config.handoff_root / "fuzzy" / "run_identity.json"
    runtime_manifest_path = config.handoff_root / "fuzzy" / "runtime_manifest.json"
    groups_path = config.handoff_root / "fuzzy" / "duplicate_groups.parquet"
    removals_path = config.handoff_root / "fuzzy" / "removal_ids.parquet"
    run_identity = read_json(run_identity_path)
    arrays = load_sut_arrays(config, groups_path=groups_path, removals_path=removals_path)
    wrapper = {
        "schema_version": "sut-run-manifest-v1",
        "sut_run_id": str(run_identity["run_id"]),
        "repository_revision": run_identity.get("repository_revision"),
        "config_digest": run_identity.get("config_digest"),
        "runtime_manifest_digest": run_identity.get("runtime_manifest_digest"),
        "upstream_manifest_digest": run_identity.get("manifest_digest"),
        "corpus_manifest_sha256": sha256_file(destination.parent / "corpus_manifest.json"),
        "corpus_row_count": corpus_manifest["dataset_row_count"],
        "duplicate_groups": {
            "path": str(groups_path),
            "sha256": sha256_file(groups_path),
            "rows": arrays.grouped_document_count,
            "groups": arrays.group_count,
        },
        "removal_ids": {
            "path": str(removals_path),
            "sha256": sha256_file(removals_path),
            "rows": arrays.removal_count,
        },
        "logical_retained_documents": config.dataset.expected_retained,
        "resolved_config_available": False,
        "upstream_reproducibility_status": "conditionally_reproducible",
        "provenance_availability": {
            "runtime_manifest": runtime_manifest_path.is_file(),
            "resolved_config": False,
            "minhash_cache": False,
            "lsh_cache": False,
            "candidate_edges": False,
            "completion_markers": False,
        },
        "run_identity_path": str(run_identity_path),
        "run_identity_sha256": sha256_file(run_identity_path),
        "runtime_manifest_path": str(runtime_manifest_path),
        "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(destination, wrapper)
    return wrapper
