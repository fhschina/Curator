# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Adapt four local Parquet inputs to the existing evaluation handoff."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from eval.dedup.core.config import DatasetConfig
from eval.dedup.core.validation import require, sha256_file, sha256_json, write_json_atomic
from eval.dedup.handoff.sut import load_sut_arrays

ID = "_curator_dedup_id"
GROUP = "_duplicate_group_id"
REQUIRED_COLUMNS = {
    "documents": (ID, "text"),
    "groups": (ID, GROUP),
    "removals": (ID,),
    "embeddings": (ID, "embeddings"),
}


def parquet_files(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet")) if path.is_dir() else []
    require(bool(files), "INPUT_NOT_FOUND", "expected a Parquet file or directory of Parquet shards", path=str(path))
    return files


def _ids(array: Any, *, rows: int | None = None) -> Any:
    import numpy as np
    import pyarrow as pa

    require(pa.types.is_integer(array.type) and array.null_count == 0, "INVALID_ID", "IDs must be non-null integers")
    if len(array):
        import pyarrow.compute as pc

        bounds = pc.min_max(array).as_py()
        require(
            bounds["min"] >= 0 and bounds["max"] < (rows if rows is not None else 2**63),
            "ID_OUT_OF_RANGE",
            "document IDs must cover 0..N-1; group IDs must be nonnegative int64 values",
            minimum=bounds["min"],
            maximum=bounds["max"],
            rows=rows,
        )
    return array.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _mark_ids(ids: Any, seen: Any, *, kind: str) -> None:
    import numpy as np

    require(
        len(np.unique(ids)) == len(ids) and not seen[ids].any(),
        "DUPLICATE_INPUT_ID",
        "each document ID must appear exactly once in this input",
        input=kind,
    )
    seen[ids] = True


def _consolidate(files: list[Path], columns: tuple[str, ...], destination: Path, rows: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([pa.field(column, pa.int64(), nullable=False) for column in columns])
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(columns=list(columns)):
                arrays = [pa.array(_ids(batch[column], rows=rows if column == ID else None)) for column in columns]
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))


def adapt_inputs(root: Path, **paths: Path) -> tuple[DatasetConfig, dict, dict, dict]:
    """Validate IDs and vectors, and write only beneath a fresh run's preparation directory."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    require(set(paths) == set(REQUIRED_COLUMNS), "INPUT_PATHS_REQUIRED", "all four input paths are required")
    files = {kind: parquet_files(path) for kind, path in paths.items()}
    counts = {}
    for kind, shards in files.items():
        count = 0
        for path in shards:
            parquet = pq.ParquetFile(path)
            require(
                set(REQUIRED_COLUMNS[kind]) <= set(parquet.schema_arrow.names),
                "INPUT_SCHEMA",
                "Parquet input is missing required columns",
                input=kind,
                path=str(path),
                required=REQUIRED_COLUMNS[kind],
            )
            count += parquet.metadata.num_rows
        counts[kind] = count
    rows = counts["documents"]
    require(rows > 0, "EMPTY_CORPUS", "documents must contain at least one row")
    require(counts["embeddings"] == rows, "EMBEDDING_COVERAGE", "embedding row count must equal document count")
    first = next(
        batch
        for path in files["embeddings"]
        for batch in pq.ParquetFile(path).iter_batches(columns=["embeddings"], batch_size=1)
        if batch.num_rows
    )["embeddings"]
    require(
        (pa.types.is_list(first.type) or pa.types.is_large_list(first.type) or pa.types.is_fixed_size_list(first.type))
        and first.null_count == 0
        and (pa.types.is_floating(first.type.value_type) or pa.types.is_integer(first.type.value_type)),
        "EMBEDDING_SCHEMA",
        "embeddings must be non-null numeric vectors",
    )
    dimensions = len(first[0].as_py())
    require(dimensions > 0, "EMBEDDING_DIMENSION", "embedding vectors must not be empty")
    require(not root.exists(), "INPUT_WORKSPACE_EXISTS", "input conversion requires a fresh preparation directory")
    parent = root.parent
    while not parent.exists():
        parent = parent.parent
    unique_files = sorted({path for shards in files.values() for path in shards})
    stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in unique_files}
    # Include the existing 260-hash retrieval matrix, outcomes, and intermediate Parquet files.
    required_bytes = rows * (dimensions * 4 + 16 + 260 * 4 + 512) + sum(v[0] for v in stats.values()) + 2**30
    free_bytes = shutil.disk_usage(parent).free
    require(
        free_bytes >= required_bytes,
        "INSUFFICIENT_DISK_SPACE",
        "not enough free space for input conversion and pair preparation",
        required_bytes=required_bytes,
        free_bytes=free_bytes,
        filesystem=str(parent),
    )
    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.mkdir(mode=0o700)
    inventory = {
        "schema_version": "dedup-parquet-inputs-v1",
        "paths": {kind: str(path.expanduser().resolve()) for kind, path in paths.items()},
        "shards": {kind: [str(path) for path in shards] for kind, shards in files.items()},
        "files": {},
        "counts": counts,
        "embedding_dimensions": dimensions,
        "disk_estimate_bytes": required_bytes,
    }
    for index, path in enumerate(unique_files, 1):
        inventory["files"][str(path)] = {
            "size_bytes": stats[path][0],
            "mtime_ns": stats[path][1],
            "sha256": sha256_file(path),
        }
        if index % 10 == 0 or index == len(unique_files):
            print(json.dumps({"input_checksums": index, "files": len(unique_files)}), flush=True)
    identity = sha256_json(inventory)
    groups_path, removals_path = root / "groups.parquet", root / "removals.parquet"
    _consolidate(files["groups"], REQUIRED_COLUMNS["groups"], groups_path, rows)
    _consolidate(files["removals"], REQUIRED_COLUMNS["removals"], removals_path, rows)
    grouped = pq.read_table(groups_path, columns=[GROUP])[GROUP].to_numpy()
    group_count = len(np.unique(grouped))
    del grouped
    dataset = DatasetConfig(
        dataset_version="parquet-" + identity[:20],
        expected_rows=rows,
        expected_grouped_documents=counts["groups"],
        expected_groups=group_count,
        expected_removals=counts["removals"],
        expected_singletons=rows - counts["groups"],
        expected_retained=rows - counts["removals"],
        embedding_rows=rows,
        embedding_dimensions=dimensions,
        embedding_dtype="float32",
        embedding_sha256="",
    )
    load_sut_arrays(
        SimpleNamespace(dataset=dataset, retrieval=SimpleNamespace(backend="gpu_cudf")),
        groups_path=groups_path,
        removals_path=removals_path,
    )

    locator_path = root / "document_locations.i64"
    locator = np.memmap(locator_path, dtype=np.int64, mode="w+", shape=(rows, 2))
    seen = np.zeros(rows, dtype=bool)
    shards = []
    for index, path in enumerate(files["documents"]):
        parquet = pq.ParquetFile(path)
        text_type = parquet.schema_arrow.field("text").type
        require(
            pa.types.is_string(text_type) or pa.types.is_large_string(text_type), "TEXT_SCHEMA", "text must be strings"
        )
        offset = 0
        for batch in parquet.iter_batches(columns=[ID, "text"], batch_size=8192):
            ids = _ids(batch[ID], rows=rows)
            _mark_ids(ids, seen, kind="documents")
            require(batch["text"].null_count == 0, "NULL_TEXT", "document text must not be null")
            locator[ids, 0] = index
            locator[ids, 1] = np.arange(offset, offset + len(ids))
            offset += len(ids)
        shards.append({"shard_index": index, "resolved_path": str(path), "rows": offset})
        if (index + 1) % 10 == 0 or index + 1 == len(files["documents"]):
            print(json.dumps({"document_shards": index + 1, "files": len(files["documents"])}), flush=True)
    require(bool(seen.all()), "DOCUMENT_COVERAGE", "document IDs must cover 0..N-1")
    locator.flush()
    del locator

    seen.fill(False)
    embedding_path = root / "embeddings.f32"
    matrix = np.memmap(embedding_path, dtype=np.float32, mode="w+", shape=(rows, dimensions))
    for index, path in enumerate(files["embeddings"], 1):
        for batch in pq.ParquetFile(path).iter_batches(columns=[ID, "embeddings"], batch_size=8192):
            ids = _ids(batch[ID], rows=rows)
            _mark_ids(ids, seen, kind="embeddings")
            vectors = batch["embeddings"]
            require(
                (
                    pa.types.is_list(vectors.type)
                    or pa.types.is_large_list(vectors.type)
                    or pa.types.is_fixed_size_list(vectors.type)
                )
                and vectors.null_count == 0
                and (pa.types.is_floating(vectors.type.value_type) or pa.types.is_integer(vectors.type.value_type)),
                "EMBEDDING_SCHEMA",
                "embeddings must be non-null numeric vectors",
                path=str(path),
            )
            lengths = pc.list_value_length(vectors).to_numpy()
            require(
                bool(np.all(lengths == dimensions)),
                "EMBEDDING_DIMENSION",
                "embedding dimensions must agree",
                path=str(path),
            )
            flat = pc.list_flatten(vectors)
            require(flat.null_count == 0, "EMBEDDING_VALUES", "embedding elements must not be null")
            values = flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, dimensions)
            require(bool(np.isfinite(values).all()), "EMBEDDING_VALUES", "embedding values must be finite")
            norms = np.linalg.norm(values, axis=1)
            require(
                bool(np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3)),
                "EMBEDDING_NORMALIZATION",
                "embedding vectors must already have unit L2 norm",
                path=str(path),
            )
            matrix[ids] = values
        if index % 10 == 0 or index == len(files["embeddings"]):
            print(json.dumps({"embedding_shards": index, "files": len(files["embeddings"])}), flush=True)
    require(bool(seen.all()), "EMBEDDING_COVERAGE", "embeddings must cover every document ID")
    matrix.flush()
    del matrix, seen
    from dataclasses import replace

    dataset = replace(dataset, embedding_sha256=sha256_file(embedding_path))
    require(
        all((path.stat().st_size, path.stat().st_mtime_ns) == stats[path] for path in unique_files),
        "INPUT_CHANGED",
        "input files changed during preparation; use an immutable input snapshot",
    )
    corpus = {
        "explicit_id_column": ID,
        "rows": rows,
        "document_locations": str(locator_path),
        "shards": shards,
        "dense_manifest_sha256": identity,
        "embedding": {"path": str(embedding_path), "sha256": dataset.embedding_sha256},
    }
    sut = {
        "sut_run_id": "parquet-" + identity[:20],
        "duplicate_groups": {"path": str(groups_path)},
        "removal_ids": {"path": str(removals_path)},
    }
    for name, value in (("inputs", inventory), ("corpus", corpus), ("sut", sut)):
        write_json_atomic(root / f"{name}_manifest.json", value)
    return dataset, corpus, sut, inventory
