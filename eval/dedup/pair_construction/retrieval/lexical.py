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

"""Relaxed anchor-centric MinHash/LSH retrieval for Step 5b."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from eval.dedup.config import EvaluationConfig, RetrievalConfig
from eval.dedup.handoff.corpus import iter_corpus_batches
from eval.dedup.validation import read_json, require, sha256_file, sha256_json, write_json_atomic


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        msg = "numpy is required for lexical retrieval"
        raise RuntimeError(msg) from exc
    return np


def _cpu_signature(text: str, *, num_hashes: int, ngram_width: int, seed: int) -> Any:
    """Small-fixture reference implementation; production uses cuDF MinHash."""

    import hashlib

    np = _numpy()
    shingles = {text[index : index + ngram_width] for index in range(max(1, len(text) - ngram_width + 1))}
    signature = np.full(num_hashes, np.iinfo(np.uint32).max, dtype=np.uint32)
    for shingle in shingles:
        raw = shingle.encode("utf-8")
        for index in range(num_hashes):
            digest = hashlib.blake2s(
                raw, digest_size=4, person=(seed + index).to_bytes(8, "little", signed=False)
            ).digest()
            signature[index] = min(signature[index], int.from_bytes(digest, "little"))
    return signature


def _gpu_signatures(texts: list[str], retrieval: RetrievalConfig) -> Any:
    try:
        import cudf
        import cupy as cp

        from nemo_curator.stages.deduplication.fuzzy.minhash import GPUMinHash
    except ImportError as exc:
        msg = "production MinHash requires the deduplication_cuda12 environment"
        raise RuntimeError(msg) from exc
    processor = GPUMinHash(
        seed=retrieval.minhash_seed,
        num_hashes=retrieval.num_hashes,
        char_ngrams=retrieval.char_ngram_width,
        use_64bit_hash=False,
    )
    result = processor.compute_minhashes(cudf.Series(texts))
    leaves = result.list.leaves
    return cp.asnumpy(leaves.values).reshape(len(texts), retrieval.num_hashes).astype("uint32", copy=False)


def _minhash_implementation_contract(backend: str) -> dict[str, str]:
    if backend == "fixture_cpu":
        return {"name": "fixture-blake2s-minhash-v1", "source_sha256": sha256_file(__file__)}
    from nemo_curator.stages.deduplication.fuzzy.minhash import GPUMinHash

    source = inspect.getsourcefile(GPUMinHash)
    require(source is not None, "MINHASH_IMPLEMENTATION_UNKNOWN", "cannot locate GPUMinHash source")
    return {
        "name": "nemo-curator-gpu-minhash32",
        "source_path": str(Path(source).resolve()),
        "source_sha256": sha256_file(source),
    }


def build_minhash_cache(
    config: EvaluationConfig,
    *,
    corpus_manifest: dict[str, Any],
    cache_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Build or validate a contiguous row-N-to-doc-N signature matrix."""

    np = _numpy()
    retrieval = config.retrieval
    contract = {
        "schema_version": 1,
        "dataset_manifest_sha256": corpus_manifest["dense_manifest_sha256"],
        "rows": config.dataset.expected_rows,
        "num_hashes": retrieval.num_hashes,
        "dtype": "uint32",
        "seed": retrieval.minhash_seed,
        "char_ngram_width": retrieval.char_ngram_width,
        "backend": retrieval.backend,
        "implementation": _minhash_implementation_contract(retrieval.backend),
    }
    digest = sha256_json(contract)
    destination = cache_dir / "minhash" / digest[:20]
    matrix_path = destination / "signatures.u32"
    manifest_path = destination / "manifest.json"
    expected_bytes = config.dataset.expected_rows * retrieval.num_hashes * 4
    if matrix_path.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        require(
            manifest.get("matrix_sha256") == sha256_file(matrix_path),
            "MINHASH_CACHE_CHECKSUM_MISMATCH",
            "MinHash signature cache failed SHA-256 validation",
        )
        require(manifest.get("contract_digest") == digest, "MINHASH_CACHE_MISMATCH", "MinHash cache contract differs")
        require(
            matrix_path.stat().st_size == expected_bytes, "MINHASH_CACHE_SIZE_MISMATCH", "MinHash cache is incomplete"
        )
        return matrix_path, manifest
    require(
        not destination.exists(),
        "INCOMPLETE_MINHASH_CACHE",
        "partial MinHash cache requires a new cache root",
        path=str(destination),
    )
    destination.mkdir(parents=True)
    temporary = destination / ".signatures.u32.tmp"
    matrix = np.memmap(
        temporary, dtype=np.uint32, mode="w+", shape=(config.dataset.expected_rows, retrieval.num_hashes)
    )
    written = 0
    for batch in iter_corpus_batches(corpus_manifest, columns=("text",), batch_size=retrieval.signature_chunk_rows):
        values = batch.to_pydict()
        texts = values["text"]
        if retrieval.backend == "fixture_cpu":
            signatures = np.stack(
                [
                    _cpu_signature(
                        text,
                        num_hashes=retrieval.num_hashes,
                        ngram_width=retrieval.char_ngram_width,
                        seed=retrieval.minhash_seed,
                    )
                    for text in texts
                ]
            )
        else:
            signatures = _gpu_signatures(texts, retrieval)
        start = int(values["doc_id"][0])
        require(start == written, "MINHASH_ID_ORDER_MISMATCH", "corpus batches are not in doc_id order")
        matrix[start : start + len(texts)] = signatures
        written += len(texts)
    matrix.flush()
    del matrix
    require(written == config.dataset.expected_rows, "MINHASH_ROW_COUNT_MISMATCH", "signature cache is incomplete")
    os.replace(temporary, matrix_path)
    manifest = {
        **contract,
        "contract_digest": digest,
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "size_bytes": expected_bytes,
    }
    write_json_atomic(manifest_path, manifest)
    return matrix_path, manifest


def _band_hash(matrix: Any) -> Any:
    np = _numpy()
    result = np.full(matrix.shape[0], 1469598103934665603, dtype=np.uint64)
    for column in range(matrix.shape[1]):
        result ^= matrix[:, column].astype(np.uint64) + np.uint64(column * 0x9E3779B1)
        result *= np.uint64(1099511628211)
    return result


def lsh_candidates(
    signature_path: Path,
    *,
    row_count: int,
    num_hashes: int,
    anchor_ids: list[int],
    bands: int,
    rows_per_band: int,
    chunk_rows: int,
    max_candidates_per_anchor: int,
) -> dict[int, set[int]]:
    """Find documents sharing at least one exact MinHash band with each anchor."""

    np = _numpy()
    signatures = np.memmap(signature_path, dtype=np.uint32, mode="r", shape=(row_count, num_hashes))
    anchor_matrix = np.asarray(signatures[anchor_ids])
    candidates = {anchor_id: set() for anchor_id in anchor_ids}
    for band in range(bands):
        start = band * rows_per_band
        end = start + rows_per_band
        anchor_band = np.ascontiguousarray(anchor_matrix[:, start:end])
        anchor_hashes = _band_hash(anchor_band)
        lookup: dict[int, list[int]] = {}
        for anchor_index, value in enumerate(anchor_hashes.tolist()):
            lookup.setdefault(int(value), []).append(anchor_index)
        keys = np.asarray(sorted(lookup), dtype=np.uint64)
        for block_start in range(0, row_count, chunk_rows):
            block_end = min(row_count, block_start + chunk_rows)
            block_band = np.ascontiguousarray(signatures[block_start:block_end, start:end])
            hashes = _band_hash(block_band)
            matched = np.nonzero(np.isin(hashes, keys))[0]
            for local_index in matched.tolist():
                doc_id = block_start + local_index
                for anchor_index in lookup[int(hashes[local_index])]:
                    if np.array_equal(block_band[local_index], anchor_band[anchor_index]):
                        candidates[anchor_ids[anchor_index]].add(doc_id)
                        require(
                            len(candidates[anchor_ids[anchor_index]]) <= max_candidates_per_anchor,
                            "LEXICAL_CANDIDATE_EXPLOSION",
                            "LSH candidate count exceeded the frozen safety limit",
                            anchor_id=anchor_ids[anchor_index],
                            limit=max_candidates_per_anchor,
                        )
    for anchor_id in anchor_ids:
        candidates[anchor_id].discard(anchor_id)
    return candidates


def choose_lsh_configuration(
    signature_path: Path,
    *,
    config: EvaluationConfig,
    pilot_anchor_ids: list[int],
    predicted_group_ids: Any,
) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    """Run the frozen pilot grid and select the valid median closest to 35."""

    np = _numpy()
    trials: list[dict[str, Any]] = []
    valid: list[tuple[float, int, int]] = []
    for bands, rows_per_band in config.retrieval.lsh_grid:
        raw = lsh_candidates(
            signature_path,
            row_count=config.dataset.expected_rows,
            num_hashes=config.retrieval.num_hashes,
            anchor_ids=pilot_anchor_ids,
            bands=bands,
            rows_per_band=rows_per_band,
            chunk_rows=config.retrieval.signature_chunk_rows,
            max_candidates_per_anchor=config.retrieval.max_candidates_per_anchor,
        )
        counts = []
        for anchor_id, ids in raw.items():
            anchor_group = predicted_group_ids[anchor_id]
            filtered = [doc_id for doc_id in ids if anchor_group == -1 or predicted_group_ids[doc_id] != anchor_group]
            counts.append(len(filtered))
        median = float(np.median(counts))
        trials.append(
            {
                "bands": bands,
                "rows_per_band": rows_per_band,
                "median_cross_group_candidates": median,
                "minimum": int(min(counts, default=0)),
                "maximum": int(max(counts, default=0)),
            }
        )
        if config.retrieval.pilot_target_min <= median <= config.retrieval.pilot_target_max:
            valid.append((abs(median - config.retrieval.pilot_target_center), bands, rows_per_band))
    require(valid, "LEXICAL_PILOT_FAILED", "no frozen LSH configuration met the 20-50 candidate target", trials=trials)
    _, bands, rows_per_band = min(valid)
    return (bands, rows_per_band), trials


def char_shingles(text: str, width: int) -> set[str]:
    if len(text) < width:
        return {text}
    return {text[index : index + width] for index in range(len(text) - width + 1)}


def pair_features_from_shingles(
    left: set[str],
    right: set[str],
    *,
    left_text_length: int,
    right_text_length: int,
) -> dict[str, float]:
    """Compute exact pair features from precomputed character-shingle sets."""

    intersection = len(left & right)
    union = len(left | right)
    shortest = min(len(left), len(right))
    longest_text = max(left_text_length, right_text_length)
    return {
        "jaccard": intersection / union if union else 1.0,
        "containment": intersection / shortest if shortest else 1.0,
        "length_ratio": min(left_text_length, right_text_length) / longest_text if longest_text else 1.0,
    }


def pair_features(text_a: str, text_b: str, *, ngram_width: int) -> dict[str, float]:
    return pair_features_from_shingles(
        char_shingles(text_a, ngram_width),
        char_shingles(text_b, ngram_width),
        left_text_length=len(text_a),
        right_text_length=len(text_b),
    )
