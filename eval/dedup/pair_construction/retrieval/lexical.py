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
import json
import os
from pathlib import Path
from typing import Any

from eval.dedup.core.config import EvaluationConfig, RetrievalConfig
from eval.dedup.core.validation import read_json, require, sha256_file, sha256_json, write_json_atomic
from eval.dedup.handoff.corpus import iter_corpus_batches


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
    for batch_index, batch in enumerate(
        iter_corpus_batches(corpus_manifest, columns=("text",), batch_size=retrieval.signature_chunk_rows), 1
    ):
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
        if corpus_manifest.get("explicit_id_column"):
            matrix[values["doc_id"]] = signatures
        else:
            start = int(values["doc_id"][0])
            require(start == written, "MINHASH_ID_ORDER_MISMATCH", "corpus batches are not in doc_id order")
            matrix[start : start + len(texts)] = signatures
        written += len(texts)
        if batch_index % 64 == 0:
            print(json.dumps({"minhash_rows": written, "expected_rows": config.dataset.expected_rows}), flush=True)
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


class LSHCandidateMatcher:
    """Match one ID batch at a time, deduplicating bands within that batch."""

    def __init__(
        self,
        signature_path: Path,
        *,
        row_count: int,
        num_hashes: int,
        anchor_ids: list[int],
        bands: int,
        rows_per_band: int,
    ) -> None:
        np = _numpy()
        self.signatures = np.memmap(signature_path, dtype=np.uint32, mode="r", shape=(row_count, num_hashes))
        self.anchor_ids = anchor_ids
        self.band_lookups = []
        for band in range(bands):
            start, end = band * rows_per_band, (band + 1) * rows_per_band
            anchor_band = np.ascontiguousarray(self.signatures[anchor_ids, start:end])
            lookup: dict[int, list[int]] = {}
            for anchor_index, value in enumerate(_band_hash(anchor_band).tolist()):
                lookup.setdefault(value, []).append(anchor_index)
            self.band_lookups.append((start, end, anchor_band, lookup, np.asarray(sorted(lookup), dtype=np.uint64)))

    def match(self, doc_ids: Any, *, group_ids: Any) -> dict[int, Any]:
        """Return eligible batch offsets per anchor, with self/same-group matches removed."""

        np = _numpy()
        doc_ids = np.asarray(doc_ids, dtype=np.int64)
        # IDs can arrive in physical Parquet order rather than signature row order.
        block = self.signatures[doc_ids, : self.band_lookups[-1][1]]
        masks: dict[int, Any] = {}
        for start, end, anchor_band, lookup, keys in self.band_lookups:
            block_band = block[:, start:end]
            hashes = _band_hash(block_band)
            matched = np.flatnonzero(np.isin(hashes, keys))
            if not len(matched):
                continue
            matched = matched[np.argsort(hashes[matched], kind="stable")]
            boundaries = np.flatnonzero(np.diff(hashes[matched])) + 1
            for offsets in np.split(matched, boundaries):
                for anchor_index in lookup[int(hashes[offsets[0]])]:
                    exact = offsets[np.all(block_band[offsets] == anchor_band[anchor_index], axis=1)]
                    if not len(exact):
                        continue
                    anchor_id = self.anchor_ids[anchor_index]
                    if anchor_id not in masks:
                        masks[anchor_id] = np.zeros(len(doc_ids), dtype=bool)
                    masks[anchor_id][exact] = True
        output = {}
        for anchor_id, mask in masks.items():
            eligible = mask & (doc_ids != anchor_id)
            if group_ids[anchor_id] != -1:
                eligible &= group_ids[doc_ids] != group_ids[anchor_id]
            offsets = np.flatnonzero(eligible)
            if len(offsets):
                output[anchor_id] = offsets
        return output


def choose_lsh_configuration(
    signature_path: Path,
    *,
    config: EvaluationConfig,
    pilot_anchor_ids: list[int],
    predicted_group_ids: Any,
) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    """Prefer medians inside the advisory target, then the closest to its center."""

    np = _numpy()
    trials: list[dict[str, Any]] = []
    require(pilot_anchor_ids, "LEXICAL_PILOT_EMPTY", "LSH pilot requires at least one anchor")
    for bands, rows_per_band in config.retrieval.lsh_grid:
        matcher = LSHCandidateMatcher(
            signature_path,
            row_count=config.dataset.expected_rows,
            num_hashes=config.retrieval.num_hashes,
            anchor_ids=pilot_anchor_ids,
            bands=bands,
            rows_per_band=rows_per_band,
        )
        candidate_counts = dict.fromkeys(pilot_anchor_ids, 0)
        for start in range(0, config.dataset.expected_rows, config.retrieval.signature_chunk_rows):
            ids = np.arange(start, min(start + config.retrieval.signature_chunk_rows, config.dataset.expected_rows))
            for anchor_id, offsets in matcher.match(ids, group_ids=predicted_group_ids).items():
                candidate_counts[anchor_id] += len(offsets)
        counts = list(candidate_counts.values())
        median = float(np.median(counts))
        trials.append(
            {
                "bands": bands,
                "rows_per_band": rows_per_band,
                "median_cross_group_candidates": median,
                "minimum": int(min(counts, default=0)),
                "maximum": int(max(counts, default=0)),
                "within_target": config.retrieval.pilot_target_min <= median <= config.retrieval.pilot_target_max,
            }
        )
        print(json.dumps({"lexical_pilot_trial": trials[-1]}), flush=True)
    require(trials, "LEXICAL_PILOT_EMPTY", "LSH pilot requires at least one grid configuration")
    selected = min(
        trials,
        key=lambda trial: (
            not trial["within_target"],
            abs(trial["median_cross_group_candidates"] - config.retrieval.pilot_target_center),
            trial["bands"],
            trial["rows_per_band"],
        ),
    )
    for trial in trials:
        trial["selected"] = trial is selected
    if not selected["within_target"]:
        print(
            json.dumps(
                {
                    "warning": "LEXICAL_PILOT_OUTSIDE_TARGET",
                    "message": "No LSH configuration met the advisory target; continuing with the closest median.",
                    "selected_lsh": selected,
                    "target_minimum": config.retrieval.pilot_target_min,
                    "target_maximum": config.retrieval.pilot_target_max,
                    "target_center": config.retrieval.pilot_target_center,
                }
            ),
            flush=True,
        )
    return (selected["bands"], selected["rows_per_band"]), trials


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
