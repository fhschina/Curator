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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval.dedup.pair_construction.retrieval.lexical import (
    _cpu_signature,
    char_shingles,
    pair_features,
    pair_features_from_shingles,
)
from eval.dedup.pair_construction.retrieval.semantic import exact_cosine_topk


def test_fixture_minhash_is_deterministic() -> None:
    first = _cpu_signature("alpha beta gamma", num_hashes=16, ngram_width=3, seed=42)
    second = _cpu_signature("alpha beta gamma", num_hashes=16, ngram_width=3, seed=42)
    np.testing.assert_array_equal(first, second)


def test_precomputed_shingle_features_match_pair_features() -> None:
    left = "a repeated template and unique left text"
    right = "a repeated template and unique right text"
    expected = pair_features(left, right, ngram_width=5)

    actual = pair_features_from_shingles(
        char_shingles(left, 5),
        char_shingles(right, 5),
        left_text_length=len(left),
        right_text_length=len(right),
    )

    assert actual == expected


def test_exact_semantic_topk_matches_cpu_oracle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    matrix = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    path = tmp_path / "embeddings.f32"
    matrix.tofile(path)
    result = exact_cosine_topk(
        path,
        rows=4,
        dimensions=2,
        anchor_ids=[0],
        predicted_group_ids=np.asarray([10, 10, -1, -1], dtype=np.int64),
        top_k=2,
        chunk_rows=2,
    )
    assert [doc_id for doc_id, _, _ in result[0]] == [2, 3]
    assert [score for _, score, _ in result[0]] == pytest.approx([0.0, -1.0])
