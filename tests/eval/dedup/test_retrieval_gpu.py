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

from eval.dedup.config import RetrievalConfig
from eval.dedup.pair_construction.retrieval.lexical import _gpu_signatures
from eval.dedup.pair_construction.retrieval.semantic import exact_cosine_topk


@pytest.mark.gpu
def test_exact_semantic_topk_matches_gpu_oracle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
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
    scores = matrix @ matrix[0]
    oracle = [
        int(index)
        for index in np.argsort(-scores, kind="stable")
        if index != 0 and np.asarray([10, 10, -1, -1])[index] != 10
    ][:2]
    assert [doc_id for doc_id, _, _ in result[0]] == oracle


@pytest.mark.gpu
def test_gpu_minhash_is_deterministic() -> None:
    pytest.importorskip("cudf")
    retrieval = RetrievalConfig(
        backend="gpu_cudf",
        minhash_seed=42,
        char_ngram_width=3,
        num_hashes=16,
        feature_ngram_width=5,
        lsh_grid=((4, 4),),
        pilot_target_min=1,
        pilot_target_max=10,
        pilot_target_center=5,
        top_k=2,
        signature_chunk_rows=2,
        semantic_chunk_rows=2,
        max_candidates_per_anchor=100,
    )
    texts = ["alpha beta gamma", "alpha beta delta"]
    first = _gpu_signatures(texts, retrieval)
    second = _gpu_signatures(texts, retrieval)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 16)
    assert first.dtype == np.uint32
