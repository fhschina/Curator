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

"""Exact cosine retrieval over the frozen normalized embedding matrix."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from eval.dedup.validation import require


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        msg = "numpy is required for semantic retrieval"
        raise RuntimeError(msg) from exc
    return np


def _cpu_topk(
    matrix: Any,
    queries: Any,
    *,
    anchor_ids: list[int],
    predicted_group_ids: Any,
    start: int,
    end: int,
    top_k: int,
    chunk_rows: int,
) -> tuple[Any, Any]:
    np = _numpy()
    best_scores = np.full((len(queries), top_k), -np.inf, dtype=np.float32)
    best_ids = np.full((len(queries), top_k), -1, dtype=np.int64)
    for chunk_start in range(start, end, chunk_rows):
        chunk_end = min(end, chunk_start + chunk_rows)
        scores = queries @ np.asarray(matrix[chunk_start:chunk_end]).T
        candidate_ids = np.arange(chunk_start, chunk_end, dtype=np.int64)
        candidate_groups = predicted_group_ids[chunk_start:chunk_end]
        for query_index, anchor_id in enumerate(anchor_ids):
            excluded = candidate_ids == anchor_id
            anchor_group = predicted_group_ids[anchor_id]
            if anchor_group != -1:
                excluded |= candidate_groups == anchor_group
            scores[query_index, excluded] = -np.inf
        candidate_k = min(top_k, scores.shape[1])
        local = np.argpartition(scores, -candidate_k, axis=1)[:, -candidate_k:]
        local_scores = np.take_along_axis(scores, local, axis=1)
        local_ids = local + chunk_start
        merged_scores = np.concatenate((best_scores, local_scores), axis=1)
        merged_ids = np.concatenate((best_ids, local_ids), axis=1)
        selected = np.argpartition(merged_scores, -top_k, axis=1)[:, -top_k:]
        best_scores = np.take_along_axis(merged_scores, selected, axis=1)
        best_ids = np.take_along_axis(merged_ids, selected, axis=1)
    return best_scores, best_ids


def _gpu_topk(
    matrix_path: Path,
    queries: Any,
    *,
    anchor_ids: list[int],
    predicted_group_ids: Any,
    rows: int,
    dimensions: int,
    start: int,
    end: int,
    top_k: int,
    chunk_rows: int,
    device: int,
) -> tuple[Any, Any]:
    np = _numpy()
    import torch

    matrix = np.memmap(matrix_path, dtype=np.float32, mode="r", shape=(rows, dimensions))
    with torch.cuda.device(device):
        query_tensor = torch.as_tensor(queries, device=f"cuda:{device}")
        best_scores = torch.full((len(queries), top_k), -torch.inf, dtype=torch.float32, device=device)
        best_ids = torch.full((len(queries), top_k), -1, dtype=torch.int64, device=device)
        for chunk_start in range(start, end, chunk_rows):
            chunk_end = min(end, chunk_start + chunk_rows)
            corpus = torch.as_tensor(np.asarray(matrix[chunk_start:chunk_end]).copy(), device=device)
            scores = query_tensor @ corpus.T
            candidate_ids = torch.arange(chunk_start, chunk_end, device=device)
            candidate_groups = torch.as_tensor(predicted_group_ids[chunk_start:chunk_end].copy(), device=device)
            anchor_id_tensor = torch.as_tensor(anchor_ids, device=device)
            anchor_groups = torch.as_tensor(predicted_group_ids[anchor_ids].copy(), device=device)
            excluded = candidate_ids.unsqueeze(0) == anchor_id_tensor.unsqueeze(1)
            excluded |= (anchor_groups.unsqueeze(1) != -1) & (
                candidate_groups.unsqueeze(0) == anchor_groups.unsqueeze(1)
            )
            scores.masked_fill_(excluded, -torch.inf)
            local_scores, local_ids = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)
            local_ids += chunk_start
            merged_scores = torch.cat((best_scores, local_scores), dim=1)
            merged_ids = torch.cat((best_ids, local_ids), dim=1)
            best_scores, selected = torch.topk(merged_scores, k=top_k, dim=1)
            best_ids = torch.gather(merged_ids, 1, selected)
        return best_scores.cpu().numpy(), best_ids.cpu().numpy()


def exact_cosine_topk(
    matrix_path: Path,
    *,
    rows: int,
    dimensions: int,
    anchor_ids: list[int],
    predicted_group_ids: Any,
    top_k: int,
    chunk_rows: int,
) -> dict[int, list[tuple[int, float, int]]]:
    """Return exact top-k neighbors, partitioning the corpus over all visible GPUs."""

    np = _numpy()
    expected_bytes = rows * dimensions * 4
    require(
        matrix_path.stat().st_size == expected_bytes, "EMBEDDING_SIZE_MISMATCH", "semantic input matrix is incomplete"
    )
    matrix = np.memmap(matrix_path, dtype=np.float32, mode="r", shape=(rows, dimensions))
    queries = np.asarray(matrix[anchor_ids]).copy()
    try:
        import torch

        device_count = torch.cuda.device_count()
    except ImportError:
        device_count = 0
    if device_count == 0:
        partials = [
            _cpu_topk(
                matrix,
                queries,
                anchor_ids=anchor_ids,
                predicted_group_ids=predicted_group_ids,
                start=0,
                end=rows,
                top_k=top_k,
                chunk_rows=chunk_rows,
            )
        ]
    else:
        ranges = [(index * rows // device_count, (index + 1) * rows // device_count) for index in range(device_count)]
        with ThreadPoolExecutor(max_workers=device_count) as executor:
            futures = [
                executor.submit(
                    _gpu_topk,
                    matrix_path,
                    queries,
                    anchor_ids=anchor_ids,
                    predicted_group_ids=predicted_group_ids,
                    rows=rows,
                    dimensions=dimensions,
                    start=start,
                    end=end,
                    top_k=top_k,
                    chunk_rows=chunk_rows,
                    device=device,
                )
                for device, (start, end) in enumerate(ranges)
            ]
            partials = [future.result() for future in futures]
    all_scores = np.concatenate([item[0] for item in partials], axis=1)
    all_ids = np.concatenate([item[1] for item in partials], axis=1)
    result: dict[int, list[tuple[int, float, int]]] = {}
    for query_index, anchor_id in enumerate(anchor_ids):
        candidates = [
            item
            for item in sorted(
                zip(all_ids[query_index].tolist(), all_scores[query_index].tolist(), strict=True),
                key=lambda item: (-item[1], item[0]),
            )
            if item[0] >= 0 and np.isfinite(item[1])
        ][:top_k]
        require(
            len(candidates) == top_k,
            "SEMANTIC_TOPK_INCOMPLETE",
            "fewer than top_k cross-group neighbors are available",
            anchor_id=anchor_id,
        )
        result[anchor_id] = [
            (int(doc_id), float(score), rank) for rank, (doc_id, score) in enumerate(candidates, start=1)
        ]
    return result
