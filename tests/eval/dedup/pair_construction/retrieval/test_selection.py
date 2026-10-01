from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.dedup.pair_construction.retrieval.lexical import LSHCandidateMatcher, pair_features
from eval.dedup.pair_construction.retrieval.selection import _lexical_records

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("chunk_rows", [1, 17, 200])
@pytest.mark.parametrize("shuffled", [False, True])
def test_streaming_top50_equals_full_sort(tmp_path: Path, chunk_rows: int, shuffled: bool) -> None:
    rows = 151
    texts = ["zzzz", "abcdef", "abcde", "abcdefghi", "abXYef"] * 30 + ["abcdef"]
    texts[0] = "abcdef"
    texts[100] = ""
    ids = np.random.default_rng(18).permutation(rows) if shuffled else np.arange(rows)
    locations = np.empty((rows, 2), dtype=np.int64)
    shards = []
    for index, selected in enumerate(np.array_split(ids, 3)):
        path = tmp_path / f"documents-{index}.parquet"
        pq.write_table(
            pa.table({"_curator_dedup_id": selected, "text": [texts[doc] for doc in selected]}),
            path,
            row_group_size=13,
        )
        locations[selected, 0] = index
        locations[selected, 1] = np.arange(len(selected))
        shards.append({"resolved_path": str(path), "shard_index": index})
    locator = tmp_path / "locations.i64"
    locations.tofile(locator)
    corpus = {
        "rows": rows,
        "explicit_id_column": "_curator_dedup_id",
        "document_locations": str(locator),
        "shards": shards,
    }
    signatures = np.ones((rows, 4), dtype=np.uint32)
    signatures[147] = 8
    signatures[148:150] = 7
    signature_path = tmp_path / "signatures.u32"
    signatures.tofile(signature_path)
    anchors = [0, 100, 147, 149, 150]
    groups = np.full(rows, -1)
    groups[:6] = 4
    matcher = LSHCandidateMatcher(
        signature_path, row_count=rows, num_hashes=4, anchor_ids=anchors, bands=2, rows_per_band=2
    )
    actual, counts = _lexical_records(
        matcher, group_ids=groups, corpus_manifest=corpus, feature_width=2, top_k=50, chunk_rows=chunk_rows
    )
    expected = []
    for anchor in anchors:
        candidates = []
        for candidate in range(rows):
            if candidate == anchor or (groups[anchor] != -1 and groups[candidate] == groups[anchor]):
                continue
            if not any(
                np.array_equal(signatures[anchor, start : start + 2], signatures[candidate, start : start + 2])
                for start in (0, 2)
            ):
                continue
            candidates.append(
                {
                    "anchor_id": anchor,
                    "candidate_id": candidate,
                    **pair_features(texts[anchor], texts[candidate], ngram_width=2),
                }
            )
        assert counts[anchor] == len(candidates)
        candidates.sort(key=lambda row: (-row["jaccard"], -row["containment"], row["candidate_id"]))
        expected.extend({**row, "lexical_rank": rank} for rank, row in enumerate(candidates[:50], 1))
    assert actual == expected
    assert any(row["anchor_id"] == 0 and row["candidate_id"] == 150 for row in actual)
    assert counts[147] == 0
    assert counts[149] == 1
