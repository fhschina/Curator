from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.dedup.analysis.comparison import build_pair_comparisons

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("left", "right", "expected"), [(None, None, None), (None, "en", None), ("en", "en", True), ("en", "zh", False)]
)
def test_optional_metadata_remains_unavailable_when_missing(
    tmp_path: Path, left: str | None, right: str | None, expected: bool | None
) -> None:
    candidate = {
        "canonical_pair_id": "pair",
        "doc_id_low": 0,
        "doc_id_high": 1,
        "token_count_low": 2,
        "token_count_high": 3,
        "language_low": left,
        "language_high": right,
        "hostname_low": left,
        "hostname_high": right,
    }
    pq.write_table(pa.Table.from_pylist([candidate]), tmp_path / "pairs.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"canonical_pair_id": "pair", "track": "5b"}]), tmp_path / "provenance.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "doc_id": index,
                    "predicted_cluster_key": f"singleton:{index}",
                    "predicted_group_size": 1,
                    "action": "KEEP",
                    "final_keeper_id": index,
                }
                for index in (0, 1)
            ]
        ),
        tmp_path / "outcomes.parquet",
    )
    (tmp_path / "errors.jsonl").write_text('{"canonical_pair_id":"pair","attempts":1}\n')
    build_pair_comparisons(
        candidate_pairs_path=tmp_path / "pairs.parquet",
        pair_provenance_path=tmp_path / "provenance.parquet",
        outcomes_path=tmp_path / "outcomes.parquet",
        judge_results_path=tmp_path / "results.jsonl",
        judge_errors_path=tmp_path / "errors.jsonl",
        destination=tmp_path / "comparisons.parquet",
    )
    row = pq.read_table(tmp_path / "comparisons.parquet").to_pylist()[0]
    assert row["same_language"] is expected
    assert row["same_hostname"] is expected
