from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.dedup.core.validation import DedupEvaluationError
from eval.dedup.handoff.corpus import iter_corpus_batches, load_documents_by_ids
from eval.dedup.handoff.paths import GROUP, ID, adapt_inputs


def make_inputs(root: Path, *, prefix: str = "fresh", combined: bool = False) -> dict[str, Path]:
    root.mkdir(parents=True)
    ids = np.arange(24)
    vectors = np.random.default_rng(17).normal(size=(24, 5)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    documents = root / "documents"
    documents.mkdir()
    embedding_dir = documents if combined else root / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    for index, selected in enumerate((ids[1::2][::-1], ids[::2])):
        data = {
            ID: selected,
            "text": [f"{prefix} example article about shared subject and document {item}" for item in selected],
        }
        if combined:
            data["embeddings"] = vectors[selected].tolist()
        pq.write_table(pa.table(data), documents / f"{index}.parquet", row_group_size=3)
    if not combined:
        for index, selected in enumerate((ids[:12][::-1], ids[12:])):
            pq.write_table(
                pa.table({ID: selected, "embeddings": vectors[selected].tolist()}),
                embedding_dir / f"{index}.parquet",
                row_group_size=4,
            )
    groups = root / "groups"
    groups.mkdir()
    for index, selected in enumerate((ids[:6][::-1], ids[6:12])):
        pq.write_table(pa.table({ID: selected, GROUP: selected // 2}), groups / f"{index}.parquet")
    removals = root / "removals.parquet"
    pq.write_table(pa.table({ID: ids[1:12:2]}), removals)
    return {"documents": documents, "groups": groups, "removals": removals, "embeddings": embedding_dir}


@pytest.mark.parametrize("combined", [True, False])
def test_shuffled_shards_align_text_and_embeddings_by_id(tmp_path: Path, combined: bool) -> None:
    paths = make_inputs(tmp_path / "input", combined=combined)
    dataset, corpus, sut, inventory = adapt_inputs(tmp_path / "work", **paths)
    assert (
        dataset.expected_rows,
        dataset.expected_groups,
        dataset.expected_removals,
        dataset.expected_singletons,
    ) == (24, 6, 6, 12)
    assert dataset.embedding_dimensions == 5
    assert all(value["sha256"] for value in inventory["files"].values())
    assert pq.read_table(sut["duplicate_groups"]["path"]).num_rows == 12
    matrix = np.memmap(corpus["embedding"]["path"], dtype=np.float32, shape=(24, 5))
    expected = np.random.default_rng(17).normal(size=(24, 5)).astype(np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_array_equal(matrix, expected)
    documents = load_documents_by_ids(corpus, [0, 3, 16, 23])
    assert {key: value["text"] for key, value in documents.items()} == {
        key: f"fresh example article about shared subject and document {key}" for key in [0, 3, 16, 23]
    }
    assert all(value["url"] is None and value["language"] is None for value in documents.values())
    streamed = [
        row
        for batch in iter_corpus_batches(corpus, columns=("text", "url"), batch_size=5)
        for row in batch.to_pylist()
    ]
    assert {row["doc_id"] for row in streamed} == set(range(24))
    assert all(row["text"].endswith(f"document {row['doc_id']}") and row["url"] is None for row in streamed)


@pytest.mark.parametrize(
    ("kind", "column", "change", "error"),
    [
        ("documents", ID, "duplicate", "DUPLICATE_INPUT_ID"),
        ("documents", ID, "missing", "ID_OUT_OF_RANGE"),
        ("documents", ID, "string", "INVALID_ID"),
        ("embeddings", ID, "duplicate", "DUPLICATE_INPUT_ID"),
        ("embeddings", ID, "missing", "ID_OUT_OF_RANGE"),
        ("embeddings", "embeddings", "dimension", "EMBEDDING_DIMENSION"),
        ("embeddings", "embeddings", "nan", "EMBEDDING_VALUES"),
        ("embeddings", "embeddings", "zero", "EMBEDDING_NORMALIZATION"),
        ("removals", ID, "unjoined", "REMOVAL_JOIN_INCOMPLETE"),
        ("removals", ID, "duplicate", "DUPLICATE_REMOVAL_ID"),
        ("removals", ID, "two_keepers", "AMBIGUOUS_GROUP_KEEPER"),
        ("removals", ID, "no_keeper", "AMBIGUOUS_GROUP_KEEPER"),
        ("groups", ID, "duplicate", "DUPLICATE_SUT_DOC_ID"),
    ],
)
def test_invalid_inputs_fail_before_pair_construction(
    tmp_path: Path, kind: str, column: str, change: str, error: str
) -> None:
    paths = make_inputs(tmp_path / "input")
    path = paths[kind] if paths[kind].is_file() else paths[kind] / "0.parquet"
    data = pq.read_table(path).to_pydict()
    values = data[column]
    if change == "duplicate":
        values[0] = values[1]
    elif change == "missing":
        values[0] = 24
    elif change == "string":
        data[column] = [str(value) for value in values]
    elif change == "dimension":
        values[-1] = values[-1][:-1]
    elif change == "nan":
        values[-1][0] = float("nan")
    elif change == "zero":
        values[-1] = [0.0] * 5
    elif change == "unjoined":
        values[0] = 23
    elif change == "two_keepers":
        values.pop()
    elif change == "no_keeper":
        values.append(0)
    pq.write_table(pa.table(data), path)
    with pytest.raises(DedupEvaluationError) as caught:
        adapt_inputs(tmp_path / "work", **paths)
    assert caught.value.issue.code == error
    assert not (tmp_path / "work/embeddings.f32").exists() or kind == "embeddings"


def test_disk_preflight_precedes_workspace_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    paths = make_inputs(tmp_path / "input")
    monkeypatch.setattr("eval.dedup.handoff.paths.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(DedupEvaluationError, match="INSUFFICIENT_DISK_SPACE"):
        adapt_inputs(tmp_path / "work", **paths)
    assert not (tmp_path / "work").exists()
