from pathlib import Path

import pytest

from eval.dedup.pair_construction.outcomes import canonicalize_url_v0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://Example.COM:443/path?q=1#fragment", ("example.com", "https://example.com/path?q=1", True)),
        ("http://example.com:80/", ("example.com", "http://example.com/", True)),
        (None, (None, None, True)),
        ("not-a-url", (None, None, False)),
    ],
)
def test_conservative_url_canonicalization(raw: str | None, expected: tuple[str | None, str | None, bool]) -> None:
    assert canonicalize_url_v0(raw) == expected


def test_relations_need_no_text_and_selected_empty_length_is_exact(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from eval.dedup.pair_construction.outcomes import build_document_outcomes, build_document_relations
    from tests.eval.dedup.pair_construction.test_prepare import baseline_population

    config, corpus, sut, counter = baseline_population(tmp_path)
    build_document_outcomes(
        config,
        evaluation_manifest={"evaluation_run_id": "baseline"},
        corpus_manifest=corpus,
        sut_manifest=sut,
        destination=tmp_path / "selected.parquet",
        tokenizer=counter,
        doc_ids=[0, 2, 2],
    )
    rows = pq.read_table(tmp_path / "selected.parquet").to_pylist()
    assert len(rows) == 2
    assert rows[0]["token_count"] == rows[0]["char_count"] == 0
    assert rows[0]["length_bucket"] == "short"
    for shard in corpus["shards"]:
        Path(shard["resolved_path"]).unlink()
    summary = build_document_relations(
        config,
        evaluation_manifest={"evaluation_run_id": "baseline"},
        corpus_manifest=corpus,
        sut_manifest=sut,
        destination=tmp_path / "relations.parquet",
    )
    assert summary["rows"] == 24
    assert "token_count" not in pq.read_schema(tmp_path / "relations.parquet").names
