from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from eval.dedup.core.config import ProfileConfig
from eval.dedup.core.validation import DedupEvaluationError, read_json
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.handoff.paths import adapt_inputs
from eval.dedup.pair_construction.prepare import build_population, preparation_config
from tests.eval.dedup.handoff.test_paths import make_inputs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@pytest.mark.parametrize("removal_budget", [4, 100])
def test_preparation_continues_outside_pilot_target_but_enforces_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    removal_budget: int,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    paths = make_inputs(tmp_path / "input")
    root = tmp_path / "run"
    dataset, corpus, sut, _ = adapt_inputs(root / "preparation", **paths)
    config = preparation_config(root, dataset)
    profile = ProfileConfig(
        "full",
        {"singleton": 2, "size_2": 2, "size_3_5": 0, "size_6_20": 0, "size_21_plus": 0},
        removal_budget,
        8,
        0,
        0,
        False,
    )
    config = replace(
        config,
        profiles={"full": profile},
        tokenizer=replace(config.tokenizer, kind="whitespace"),
        retrieval=replace(
            config.retrieval,
            backend="fixture_cpu",
            num_hashes=4,
            char_ngram_width=3,
            lsh_grid=((1, 1),),
            pilot_target_min=0,
            pilot_target_max=0,
            top_k=6,
        ),
    )
    if removal_budget == 100:
        with pytest.raises(DedupEvaluationError) as caught:
            build_population(config, corpus, sut, TokenCounter(config.tokenizer))
        assert caught.value.issue.code == "REMOVAL_BUDGET_UNFILLED"
        assert caught.value.issue.details == {"available": 6, "required": 100}
        assert not (config.output_root / "candidate_pairs.parquet").exists()
        return
    summary = build_population(config, corpus, sut, TokenCounter(config.tokenizer))
    assert summary["removal"]["rows"] == 4
    assert summary["cross_group"]["unique_selected_pairs"] == 8
    assert (config.output_root / "candidate_pairs.parquet").is_file()
    retrieval = read_json(config.output_root / "retrieval_config.json")
    assert retrieval["pilot_candidate_count_target"]["advisory"] is True
    assert retrieval["pilot_selection_policy"] == "prefer_target_then_closest_center"
    selected = next(trial for trial in retrieval["lexical_trials"] if trial["selected"])
    assert selected["within_target"] is False
    assert selected["median_cross_group_candidates"] > 0
    assert read_json(config.output_root / "lexical_pilot_summary.json")["trials"] == retrieval["lexical_trials"]


def baseline_population(root: Path):
    """Exercise shuffled real Parquet, optional metadata, and tokenizer windows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    paths = make_inputs(root / "input")
    for path in paths["documents"].glob("*.parquet"):
        values = pq.read_table(path).to_pydict()
        ids = values["_curator_dedup_id"]
        values["text"] = [
            "" if doc_id == 0 else ("hello world ! " * 80 if doc_id in {2, 15} else text)
            for doc_id, text in zip(ids, values["text"], strict=True)
        ]
        values["url"] = [None if doc_id % 3 == 0 else f"HTTPS://Example.COM:443/{doc_id}#fragment" for doc_id in ids]
        values["language"] = ["en" if doc_id % 2 else "zh" for doc_id in ids]
        pq.write_table(pa.table(values), path, row_group_size=3)
    dataset, corpus, sut, _ = adapt_inputs(root / "run/preparation", **paths)
    dataset = replace(dataset, dataset_version="baseline")
    sut = {**sut, "sut_run_id": "baseline"}
    config = preparation_config(root / "run", dataset)
    profile = ProfileConfig(
        "full", {"singleton": 2, "size_2": 2, "size_3_5": 0, "size_6_20": 0, "size_21_plus": 0}, 4, 8, 0, 0, False
    )
    config = replace(
        config,
        profiles={"full": profile},
        tokenizer=replace(config.tokenizer, kind="whitespace"),
        judge=replace(config.judge, max_visible_tokens=40, window_tokens=8, window_overlap_tokens=2),
        retrieval=replace(
            config.retrieval,
            backend="fixture_cpu",
            num_hashes=4,
            char_ngram_width=3,
            lsh_grid=((1, 1),),
            pilot_target_min=0,
            pilot_target_max=100,
            top_k=6,
        ),
    )
    unknown = "[UNK]"
    backend = Tokenizer(WordLevel({unknown: 0, "hello": 1, "world": 2, "!": 3}, unk_token=unknown))
    backend.pre_tokenizer = Whitespace()
    counter = TokenCounter(config.tokenizer)
    counter.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=unknown)
    return config, corpus, sut, counter


def test_selected_population_matches_full_scan_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from pathlib import Path

    import pyarrow.parquet as pq
    import torch

    from eval.dedup.handoff.corpus import load_documents_by_ids
    from eval.dedup.judging.payload import build_visible_payload

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    config, corpus, sut, counter = baseline_population(tmp_path)
    expected = json.loads((Path(__file__).parent / "fixtures/population.json").read_text())
    counted = []
    original = counter.count_many

    def count(texts: Sequence[str], **kwargs) -> list[int]:
        counted.extend(texts)
        return original(texts, **kwargs)

    monkeypatch.setattr(counter, "count_many", count)
    summary = build_population(config, corpus, sut, counter)
    for name in ("anchors", "removal_pairs", "cross_group_pairs", "candidate_pairs", "pair_provenance"):
        rows = pq.read_table(config.output_root / (name + ".parquet")).to_pylist()
        if name == "anchors":
            baseline = [{key: row[key] for key in rows[0]} for row in expected[name]]
        else:
            baseline = expected[name]
        assert rows == baseline
    outcomes = pq.read_table(config.output_root / "document_outcomes.parquet").sort_by("doc_id").to_pylist()
    assert outcomes == expected["document_outcomes"]
    assert len(counted) == len(outcomes) < config.dataset.expected_rows
    assert summary["outcomes"]["scope"] == "selected_pair_endpoints"
    assert "token_count" not in pq.read_schema(config.output_root / "document_relations.parquet").names
    documents = load_documents_by_ids(corpus, [row["doc_id"] for row in outcomes])
    by_id = {row["doc_id"]: row for row in outcomes}
    for row in expected["candidate_pairs"]:
        a, b = row["presented_doc_a"], row["presented_doc_b"]
        payload, digest = build_visible_payload(
            documents[a],
            documents[b],
            counter=counter,
            config=config.judge,
            token_counts=(by_id[a]["token_count"], by_id[b]["token_count"]),
        )
        assert payload == expected["payloads"][row["canonical_pair_id"]]
        assert digest == row["judge_payload_hash"]
    assert len(counted) == len(outcomes)
