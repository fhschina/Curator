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
