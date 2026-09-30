from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from eval.dedup.core.config import ProfileConfig
from eval.dedup.core.validation import DedupEvaluationError
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.handoff.paths import adapt_inputs
from eval.dedup.pair_construction.prepare import build_population, preparation_config
from tests.eval.dedup.handoff.test_paths import make_inputs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("failure", ["LEXICAL_PILOT_FAILED", "REMOVAL_BUDGET_UNFILLED"])
def test_preparation_reports_unfillable_frozen_parameters(tmp_path: Path, failure: str) -> None:
    paths = make_inputs(tmp_path / "input")
    root = tmp_path / "run"
    dataset, corpus, sut, _ = adapt_inputs(root / "preparation", **paths)
    config = preparation_config(root, dataset)
    profile = ProfileConfig(
        "full",
        {"singleton": 2, "size_2": 2, "size_3_5": 0, "size_6_20": 0, "size_21_plus": 0},
        4 if failure == "LEXICAL_PILOT_FAILED" else 100,
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
        ),
    )
    with pytest.raises(DedupEvaluationError) as caught:
        build_population(config, corpus, sut, TokenCounter(config.tokenizer))
    assert caught.value.issue.code == failure
    details = caught.value.issue.details
    if failure == "LEXICAL_PILOT_FAILED":
        assert details["trials"][0]["median_cross_group_candidates"] > 0
    else:
        assert details == {"available": 6, "required": 100}
    assert not (config.output_root / "candidate_pairs.parquet").exists()
