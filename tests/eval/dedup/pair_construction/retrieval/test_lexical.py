from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from eval.dedup.core.validation import DedupEvaluationError
from eval.dedup.handoff.paths import adapt_inputs
from eval.dedup.pair_construction.prepare import preparation_config
from eval.dedup.pair_construction.retrieval.lexical import LSHCandidateMatcher, choose_lsh_configuration
from tests.eval.dedup.handoff.test_paths import make_inputs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("rows_per_band", [1, 2])
@pytest.mark.parametrize("collide", [False, True])
def test_batches_match_exact_band_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows_per_band: int, collide: bool
) -> None:
    signatures = np.random.default_rng(5).integers(0, 3, size=(83, 6), dtype=np.uint32)
    signatures[1] = signatures[0]
    path = tmp_path / "signatures.u32"
    signatures.tofile(path)
    anchors = [0, 1, 41, 79]
    groups = np.full(83, -1)
    groups[:5] = 12
    if collide:
        monkeypatch.setattr(
            "eval.dedup.pair_construction.retrieval.lexical._band_hash",
            lambda block: np.zeros(len(block), dtype=np.uint64),
        )
    matcher = LSHCandidateMatcher(
        path, row_count=83, num_hashes=6, anchor_ids=anchors, bands=3, rows_per_band=rows_per_band
    )
    found = {anchor: [] for anchor in anchors}
    order = np.random.default_rng(7).permutation(83)
    for ids in np.array_split(order, 11):
        for anchor, offsets in matcher.match(ids, group_ids=groups).items():
            found[anchor].extend(ids[offsets].tolist())
    for anchor in anchors:
        expected = [
            doc
            for doc in range(83)
            if doc != anchor
            and (groups[anchor] == -1 or groups[doc] != groups[anchor])
            and any(
                np.array_equal(
                    signatures[doc, start : start + rows_per_band], signatures[anchor, start : start + rows_per_band]
                )
                for start in range(0, 3 * rows_per_band, rows_per_band)
            )
        ]
        assert sorted(found[anchor]) == expected


def test_pilot_counts_and_choice_match_full_union(tmp_path: Path) -> None:
    dataset, _, _, _ = adapt_inputs(tmp_path / "work", **make_inputs(tmp_path / "input"))
    config = preparation_config(tmp_path / "run", dataset)
    config = replace(
        config,
        retrieval=replace(
            config.retrieval,
            num_hashes=4,
            lsh_grid=((1, 1), (2, 2), (3, 1)),
            signature_chunk_rows=5,
            pilot_target_min=0,
            pilot_target_max=100,
            pilot_target_center=12,
            max_candidates_per_anchor=1,
        ),
    )
    signatures = np.random.default_rng(5).integers(0, 3, size=(24, 4), dtype=np.uint32)
    path = tmp_path / "signatures.u32"
    signatures.tofile(path)
    groups = np.full(24, -1)
    groups[:6] = 3
    anchors = [0, 8, 15, 23]
    expected_trials = []
    for bands, width in config.retrieval.lsh_grid:
        counts = [
            sum(
                doc != anchor
                and (groups[anchor] == -1 or groups[doc] != groups[anchor])
                and any(
                    np.array_equal(signatures[doc, start : start + width], signatures[anchor, start : start + width])
                    for start in range(0, bands * width, width)
                )
                for doc in range(24)
            )
            for anchor in anchors
        ]
        expected_trials.append(
            {
                "bands": bands,
                "rows_per_band": width,
                "median_cross_group_candidates": float(np.median(counts)),
                "minimum": min(counts),
                "maximum": max(counts),
                "within_target": True,
            }
        )
    selected, trials = choose_lsh_configuration(
        path, config=config, pilot_anchor_ids=anchors, predicted_group_ids=groups
    )
    best = min(
        expected_trials,
        key=lambda row: (abs(row["median_cross_group_candidates"] - 12), row["bands"], row["rows_per_band"]),
    )
    for trial in expected_trials:
        trial["selected"] = trial is best
    assert trials == expected_trials
    assert selected == (best["bands"], best["rows_per_band"])


def test_pilot_counts_over_500k_without_candidate_cap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset, _, _, _ = adapt_inputs(tmp_path / "work", **make_inputs(tmp_path / "input"))
    config = preparation_config(tmp_path / "run", dataset)
    rows = 500_002
    config = replace(
        config,
        dataset=replace(config.dataset, expected_rows=rows),
        retrieval=replace(config.retrieval, num_hashes=1, lsh_grid=((1, 1),)),
    )
    path = tmp_path / "signatures.u32"
    np.ones((rows, 1), dtype=np.uint32).tofile(path)
    selected, trials = choose_lsh_configuration(
        path, config=config, pilot_anchor_ids=[0], predicted_group_ids=np.full(rows, -1)
    )
    assert selected == (1, 1)
    assert trials == [
        {
            "bands": 1,
            "rows_per_band": 1,
            "median_cross_group_candidates": 500001.0,
            "minimum": 500001,
            "maximum": 500001,
            "within_target": False,
            "selected": True,
        }
    ]
    warning = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert warning["warning"] == "LEXICAL_PILOT_OUTSIDE_TARGET"
    assert warning["selected_lsh"] == trials[0]


@pytest.mark.parametrize(
    ("target", "expected_bands", "within_target"),
    [((5, 12, 5), 2, True), ((0, 3, 2), 1, False), ((13, 20, 16), 3, False), ((5, 7, 6), 1, False)],
)
def test_pilot_prefers_target_then_closest_median(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: tuple[int, int, int],
    expected_bands: int,
    within_target: bool,
) -> None:
    dataset, _, _, _ = adapt_inputs(tmp_path / "work", **make_inputs(tmp_path / "input"))
    config = preparation_config(tmp_path / "run", dataset)
    config = replace(
        config,
        retrieval=replace(
            config.retrieval,
            num_hashes=3,
            lsh_grid=((3, 1), (2, 1), (1, 1)),
            pilot_target_min=target[0],
            pilot_target_max=target[1],
            pilot_target_center=target[2],
        ),
    )
    signatures = np.arange(24 * 3, dtype=np.uint32).reshape(24, 3)
    for column in range(3):
        signatures[1 + column * 4 : 5 + column * 4, column] = signatures[0, column]
    path = tmp_path / "signatures.u32"
    signatures.tofile(path)
    selected, trials = choose_lsh_configuration(
        path, config=config, pilot_anchor_ids=[0], predicted_group_ids=np.full(24, -1)
    )
    assert selected == (expected_bands, 1)
    chosen = [trial for trial in trials if trial["selected"]]
    assert len(chosen) == 1
    assert chosen[0]["within_target"] is within_target
    assert chosen[0]["median_cross_group_candidates"] == expected_bands * 4
    assert ("LEXICAL_PILOT_OUTSIDE_TARGET" in capsys.readouterr().out) is not within_target


@pytest.mark.parametrize("empty", ["anchors", "grid"])
def test_empty_pilot_cannot_select_a_fallback(tmp_path: Path, empty: str) -> None:
    dataset, _, _, _ = adapt_inputs(tmp_path / "work", **make_inputs(tmp_path / "input"))
    config = preparation_config(tmp_path / "run", dataset)
    config = replace(
        config, retrieval=replace(config.retrieval, num_hashes=1, lsh_grid=() if empty == "grid" else ((1, 1),))
    )
    path = tmp_path / "signatures.u32"
    np.ones((24, 1), dtype=np.uint32).tofile(path)
    with pytest.raises(DedupEvaluationError, match="LEXICAL_PILOT_EMPTY"):
        choose_lsh_configuration(
            path,
            config=config,
            pilot_anchor_ids=[] if empty == "anchors" else [0],
            predicted_group_ids=np.full(24, -1),
        )
