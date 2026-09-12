# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from eval.dedup.analysis import critic_full_diagnostic as subject
from eval.dedup.validation import DedupEvaluationError

ROOT = subject.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v4-paired96"


@pytest.mark.skipif(not (ROOT / "review_complete.json").exists(), reason="local completed failed trial unavailable")
def test_real_repeated_guard_regression_cannot_be_expanded(tmp_path):
    with pytest.raises(DedupEvaluationError) as caught:
        subject.prepare(tmp_path / "full", ROOT, ROOT / "manifest.json")
    assert caught.value.issue.code == "FULL_CRITIC_PROTECTION"
    assert not (tmp_path / "full").exists()


@pytest.mark.skipif(
    not subject.selected.base.previous.REFERENCE_ROOT.exists(), reason="local original development unavailable"
)
def test_full_original_replay_retains_every_label_weight_and_bypass():
    rows, sources = subject.original_rows()
    assert len(rows) == len({r["canonical_pair_id"] for r in rows}) == 1000
    assert len({r["review_id"] for r in rows}) == 1000
    assert sources
    from collections import Counter

    routes = Counter(subject.selected.base.critic.veto.route(r["main_public"], r["payload"]) for r in rows)
    assert routes == {
        "REVIEW_POSITIVE_DIRECTIONS_ONLY": 250,
        "PRESERVE_MAIN_NEGATIVE_OR_UNRESOLVED": 738,
        "PRESERVE_COMPLETE_EXACT_INPUT": 12,
    }
    weight = subject.selected.base.previous._weight
    assert sum(weight(r["draft_label"]) for r in rows) == pytest.approx(10023)
    assert all(weight(r["draft_label"]) == weight(r["historical_label"]) for r in rows)
