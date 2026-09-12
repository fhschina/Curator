# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.analysis import exp1_reproduction as subject
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_retention_context_full1000 import fixture


def test_identical_scoring_masks_weights_and_history_are_preserved():
    labels, histories, results = fixture()
    before = deepcopy((labels, histories, results))
    out = subject.assess(labels, histories, results, {"3"})
    for score in out["scores"].values():
        assert score["original_rows"] == 4
        assert score["scored_rows"] == 3
        assert score["weighted"]["weight_total"] == 6
        assert score["weighted"]["primary_decision_exact"] == 1
    assert out["ledger"][-1]["errors"][subject.FRESH] == "EXCLUDED_NOT_SCORED"
    assert (labels, histories, results) == before
    assert not out["full20k_admitted"]
    assert not out["release_eligible"]


def test_engineering_failure_cannot_receive_unresolved_match_credit():
    labels, histories, results = fixture()
    labels[0].update({f"human_{k}": "UNRESOLVED" for k in subject.benchmark.primary(results[0]["public"])})
    results[0].update(status="ENGINEERING_FAILURE", public=subject.runtime.unresolved_judge_output_v3())
    out = subject.assess(labels, histories, results, set())
    assert out["scores"][subject.FRESH]["weighted"]["primary_decision_exact"] == pytest.approx(0.9)
    assert out["ledger"][0]["errors"][subject.FRESH] == "ENGINEERING_MISSING_OUTPUT"
    assert subject.projection(results[0])["metric_only_missing_output"]
    assert not subject.projection(results[0], "main").get("metric_only_missing_output")


@pytest.mark.parametrize("damage", ["missing", "duplicate", "missing_history"])
def test_missing_rows_and_histories_fail_closed(damage):
    labels, histories, results = fixture()
    if damage == "missing":
        results.pop()
    elif damage == "duplicate":
        results.append(deepcopy(results[0]))
    else:
        histories.pop(subject.HISTORICAL[0])
    with pytest.raises(DedupEvaluationError):
        subject.assess(labels, histories, results, set())


def test_reproduction_decline_is_not_hidden_by_absolute_thresholds():
    labels, histories, results = fixture()
    results[0]["public"] = subject.runtime.unresolved_judge_output_v3()
    out = subject.assess(labels, histories, results, set())
    assert out["development_numeric_checks"]["weighted_recall_75"]
    assert not out["reproduction_point_checks"]["duplicate_recall"]


def test_existing_root_and_unbound_run_are_rejected_before_credentials(tmp_path):
    with pytest.raises(DedupEvaluationError, match="EXP1_ROOT"):
        subject.prepare(tmp_path)
    with pytest.raises(FileNotFoundError):
        subject.run(tmp_path, tmp_path / "never-read.env")
