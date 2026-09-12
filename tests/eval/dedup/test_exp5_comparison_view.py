# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

from eval.dedup.analysis import exp5_comparison_view as subject


def fixture():
    return {
        "collected": 2,
        "population": 20000,
        "complete": False,
        "valid": 1,
        "engineering_failures": 1,
        "semantic_unresolved": 0,
        "primary_changed_common_valid": 1,
        "exp5_groups": {"NO": 1},
        "v05_groups_same_collected_pairs": {"YES": 2},
        "agreement": {"common_valid": 1, "same_duplicate_group_matrix": {"YES": {"NO": 1}}},
        "development_calibration": {"status": "PENDING_COMPLETE_DEVELOPMENT_COHORT", "available": 1, "required": 1000},
    }


def test_partial_overview_has_no_invented_calibration_and_escapes_pair_data():
    summary = fixture()
    output = subject.overview_html(summary, [{"canonical_pair_id": "</script><script>alert(1)</script>"}])
    assert "运行中 · 部分结果快照" in output
    assert "等待完整开发集：1 / 1000" in output
    assert "</script><script>alert(1)</script>" not in output
    assert 'id="change"' in output
    assert "v05_exp5_pairs.csv" in output
    assert "proxy" in output


def test_final_calibration_keeps_weighted_unweighted_and_undefined_rates_distinct():
    summary = fixture()
    block = {"duplicate_precision": None, "duplicate_recall": 0.8, "primary_decision_exact": 0.9}
    summary["development_calibration"] = {
        "status": "AVAILABLE",
        "v05": {"weighted": block, "unweighted": block},
        "exp5": {"weighted": {**block, "duplicate_precision": 0.85}, "unweighted": block},
        "exp5_empirical_confidence_tiers": {},
    }
    rendered = subject.calibration_table(summary)
    assert "85.00%" in rendered
    assert "—" in rendered
    assert "非加权" in rendered
    report = subject.results_markdown(summary)
    assert "| 加权 | Exp5 | 85.00% | 80.00% | 90.00% |" in report
    assert "不是独立人工金标" in report
