# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from eval.dedup.analysis import exp5_dashboard as subject


def test_comparison_does_not_score_failures_or_missing_baseline_as_agreement():
    yes = dict.fromkeys(subject.PRIMARY, "YES")
    no = dict.fromkeys(subject.PRIMARY, "NO")
    rows = subject.comparison_rows(
        [
            {"canonical_pair_id": "same", "status": "VALID", "public": yes},
            {"canonical_pair_id": "changed", "status": "VALID", "public": no},
            {"canonical_pair_id": "failed", "status": "ENGINEERING_FAILURE", "public": no},
            {"canonical_pair_id": "missing", "status": "VALID", "public": yes},
        ],
        {"same": yes, "changed": yes, "failed": no},
    )
    assert [r["primary_changed"] for r in rows] == [False, True, None, None]
    assert rows[2]["exp5_same_duplicate_group"] == "ENGINEERING_FAILURE"
    assert rows[3]["v05_status"] == "NO_VALID_OUTPUT"


def test_native_explorer_retained_and_partial_scope_explicit():
    summary = {"complete": False, "collected": 100, "population": 20000, "pending": 19900, "engineering_failures": 1}
    output = subject.explorer_html([], {}, summary)
    assert "Dedup Pair Explorer" in output
    assert "PARTIAL SNAPSHOT" in output
    assert "19,900 pending" in output
    assert "baselineComparison(r)" in output
    assert "UNAVAILABLE_MISSING_CONTRACT" in output
    assert 'id="language"' in output
    assert 'id="reason-grid"' in output
    assert "Version agreement is not accuracy" in output or "version agreement is not accuracy" in output
