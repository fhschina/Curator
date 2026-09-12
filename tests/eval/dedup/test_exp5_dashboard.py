# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from eval.dedup.analysis import exp5_dashboard as subject
from eval.dedup.validation import DedupEvaluationError, sha256_file, write_json_atomic


def test_baseline_resolves_published_sarah_qwen_instead_of_source_v0(tmp_path):
    artifact = tmp_path / "comparison.json"
    write_json_atomic(artifact, {"version": "V0.5"})
    release = tmp_path / "release_manifest.json"
    write_json_atomic(
        release,
        {
            "version": "V0.5",
            "version_definition": {"framework": "Sarah MinHash judging framework"},
            "result_run_id": "sarah-qwen",
            "result_run_root": str(tmp_path / "sarah-qwen"),
            "source_v0_run_root": str(tmp_path / "v0-deepseek"),
            "artifacts": {"comparison": {"path": str(artifact), "sha256": sha256_file(artifact)}},
        },
    )
    assert subject.baseline_root(release) == tmp_path / "sarah-qwen"


def test_mislabeled_v0_cannot_be_used_as_published_v05(tmp_path):
    release = tmp_path / "release_manifest.json"
    write_json_atomic(release, {"version": "V0", "version_definition": {"framework": "DeepSeek"}})
    with pytest.raises(DedupEvaluationError, match="EXP5_V05_RELEASE"):
        subject.baseline_root(release)


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
