# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

from eval.dedup.analysis import exp6_coverage_format_fix as candidate
from eval.dedup.analysis import v07 as subject
from tests.eval.dedup import test_exp6_coverage_format_fix as coverage_tests
from tests.eval.dedup.test_exp1_conflict_evidence_fix import conflict_fixture, run_case


def test_release_candidate_preserves_selected_runtime_except_identity(tmp_path):
    payload, main, _, coverage = conflict_fixture()
    old, old_calls = run_case(
        candidate, tmp_path / "old", deepcopy(payload), [main, coverage_tests.json.dumps(coverage)]
    )
    new, new_calls = run_case(
        subject, tmp_path / "new", deepcopy(payload), [main, coverage_tests.json.dumps(coverage)]
    )

    assert subject.SOURCE_VERSION == candidate.VERSION
    assert old_calls == new_calls
    assert {key: value for key, value in old.items() if key != "version"} == {
        key: value for key, value in new.items() if key != "version"
    }
    assert old["version"] == "v0.6.2.33-exp6+coverage-format-fix1"
    assert new["version"] == "v0.7"
