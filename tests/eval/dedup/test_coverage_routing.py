# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import pytest

from eval.dedup.judging.coverage_routing import route_coverage
from eval.dedup.judging.coverage_witness import adapt_coverage_witness
from eval.dedup.validation import DedupEvaluationError
from tests.eval.dedup.test_coverage_witness import _case, _uncover


@pytest.mark.parametrize("basis", ["none", "unresolved"])
def test_owned_main_result_is_identical_for_valid_covered_or_uncovered_critic(basis):
    main, coverage, payload = _case()
    main["span_shared_basis"]["score"] = basis
    snapshot = deepcopy((main, payload))
    result = route_coverage(main, payload)
    assert result.route == ("MAIN_NO" if basis == "none" else "MAIN_UNRESOLVED")
    assert result.public_output == adapt_coverage_witness(main, coverage, payload)
    _uncover(coverage, payload)
    assert result.public_output == adapt_coverage_witness(main, coverage, payload)
    assert (main, payload) == snapshot


def test_complete_nonempty_exact_returns_the_same_valid_public_result_without_critic():
    main, coverage, payload = _case("Identical text.", "Identical text.")
    coverage["record_scope"] = "IDENTICAL_TEXT"
    result = route_coverage(main, payload)
    assert result.route == "VISIBLE_EXACT"
    assert result.public_output == adapt_coverage_witness(main, coverage, payload)


def test_incomplete_input_is_low_confidence_unknown_without_guessing_main_or_critic():
    result = route_coverage(
        {},
        {
            "semantic_diff_evidence": {"status": "INCOMPLETE"},
            "long_document_evidence": {"truncated": True, "windows": []},
        },
    )
    assert result.route == "INCOMPLETE_INPUT"
    assert result.public_output["same_duplicate_group"] == "UNRESOLVED"
    assert result.public_output["confidence_tier"] == "LOW"


@pytest.mark.parametrize("non_main", [False, True])
def test_every_nonexact_positive_still_requires_fresh_coverage_even_if_it_looks_benign(non_main):
    main, _, payload = _case("Shared policy.", "Shared policy. Settings", non_main=non_main)
    result = route_coverage(main, payload)
    assert result.route == "NEEDS_COVERAGE"
    assert result.public_output is None


def test_complete_main_unknown_is_not_invented_when_main_response_is_missing():
    _, _, payload = _case()
    with pytest.raises((DedupEvaluationError, KeyError)):
        route_coverage({}, payload)
