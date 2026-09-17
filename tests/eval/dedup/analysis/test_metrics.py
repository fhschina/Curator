import pytest

from eval.dedup.analysis.metrics import wilson_interval


def test_wilson_interval_handles_empty_and_bounded_samples() -> None:
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4901624715)
    assert high == pytest.approx(0.9433178485)
