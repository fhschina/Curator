from copy import deepcopy

from eval.dedup.judging import coverage_format_recovery as subject


def test_tab_loop_recovery_is_failure_only_and_non_mutating() -> None:
    response = {"choices": [{"finish_reason": "length", "message": {"content": "x" + "\t" * 5000}}]}
    finding = subject.detect(response)
    assert finding is not None
    assert finding["tail_tab_fraction"] == 1.0

    request = {"messages": [{"role": "user", "content": "original"}]}
    original = deepcopy(request)
    repaired = subject.repair_request(request)
    assert request == original
    assert repaired["messages"][:-1] == original["messages"]
    assert subject.detect({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}) is None
