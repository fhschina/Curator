from copy import deepcopy

from eval.dedup.judging import repetition_recovery as subject


def _response(content: str, finish_reason: str = "length") -> dict:
    return {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}


def test_detects_only_repetitive_incomplete_main_output() -> None:
    finding = subject.detect(_response("identifier A001 " * 5000))

    assert finding is not None
    assert finding["contract_version"] == "dedup-main-truncated-repetition-recovery-v1"
    assert subject.detect(_response("{}", "stop")) is None
    assert subject.REPAIR_CONTRACT == "dedup-main-truncated-repetition-recovery-v2"


def test_repair_request_preserves_original_and_requires_citations() -> None:
    request = {"model": "judge", "messages": [{"role": "user", "content": "original"}]}
    original = deepcopy(request)

    repaired = subject.repair_request(request)

    assert request == original
    assert repaired["messages"][:-1] == original["messages"]
    assert "span IDs are not optional" in repaired["messages"][-1]["content"][0]["text"]
