# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time
from copy import deepcopy

import pytest
from werkzeug.wrappers import Response

from eval.dedup.analysis import critic_ordered_experiment as subject
from eval.dedup.validation import DedupEvaluationError, write_json_atomic

ROOT = subject.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v4-paired96"


def frozen_case():
    rows = json.loads((ROOT / "panel_private.json").read_text())
    row = next(r for r in rows if r["review_id"] == "H0232")
    request = next(
        r
        for r in json.loads((ROOT / "requests_frozen.json").read_text())
        if r["canonical_pair_id"] == row["canonical_pair_id"] and r["arm"] == "candidate" and r["repeat"] == 1
    )
    receipt = json.loads((ROOT / "responses" / f"1-candidate-{row['canonical_pair_id']}.json").read_text())
    return row, request, receipt


@pytest.mark.skipif(not (ROOT / "complete.json").exists(), reason="local frozen real proof unavailable")
def test_freeze_preserves_order_despite_canonical_artifact_sorting_and_rejects_tampering(tmp_path):
    _, original, _ = frozen_case()
    order = list(original["body"]["response_format"]["json_schema"]["schema"]["properties"])
    order.remove("explanation")
    order.insert(0, "explanation")
    request = subject.with_property_order(original, order)
    write_json_atomic(tmp_path / "request.json", request)
    loaded = json.loads((tmp_path / "request.json").read_text())
    assert next(iter(loaded["body"]["response_format"]["json_schema"]["schema"]["properties"])) != "explanation"
    actual = subject.thaw(loaded)
    assert actual["body"] == original["body"]
    assert list(actual["body"]["response_format"]["json_schema"]["schema"]["properties"]) == order
    corrupted = deepcopy(loaded)
    corrupted["ordered_body_json"] += " "
    with pytest.raises(DedupEvaluationError) as caught:
        subject.thaw(corrupted)
    assert caught.value.issue.code == "CRITIC_ORDER_BINDING"
    with pytest.raises(DedupEvaluationError):
        subject.with_property_order(original, ["explanation"] * 8)


@pytest.mark.skipif(not (ROOT / "complete.json").exists(), reason="local frozen real proof unavailable")
def test_actual_client_and_paced_relay_preserve_nested_order_and_local_proof(httpserver, tmp_path):
    row, original, saved = frozen_case()
    order = list(original["body"]["response_format"]["json_schema"]["schema"]["properties"])
    order.remove("explanation")
    order.insert(0, "explanation")
    request = subject.with_property_order(original, order)
    request["deadline_utc_epoch"] = time.time() + 3600
    write_json_atomic(tmp_path / "request.json", request)
    observed = []

    def respond(received):
        observed.append(json.loads(received.get_data()))
        return Response(json.dumps(saved["raw_response"]), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions").respond_with_handler(respond)
    previous = subject.selected.base.previous
    with previous.PacedRelay(
        profile=previous.TransportProfile(min_interval_seconds=0.01, max_in_flight=1, max_attempts=1),
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_model="local-upstream",
        upstream_base_url=httpserver.url_for("/v1"),
        upstream_api_key="local-test-secret",
        timeout_seconds=5,
        expected_generation_parameters=previous.GENERATION,
    ) as relay:
        relay.set_context(previous.RelayContext("order-test", 0, tmp_path / "transport.jsonl"))
        receipt = subject.selected.collect(
            subject.thaw(json.loads((tmp_path / "request.json").read_text())),
            row,
            endpoint=relay.endpoint,
            output=tmp_path / "responses/receipt.json",
            specs={"candidate": {"adapter": "eval.dedup.judging.critic_retention_v4"}},
        )
    assert len(observed) == 1
    assert list(observed[0]["response_format"]["json_schema"]["schema"]["properties"]) == order
    assert observed[0]["messages"] == original["body"]["messages"]
    assert receipt["status"] == "VALID"
    assert receipt["public"] == saved["public"]
    assert "local-test-secret" not in (tmp_path / "transport.jsonl").read_text()


@pytest.mark.skipif(
    not (ROOT.parent / "retention-v5-paired96/review_complete.json").exists(), reason="review unavailable"
)
def test_real_frozen_trial_is_semantically_identical_and_only_candidate_order_changes(tmp_path):
    from pathlib import Path

    root = tmp_path / "order"
    result = subject.prepare(
        root,
        ROOT.parent / "retention-v5-paired96",
        Path(subject.__file__).with_name("critic_ordered_experiment_v1.md"),
    )
    subject.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    assert result["population"] == 96
    assert result["logical_requests"] == 354
    assert result["specs"]["candidate"] == result["specs"]["encoding_control"]
    requests = json.loads((root / "requests_frozen.json").read_text())
    controls = {(r["repeat"], r["canonical_pair_id"]): r for r in requests if r["arm"] == "encoding_control"}
    for request in (r for r in requests if r["arm"] == "candidate"):
        control = controls[(request["repeat"], request["canonical_pair_id"])]
        assert request["body"] == control["body"]
        assert request["request_sha256"] == control["request_sha256"]
        candidate_order = list(subject.thaw(request)["body"]["response_format"]["json_schema"]["schema"]["properties"])
        control_order = list(control["body"]["response_format"]["json_schema"]["schema"]["properties"])
        assert candidate_order == ["explanation", *(k for k in control_order if k != "explanation")]
        assert "ordered_body_json" not in control
