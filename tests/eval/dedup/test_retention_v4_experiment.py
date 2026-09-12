# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from eval.dedup.analysis import retention_v4_experiment as subject
from eval.dedup.judging import retention_v4 as adapter
from eval.dedup.validation import DedupEvaluationError, sha256_json
from tests.eval.dedup.test_retention_v4 import payload, review


def test_old_nonexact_raw_is_not_promoted_to_a_new_retention_proof_or_hybrid_score():
    rows = [
        {
            "canonical_pair_id": "one",
            "review_id": "ONE",
            "raw_main": {"old": "saved"},
            "payload": payload("Policy. Product.", "Policy."),
        }
    ]
    before = deepcopy(rows)
    result = subject.offline_replay(rows)
    assert result[0]["status"] == "UNAVAILABLE_MISSING_RETENTION_PROOF"
    assert result[0]["candidate_primary"] is None
    assert not result[0]["raw_output_reinterpreted"]
    assert rows == before
    rows[0]["payload"] = payload("Policy.", "Policy.")
    assert subject.offline_replay(rows)[0]["candidate_primary"]["same_duplicate_group"] == "YES"


def test_baseline_uses_native_historical_score_recipe_and_same_generation_contract():
    render, schema = subject.baseline_renderer()
    p = payload("Same policy.\nProduct", "Same policy.")
    messages = render(p)
    assert "V0.6.2.12" in messages[0]["content"]
    assert "span_content_profile_a" in schema["properties"]
    body = subject.request_body(messages, schema)
    other = subject.request_body(subject.runtime.messages(p), adapter.response_schema())
    assert all(body[k] == other[k] for k in subject.GENERATION)
    assert body["model"] == other["model"]
    assert body["response_format"]["type"] == other["response_format"]["type"] == "json_schema"


def test_native_response_transport_persists_request_and_raw_then_refuses_reuse(tmp_path):
    parsed = {"answer": "example"}
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(parsed)}}]}
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    body = {"messages": [{"role": "user", "content": "Test input"}]}
    try:
        result = subject.collect(tmp_path, "one", body, endpoint)
        assert result["status"] == "RECEIVED"
        assert result["parsed"] == parsed
        assert result["request_sha256"] == sha256_json(body)
        assert bodies == [body]
        assert (
            json.loads((tmp_path / "responses/one.json").read_text())["raw_response"]["choices"][0]["finish_reason"]
            == "stop"
        )
        with pytest.raises(DedupEvaluationError, match="RETENTION_PILOT_REUSE"):
            subject.collect(tmp_path, "one", body, endpoint)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def assessment_fixture():
    p = payload("Policy.\nNext", "Policy.")
    public = adapter.adapt_main(review(p, "NON_MAIN_ADDITION"), p)
    rows = [
        {
            "canonical_pair_id": "one",
            "review_id": "ONE",
            "payload": p,
            "gate_expected": subject.preflight.primary(public),
        }
    ]
    results = [
        {
            "canonical_pair_id": "one",
            "review_id": "ONE",
            "arm": arm,
            "repeat": rep,
            "status": "VALID",
            "stages": [],
            "public": deepcopy(public),
        }
        for rep in (1, 2)
        for arm in subject.ARMS
    ]
    return rows, results


def test_pilot_assessment_requires_all_cases_and_repeats_and_never_auto_admits_full1000():
    rows, results = assessment_fixture()
    out = subject.assessment(rows, results)
    assert out["predeclared_candidate_checks_passed"]
    assert not out["full1000_admitted"]
    with pytest.raises(DedupEvaluationError, match="RETENTION_PILOT_JOIN"):
        subject.assessment(rows, results[:-1])
    with pytest.raises(DedupEvaluationError, match="RETENTION_PILOT_JOIN"):
        subject.assessment(rows, [*results, results[0]])


def test_repeat_change_or_engineering_failure_blocks_pilot_even_if_other_cases_pass():
    rows, results = assessment_fixture()
    results[-1]["status"] = "ENGINEERING_FAILURE"
    results[-1]["public"] = subject.unresolved_judge_output_v4()
    out = subject.assessment(rows, results)
    assert not out["predeclared_candidate_checks_passed"]
    assert out["repeat_primary_changes"]["candidate"] == ["ONE"]
    assert out["cells"]["2/candidate"]["population"] == 1


def test_exact_input_orientation_comparison_is_independent_of_reference_labels():
    rows, results = assessment_fixture()
    other = deepcopy(rows[0])
    other.update(canonical_pair_id="two", review_id="TWO", gate_expected=None)
    rows.append(other)
    extra = []
    for item in results:
        copy = deepcopy(item)
        copy.update(canonical_pair_id="two", review_id="TWO")
        if copy["arm"] == "candidate":
            copy["public"] = adapter.adapt_main(review(other["payload"]), other["payload"])
        extra.append(copy)
    out = subject.assessment(rows, results + extra)
    assert out["candidate_exact_input_disagreements"]["1"] == [["ONE", "TWO"]]
    assert not out["predeclared_candidate_checks_passed"]
