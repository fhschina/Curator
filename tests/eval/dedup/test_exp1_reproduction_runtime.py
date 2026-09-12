# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from eval.dedup.analysis import checkpoint_preflight as fixtures
from eval.dedup.analysis import exp1_reproduction_runtime as subject
from eval.dedup.validation import DedupEvaluationError, sha256_json
from tests.eval.dedup.test_critic_retention_v3 import fixture as coverage_fixture
from tests.eval.dedup.test_critic_subject_binding import fixture as subject_fixture


def response(value):
    content = value if isinstance(value, str) else "```json\n" + json.dumps(value) + "\n```"
    return {
        "id": "offline",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@contextmanager
def server(answers):
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps(response(answers[min(len(bodies) - 1, len(answers) - 1)])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    listener = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=listener.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.server_port}", bodies
    finally:
        listener.shutdown()
        listener.server_close()
        thread.join()


def test_original_missing_feedback_rendering_and_native_correction_are_preserved():
    p = fixtures.synthetic_payload("Policy. Extra.", "Policy.")
    messages = subject.main_messages(p)
    assert "this safe structured issue: nan" in messages[1]["content"][0]["text"]
    raw = fixtures.synthetic_raw(p)
    observed = []

    def call(body):
        observed.append(deepcopy(body))
        return response("invalid json" if len(observed) == 1 else raw)

    assert subject.generate_main(messages, call) == raw
    assert observed[0] == subject.body(messages)
    assert "response_format" not in observed[0]
    assert len(observed) == 2
    assert [m["role"] for m in observed[1]["messages"]] == ["system", "user", "assistant", "user"]
    assert "invalid json" in json.dumps(observed[1]["messages"])


def test_complete_fresh_chain_preserves_payload_specific_subject_schema_and_verified_veto(tmp_path):
    p, _, proposal = subject_fixture()
    p["payload_schema_version"] = "judge-visible-payload-v3"
    raw = fixtures.synthetic_raw(
        p,
        profiles=("non_main_only", "non_main_only"),
        deltas=("universal_ui_or_repetition", "universal_ui_or_repetition"),
        basis="verified_equivalent_non_main_message",
    )
    _, _, coverage = coverage_fixture(p["document_a"]["text"], p["document_b"]["text"])
    verifier = {
        "a_subject_kind": "NAMED_ACTUAL_TARGET",
        "b_subject_kind": "NAMED_ACTUAL_TARGET",
        "comparison": "SUPPORTED_DIFFERENT_NAMED_TARGETS",
        "explanation": "Distinct named liability parties.",
    }
    row = {"canonical_pair_id": "pair", "review_id": "R1", "payload": p}
    original = deepcopy(row)
    body = subject.body(subject.main_messages(p))
    frozen = {"body": body, "request_sha256": sha256_json(body)}
    with server([raw, json.dumps(coverage), json.dumps(proposal), json.dumps(verifier)]) as (endpoint, bodies):
        result = subject.execute_case(tmp_path, row, endpoint, frozen, subject.coverage_renderer())
    assert result["status"] == "VALID", result.get("error_code")
    assert result["components"]["main"]["same_duplicate_group"] == "YES"
    assert result["public"]["same_duplicate_group"] == "NO"
    assert [s["stage"] for s in result["stages"]] == ["main-01-01", "coverage", "subject", "verifier"]
    assert bodies[0] == body
    schema = bodies[2]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["a_subject_span_id"]["enum"] == ["", "A001"]
    assert "S002" not in schema["properties"]["a_subject_span_id"]["enum"]
    assert row == original
    assert len(list((tmp_path / "responses").glob("*.json"))) == 4


def test_exact_input_still_gets_a_fresh_main_and_does_not_get_new_input_bypass(tmp_path):
    p = fixtures.synthetic_payload("Same policy.", "Same policy.")
    p["payload_schema_version"] = "judge-visible-payload-v3"
    raw = fixtures.synthetic_raw(p, deltas=("none", "none"))
    body = subject.body(subject.main_messages(p))
    with server([raw]) as (endpoint, bodies):
        result = subject.execute_case(
            tmp_path,
            {"canonical_pair_id": "pair", "review_id": "R1", "payload": p},
            endpoint,
            {"body": body, "request_sha256": sha256_json(body)},
            lambda _: pytest.fail("exact cannot route to coverage"),
        )
    assert result["status"] == "VALID"
    assert len(bodies) == 1
    assert result["public"]["same_duplicate_group"] == "YES"


def test_native_main_parser_corrections_remain_bounded():
    from data_designer.engine.models.errors import ModelGenerationValidationFailureError

    p = fixtures.synthetic_payload("A", "B")
    calls = []

    def invalid(body):
        calls.append(body)
        return response("invalid json")

    with pytest.raises(ModelGenerationValidationFailureError, match="could not be parsed"):
        subject.generate_main(subject.main_messages(p), invalid)
    assert len(calls) == 3


def test_existing_receipt_cannot_be_reused(tmp_path):
    p = fixtures.synthetic_payload("A", "B")
    body = subject.body(subject.main_messages(p))
    with server([fixtures.synthetic_raw(p)]) as (endpoint, bodies):
        assert subject.collect(tmp_path, "pair", body, endpoint)["status"] == "RECEIVED"
        with pytest.raises(DedupEvaluationError, match="EXP1_REUSE"):
            subject.collect(tmp_path, "pair", body, endpoint)
    assert len(bodies) == 1
