# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from eval.dedup.analysis import retention_context_experiment as subject
from eval.dedup.validation import DedupEvaluationError, sha256_json
from tests.eval.dedup.test_retention_context_v1 import payload, review
from tests.eval.dedup.test_retention_v4_experiment import assessment_fixture


def test_actual_paired_runner_uses_new_main_and_critic_contract_and_fresh_bound_receipts(tmp_path):
    p = payload("This claim is false: The service is free.", "The service is free.")
    answers = [review(p), review(p, conflict="NEGATION_CONFLICT")]
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            value = {
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answers[len(bodies) - 1])}}]
            }
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = subject.request_body(subject.runtime.messages(p), subject.candidate.response_schema())
    row = {"canonical_pair_id": "test", "review_id": "local", "payload": p}
    original = deepcopy(row)
    subject.warmup()
    try:
        result = subject.execute_case(
            tmp_path,
            row,
            "candidate",
            1,
            f"http://127.0.0.1:{server.server_port}",
            {"body": body, "request_sha256": sha256_json(body)},
            lambda _: pytest.fail("baseline renderer used"),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result["status"] == "VALID"
    assert result["components"]["main"]["same_duplicate_group"] == "YES"
    assert result["public"]["same_duplicate_group"] == "NO"
    assert [s["stage"] for s in result["stages"]] == ["main", "coverage"]
    assert len(bodies) == 2
    assert bodies[0] == body
    assert bodies[1]["messages"] == subject.runtime.messages(p, stage="critic")
    assert len(list((tmp_path / "responses").glob("*.json"))) == 2
    assert row == original


def test_same_gates_reject_instability_and_account_for_all_case_repeat_cells():
    rows, results = assessment_fixture()
    out = subject.assessment(rows, results)
    assert out["version"] == subject.runtime.VERSION
    assert out["predeclared_candidate_checks_passed"]
    assert not out["full1000_admitted"]
    results[-1]["status"] = "ENGINEERING_FAILURE"
    assert not subject.assessment(rows, results)["predeclared_candidate_checks_passed"]
    with pytest.raises(DedupEvaluationError):
        subject.assessment(rows, results[:-1])


def test_run_requires_bound_freeze_before_any_credentials_or_online_calls(tmp_path):
    with pytest.raises((FileNotFoundError, DedupEvaluationError)):
        subject.run(tmp_path, tmp_path / "never_read.env")


def test_prepare_rejects_reusing_existing_root_even_before_reading_predecessor(tmp_path):
    with pytest.raises(DedupEvaluationError, match="RETENTION_CONTEXT_ROOT"):
        subject.prepare(tmp_path, tmp_path / "missing")
