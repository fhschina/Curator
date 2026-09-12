# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from eval.dedup.analysis import critic_retention_experiment as subject
from eval.dedup.validation import DedupEvaluationError, sha256_json


@pytest.mark.parametrize("kind", ["schema", "json", "truncated", "http", "deadline"])
def test_real_http_failures_and_deadline_preserve_receipts_without_fake_valid_outputs(tmp_path, kind):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            raw = {
                "choices": [
                    {
                        "finish_reason": "length" if kind == "truncated" else "stop",
                        "message": {"content": "not json" if kind == "json" else "{}"},
                    }
                ]
            }
            body = json.dumps(raw).encode()
            self.send_response(503 if kind == "http" else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = {"model": "local", "messages": []}
    request = {
        "canonical_pair_id": "p",
        "repeat": 1,
        "arm": "control",
        "body": body,
        "request_sha256": sha256_json(body),
        "deadline_utc_epoch": time.time() + (0 if kind == "deadline" else 3600),
    }
    output = tmp_path / "receipt.json"
    try:
        result = subject.collect(request, {}, endpoint=f"http://127.0.0.1:{server.server_port}", output=output)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result == json.loads(output.read_text())
    assert "public" not in result
    assert len(received) == (0 if kind == "deadline" else 1)
    assert (tmp_path / "request_started/receipt.json").exists()
    if kind not in ("deadline", "http"):
        assert (tmp_path / "raw_received/receipt.json").exists()
    if kind == "schema":
        assert result["error_code"] == "CRITIC_SCOPE_CONTROL_SCHEMA"
    if kind == "deadline":
        assert result["status"] == "DEADLINE_NOT_SUBMITTED"


@pytest.mark.skipif(not (subject.PRIOR / "manifest.json").exists(), reason="local frozen pilot unavailable")
def test_real_prepare_is_blind_preserves_all_denominators_and_separates_format_arm(tmp_path):
    root = tmp_path / "pilot"
    manifest = subject.prepare(root)
    subject.previous.reference.verify_freeze(root / "manifest.json")
    rows = json.loads((root / "panel_private.json").read_text())
    requests = json.loads((root / "requests_frozen.json").read_text())
    assert len(rows) == 19
    assert len(requests) == 96
    assert manifest["reference_changed"] is False
    assert manifest["main_online_calls"] == 0
    by_id = {r["canonical_pair_id"]: r for r in rows}
    for request in requests:
        row = by_id[request["canonical_pair_id"]]
        text = json.dumps(request["body"]["messages"])
        assert row["review_id"] not in text
        assert row["canonical_pair_id"] not in text
        assert "human_same_duplicate_group" not in text
        assert request["request_sha256"] == sha256_json(request["body"])
        assert request["deadline_utc_epoch"] == subject.DEADLINE
    assessment = subject.assess(root)
    assert assessment["next_step"] == "STOP_REVIEW_AND_REPAIR"
    assert assessment["full_development_75_gate"] is False
    assert len(assessment["cells"]) == 6
    for cell in assessment["cells"].values():
        assert cell["scores"]["partial_draft"]["unweighted"]["rows"] == 19
        assert cell["engineering_failures"] == 16
    with pytest.raises(DedupEvaluationError, match="RETENTION_ROOT_EXISTS"):
        subject.prepare(root)


def test_expansion_requires_a_reviewed_passing_predecessor(tmp_path):
    with pytest.raises(DedupEvaluationError, match="RETENTION_PARENT"):
        subject.prepare(tmp_path / "unapproved", stage="full")
