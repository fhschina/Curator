# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from eval.dedup.analysis import critic_scope_execution as subject
from eval.dedup.validation import sha256_json


@pytest.mark.parametrize("kind", ["bad_schema", "bad_json", "truncated", "http_error"])
def test_actual_http_failures_always_preserve_receipt_and_available_raw_response(tmp_path, kind):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            raw = {
                "choices": [
                    {
                        "finish_reason": "length" if kind == "truncated" else "stop",
                        "message": {"content": "not json" if kind == "bad_json" else "{}"},
                    }
                ]
            }
            content = json.dumps(raw).encode()
            self.send_response(503 if kind == "http_error" else 200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = {"messages": [], "model": "local-test"}
    request = {
        "canonical_pair_id": "p",
        "repeat": 1,
        "arm": "control",
        "body": body,
        "request_sha256": sha256_json(body),
    }
    path = tmp_path / "receipt.json"
    try:
        result = subject.collect(request, {}, endpoint=f"http://127.0.0.1:{server.server_port}", output=path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result == json.loads(path.read_text())
    assert result["status"] != "VALID"
    if kind != "http_error":
        assert "raw_response" in result
        assert (path.parent / "raw_received" / path.name).exists()
    assert (path.parent / "request_started" / path.name).exists()
    if kind == "bad_schema":
        assert result["error_code"] == "CRITIC_SCOPE_CONTROL_SCHEMA"
    if kind == "truncated":
        assert result["error_code"] == "CRITIC_SCOPE_FINISH"
    assert "public" not in result


@pytest.mark.skipif(not (subject.PRIOR / "manifest.json").exists(), reason="local immutable pilot unavailable")
def test_execution_freeze_keeps_all_original_requests_without_overwriting_base_manifest(tmp_path):
    root = tmp_path / "execution"
    manifest = subject.prepare(root)
    subject.experiment.reference.verify_freeze(root / "manifest.json")
    subject.experiment.reference.verify_freeze(root / "base_freeze/manifest.json")
    assert manifest["execution_contract"] == subject.EXECUTION_CONTRACT
    assert manifest["semantic_change_from_first_pilot"] is False
    assert subject.sha256_file(root / "requests_frozen.json") == subject.sha256_file(
        subject.PRIOR / "requests_frozen.json"
    )
    assert not (root / "started.json").exists()
