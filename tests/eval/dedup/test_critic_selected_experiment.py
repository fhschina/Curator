# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from eval.dedup.analysis import critic_selected_experiment as subject
from eval.dedup.validation import sha256_json


@pytest.mark.parametrize("kind", ["schema", "json", "truncated", "http", "deadline"])
def test_real_http_failure_receipts_remain_durable_and_deadline_prevents_calls(tmp_path, kind):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            raw = {
                "choices": [
                    {
                        "finish_reason": "length" if kind == "truncated" else "stop",
                        "message": {"content": "not json" if kind == "json" else "{}"},
                    }
                ]
            }
            content = json.dumps(raw).encode()
            self.send_response(503 if kind == "http" else 200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = {"messages": [], "model": "local"}
    request = {
        "body": body,
        "canonical_pair_id": "p",
        "repeat": 1,
        "arm": "control",
        "request_sha256": sha256_json(body),
        "deadline_utc_epoch": time.time() + (0 if kind == "deadline" else 3600),
    }
    path = tmp_path / "receipt.json"
    try:
        result = subject.collect(request, {}, endpoint=f"http://127.0.0.1:{server.server_port}", output=path, specs={})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result == json.loads(path.read_text())
    assert result["status"] != "VALID"
    assert "public" not in result
    assert len(received) == (0 if kind == "deadline" else 1)
    assert (tmp_path / "request_started/receipt.json").exists()
    if kind not in ("http", "deadline"):
        assert (tmp_path / "raw_received/receipt.json").exists()
        assert "assistant_content" in result


@pytest.mark.skipif(not (subject.base.PRIOR / "manifest.json").exists(), reason="local frozen evidence unavailable")
def test_real_http_selected_success_binds_original_raw_text_and_public_evidence(tmp_path):
    rows = json.loads((subject.base.PRIOR / "panel_private.json").read_text())
    row = next(r for r in rows if r["review_id"] == "H0108")
    value = {
        "a_loss_span_id": "",
        "b_loss_span_id": "B001",
        "a_context_span_id": "S003",
        "b_context_span_id": "B001",
        "conflict": "NONE",
        "overlap_basis": "RETAINED_CONTENT",
        "shared_anchor_ids": [],
        "explanation": "Both preserve the same full policy; only B contains the specific product.",
    }
    content = json.dumps(value)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = {"messages": [], "model": "local"}
    request = {
        "body": body,
        "canonical_pair_id": row["canonical_pair_id"],
        "repeat": 1,
        "arm": "candidate",
        "request_sha256": sha256_json(body),
        "deadline_utc_epoch": time.time() + 3600,
    }
    try:
        result = subject.collect(
            request,
            row,
            endpoint=f"http://127.0.0.1:{server.server_port}",
            output=tmp_path / "receipt.json",
            specs={"candidate": {"adapter": subject.DEFAULT_ADAPTER}},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result["status"] == "VALID"
    assert result["assistant_content"] == content == result["raw_response"]["choices"][0]["message"]["content"]
    assert result["public"] == row["main_public"]


@pytest.mark.skipif(not (subject.base.PRIOR / "manifest.json").exists(), reason="local frozen evidence unavailable")
def test_real_prepare_binds_reviewed_predecessor_and_keeps_all_three_arms_blind(tmp_path):
    root = tmp_path / "selected"
    manifest = subject.prepare(
        root,
        stage="pilot",
        parent=None,
        adapter=subject.DEFAULT_ADAPTER,
        config=subject.DEFAULT_CONFIG,
        protocol=subject.DEFAULT_PROTOCOL,
        control_adapter="eval.dedup.judging.critic_retention_v2",
        control_config=subject.base.previous.RESOURCES / "v06212_retention_v2.yaml",
    )
    subject.base.previous.reference.verify_freeze(root / "manifest.json")
    subject.base.previous.reference.verify_freeze(root / "input_freeze/manifest.json")
    assert manifest["logical_requests"] == 96
    assert manifest["population"] == 19
    assert manifest["main_online_calls"] == 0
    assert any(p.endswith("retention-v2-pilot/review_complete.json") for p in manifest["sources"])
    rows = json.loads((root / "panel_private.json").read_text())
    by_id = {r["canonical_pair_id"]: r for r in rows}
    requests = json.loads((root / "requests_frozen.json").read_text())
    for request in requests:
        row = by_id[request["canonical_pair_id"]]
        text = json.dumps(request["body"]["messages"])
        assert row["review_id"] not in text
        assert row["canonical_pair_id"] not in text
        assert "human_same_duplicate_group" not in text
        assert request["request_sha256"] == sha256_json(request["body"])
    assessment = subject.assess(root)
    assert len(assessment["cells"]) == 6
    assert assessment["full_development_75_gate"] is False
    for cell in assessment["cells"].values():
        assert cell["scores"]["partial_draft"]["unweighted"]["rows"] == 19
        assert cell["engineering_failures"] == 16
