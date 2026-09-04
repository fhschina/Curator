# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from eval.dedup.judging.request_relay import RelayContext, RequestRelay


def test_relay_rewrites_model_and_keeps_credentials_and_content_out_of_events(tmp_path) -> None:
    observed: dict[str, Any] = {}

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            observed["authorization"] = self.headers["Authorization"]
            observed["body"] = json.loads(self.rfile.read(length))
            response = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format: str, *args: object) -> None:
            del args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    secret = "super-secret-api-key"  # noqa: S105 - deliberately recognizable fixture value
    relay = RequestRelay(
        logical_model="logical-qwen",
        upstream_base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
        upstream_model="hub-qwen",
        upstream_api_key=secret,
        timeout_seconds=5,
        expected_generation_parameters={"temperature": 0.0, "max_tokens": 16},
    )
    events_path = tmp_path / "events.jsonl"
    try:
        with relay:
            relay.set_context(RelayContext(block_id="pilot", outer_attempt=1, events_path=events_path))
            body = {
                "model": "logical-qwen",
                "messages": [{"role": "user", "content": "private document text"}],
                "temperature": 0.0,
                "max_tokens": 16,
            }
            request = urllib.request.Request(  # noqa: S310 - loopback fixture server
                relay.endpoint + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer unused"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - loopback fixture server
                assert response.status == 200
    finally:
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)

    assert observed["authorization"] == f"Bearer {secret}"
    assert observed["body"]["model"] == "hub-qwen"
    event_text = events_path.read_text()
    assert secret not in event_text
    assert "private document text" not in event_text
    event = json.loads(event_text)
    assert event["http_status"] == 200
    assert event["request_hash"]
