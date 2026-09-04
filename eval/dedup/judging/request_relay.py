# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Credential-safe loopback relay for OpenAI-compatible Judge requests."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self


class _LoopbackServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        _host, port = self.server_address[:2]
        self.server_name = "localhost"
        self.server_port = port


@dataclass(frozen=True, slots=True)
class RelayContext:
    block_id: str
    outer_attempt: int
    events_path: Path


class RequestRelay:
    """Inject upstream credentials without exposing them to Ray or pipeline logs."""

    def __init__(
        self,
        *,
        logical_model: str,
        upstream_base_url: str,
        upstream_model: str,
        upstream_api_key: str,
        timeout_seconds: float,
        expected_generation_parameters: dict[str, Any],
    ) -> None:
        self.logical_model = logical_model
        self.upstream_base_url = upstream_base_url
        self.upstream_model = upstream_model
        self._upstream_api_key = upstream_api_key
        self.timeout_seconds = timeout_seconds
        self.expected_generation_parameters = expected_generation_parameters
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._active = 0
        self._sequence = 0
        self._context: RelayContext | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        if self._server is None:
            raise RuntimeError("request relay is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def set_context(self, context: RelayContext) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("cannot change relay context while requests are active")
            context.events_path.parent.mkdir(parents=True, exist_ok=True)
            self._context = context

    def start(self) -> Self:
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                relay._handle(self)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self._server = _LoopbackServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="dedup-judge-relay", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def _begin(self) -> tuple[int, int, RelayContext]:
        with self._lock:
            if self._context is None:
                raise RuntimeError("relay context is not configured")
            self._active += 1
            self._sequence += 1
            return self._sequence, self._active, self._context

    def _finish(self) -> None:
        with self._lock:
            self._active -= 1

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, status: int, content_type: str, body: bytes) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.close_connection = True

    def _write_event(self, context: RelayContext, event: dict[str, Any]) -> None:
        encoded = json.dumps(event, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        with self._write_lock, context.events_path.open("a", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        sequence = 0
        outstanding = 0
        context: RelayContext | None = None
        started = time.monotonic()
        status = 500
        content_type = "application/json"
        response_body = b'{"error":{"message":"relay forwarding failed"}}'
        request_hash = ""
        error_type: str | None = None
        try:
            sequence, outstanding, context = self._begin()
            if handler.path != "/v1/chat/completions":
                status = 404
                response_body = b'{"error":{"message":"unsupported relay path"}}'
                return
            length = int(handler.headers.get("Content-Length", "0"))
            request_body = json.loads(handler.rfile.read(length))
            if not isinstance(request_body, dict):
                raise TypeError("chat request body must be an object")  # noqa: TRY301
            if request_body.get("model") != self.logical_model:
                raise ValueError("chat request model differs from the logical model")  # noqa: TRY301
            if request_body.get("stream") is True:
                raise ValueError("streaming responses are not supported")  # noqa: TRY301
            for key, expected in self.expected_generation_parameters.items():
                if request_body.get(key) != expected:
                    message = f"chat request {key} differs from the frozen contract"
                    raise ValueError(message)  # noqa: TRY301
            request_hash = hashlib.sha256(
                json.dumps(request_body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            forwarded = {**request_body, "model": self.upstream_model}
            request = urllib.request.Request(  # noqa: S310 - target is the frozen NVIDIA HTTPS endpoint
                self.upstream_base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(forwarded, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                headers={"Authorization": f"Bearer {self._upstream_api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    response_body = response.read()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as exc:
                response_body = exc.read()
                status = exc.code
                content_type = exc.headers.get("Content-Type", "application/json")
                error_type = "http_error"
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            status = 400
            error_type = exc.__class__.__name__
            response_body = json.dumps({"error": {"message": str(exc)}}).encode("utf-8")
        except Exception as exc:  # noqa: BLE001 - relay must return a terminal, audited response
            status = 502
            error_type = (
                "timeout"
                if isinstance(exc, (TimeoutError, socket.timeout))
                or (isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout)))
                else exc.__class__.__name__
            )
        finally:
            if context is not None:
                try:
                    self._write_event(
                        context,
                        {
                            "schema_version": "dedup-judge-relay-event-v1",
                            "sequence": sequence,
                            "block_id": context.block_id,
                            "outer_attempt": context.outer_attempt,
                            "outstanding_at_submit": outstanding,
                            "request_hash": request_hash,
                            "completed_at_utc": datetime.now(UTC).isoformat(),
                            "duration_seconds": time.monotonic() - started,
                            "http_status": status,
                            "error_type": error_type,
                        },
                    )
                finally:
                    self._finish()
            self._send(handler, status, content_type, response_body)
