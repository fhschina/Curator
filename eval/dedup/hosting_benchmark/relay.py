# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Audited loopback relay for endpoint-neutral OpenAI chat requests."""

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
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

from eval.dedup.validation import require


class LoopbackThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that avoids a reverse-DNS lookup for its numeric loopback bind."""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        _host, port = self.server_address[:2]
        self.server_name = "localhost"
        self.server_port = port


def canonical_request_hash(body: dict[str, Any]) -> str:
    """Hash the exact logical request before target-specific model rewriting."""

    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_paired_request_events(
    left_endpoint: str,
    left_events: list[dict[str, Any]],
    right_endpoint: str,
    right_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate client-observable prompt identity and report provider usage drift."""

    require(left_endpoint != right_endpoint, "HOSTING_ENDPOINT_INVALID", "paired endpoint labels must differ")
    require(
        Counter(event["request_hash"] for event in left_events)
        == Counter(event["request_hash"] for event in right_events),
        "HOSTING_REQUEST_CONTRACT_MISMATCH",
        "paired endpoints received different initial requests",
    )
    canonical: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    provider: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for endpoint, events in ((left_endpoint, left_events), (right_endpoint, right_events)):
        for event in events:
            prompt_tokens_local = event.get("prompt_tokens_local")
            require(
                isinstance(prompt_tokens_local, int) and not isinstance(prompt_tokens_local, bool),
                "HOSTING_CANONICAL_PROMPT_USAGE_MISSING",
                "relay omitted the pinned-tokenizer prompt count",
                endpoint=endpoint,
            )
            prompt_tokens_provider = event.get("usage", {}).get("prompt_tokens")
            require(
                isinstance(prompt_tokens_provider, int) and not isinstance(prompt_tokens_provider, bool),
                "HOSTING_PROMPT_USAGE_MISSING",
                "provider omitted prompt token usage",
                endpoint=endpoint,
            )
            request_hash = str(event["request_hash"])
            canonical[endpoint][request_hash].append(prompt_tokens_local)
            provider[endpoint][request_hash].append(prompt_tokens_provider)

    canonical_left = {request_hash: sorted(values) for request_hash, values in canonical[left_endpoint].items()}
    canonical_right = {request_hash: sorted(values) for request_hash, values in canonical[right_endpoint].items()}
    require(
        canonical_left == canonical_right,
        "HOSTING_CANONICAL_PROMPT_TOKEN_MISMATCH",
        "paired endpoints produced different pinned-tokenizer prompt counts",
    )
    drift = Counter()
    mismatches = 0
    for request_hash, left_values in provider[left_endpoint].items():
        right_values = provider[right_endpoint][request_hash]
        require(
            len(left_values) == len(right_values),
            "HOSTING_REQUEST_ACCOUNTING_MISMATCH",
            "paired provider usage counts have different multiplicity",
        )
        for left_value, right_value in zip(sorted(left_values), sorted(right_values), strict=True):
            if left_value != right_value:
                mismatches += 1
                drift[right_value - left_value] += 1
    return {
        "request_count": len(left_events),
        "request_hash_equality": True,
        "canonical_prompt_token_equality": True,
        "prompt_token_count_source": "pinned_client_tokenizer",
        "provider_prompt_token_usage_equality": mismatches == 0,
        "provider_prompt_token_usage_mismatched_requests": mismatches,
        "provider_prompt_token_usage_delta_counts": {str(delta): count for delta, count in sorted(drift.items())},
    }


@dataclass(frozen=True, slots=True)
class RelayTarget:
    label: str
    base_url: str
    model: str
    api_key: str | None = None


@dataclass(frozen=True, slots=True)
class RelayContext:
    endpoint: str
    block_id: str
    outer_attempt: int
    events_path: Path
    queued_at_utc: str


class RequestRelay:
    """Forward requests while recording only timings, hashes, status, and token counts."""

    def __init__(
        self,
        *,
        logical_model: str,
        max_model_len: int,
        prompt_token_counter: Callable[[list[dict[str, Any]]], int] | None = None,
        upstream_timeout_seconds: float = 600,
        expected_generation_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.logical_model = logical_model
        self.max_model_len = max_model_len
        self.prompt_token_counter = prompt_token_counter
        self.upstream_timeout_seconds = upstream_timeout_seconds
        self.expected_generation_parameters = expected_generation_parameters or {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._active = 0
        self._sequence = 0
        self._target: RelayTarget | None = None
        self._context: RelayContext | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        if self._server is None:
            msg = "request relay is not running"
            raise RuntimeError(msg)
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def set_route(self, *, target: RelayTarget, context: RelayContext) -> None:
        with self._lock:
            if self._active:
                msg = "cannot change relay route while requests are active"
                raise RuntimeError(msg)
            context.events_path.parent.mkdir(parents=True, exist_ok=True)
            self._target = target
            self._context = context

    def start(self) -> Self:
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                relay._handle(self)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self._server = LoopbackThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="judge-request-relay", daemon=True)
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

    def _snapshot_route(self) -> tuple[int, int, RelayTarget, RelayContext]:
        with self._lock:
            if self._target is None or self._context is None:
                msg = "relay route is not configured"
                raise RuntimeError(msg)
            self._active += 1
            self._sequence += 1
            return self._sequence, self._active, self._target, self._context

    def _release_route(self) -> None:
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

    def _event(self, context: RelayContext, event: dict[str, Any]) -> None:
        encoded = json.dumps(event, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        with self._write_lock, context.events_path.open("a", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        sequence = 0
        outstanding_at_submit = 0
        target: RelayTarget | None = None
        context: RelayContext | None = None
        started = time.monotonic()
        started_at = datetime.now(UTC).isoformat()
        status = 500
        content_type = "application/json"
        response_body = b'{"error":{"message":"relay forwarding failed"}}'
        request_hash = ""
        prompt_tokens_local: int | None = None
        usage: dict[str, int] = {}
        error_type: str | None = None
        message_count: int | None = None
        message_roles: list[str] = []
        thinking_content_observed = False
        generation_parameters_valid = False
        try:
            sequence, outstanding_at_submit, target, context = self._snapshot_route()
            if handler.path != "/v1/chat/completions":
                status = 404
                response_body = b'{"error":{"message":"unsupported relay path"}}'
                return
            length = int(handler.headers.get("Content-Length", "0"))
            raw = handler.rfile.read(length)
            request_body = json.loads(raw)
            if not isinstance(request_body, dict):
                raise TypeError("chat request body must be an object")  # noqa: TRY301
            if request_body.get("model") != self.logical_model:
                raise ValueError("chat request model differs from the frozen logical model")  # noqa: TRY301
            if request_body.get("stream") is True:
                raise ValueError("hosting benchmark requires non-streaming responses")  # noqa: TRY301
            for key, expected in self.expected_generation_parameters.items():
                if request_body.get(key) != expected:
                    message = f"chat request {key} differs from the frozen generation contract"
                    raise ValueError(message)  # noqa: TRY301
            generation_parameters_valid = True
            messages = request_body.get("messages")
            if not isinstance(messages, list):
                raise TypeError("chat request messages must be a list")  # noqa: TRY301
            message_count = len(messages)
            message_roles = [str(message.get("role")) for message in messages if isinstance(message, dict)]
            request_hash = canonical_request_hash(request_body)
            if self.prompt_token_counter is not None:
                prompt_tokens_local = self.prompt_token_counter(messages)
                max_tokens = request_body.get("max_tokens")
                if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
                    raise TypeError("chat request max_tokens must be an integer")  # noqa: TRY301
                if prompt_tokens_local + max_tokens > self.max_model_len:
                    status = 400
                    error_type = "context_overflow"
                    response_body = b'{"error":{"message":"frozen request exceeds benchmark context budget"}}'
                    return

            forwarded = dict(request_body)
            forwarded["model"] = target.model
            target_url = target.base_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if target.api_key:
                headers["Authorization"] = f"Bearer {target.api_key}"
            request = urllib.request.Request(  # noqa: S310 - target is a configured HTTP(S) API base URL
                target_url,
                data=json.dumps(forwarded, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.upstream_timeout_seconds) as response:  # noqa: S310
                    response_body = response.read()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as exc:
                response_body = exc.read()
                status = exc.code
                content_type = exc.headers.get("Content-Type", "application/json")
                error_type = "http_error"
            try:
                parsed_response = json.loads(response_body)
                if isinstance(parsed_response, dict) and isinstance(parsed_response.get("usage"), dict):
                    usage = {
                        str(key): value
                        for key, value in parsed_response["usage"].items()
                        if isinstance(value, int) and not isinstance(value, bool)
                    }
                if isinstance(parsed_response, dict) and isinstance(parsed_response.get("choices"), list):
                    for choice in parsed_response["choices"]:
                        message = choice.get("message") if isinstance(choice, dict) else None
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        reasoning = message.get("reasoning_content")
                        thinking_content_observed = bool(reasoning) or (
                            isinstance(content, str) and "<think>" in content.lower()
                        )
                        if thinking_content_observed:
                            break
            except json.JSONDecodeError:
                error_type = error_type or "non_json_response"
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            status = 400
            error_type = exc.__class__.__name__
            response_body = json.dumps({"error": {"message": str(exc)}}).encode("utf-8")
        except Exception as exc:  # noqa: BLE001 - relay must return and audit terminal failures
            status = 502
            error_type = (
                "timeout"
                if isinstance(exc, (TimeoutError, socket.timeout))
                or (isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout)))
                else exc.__class__.__name__
            )
        finally:
            if context is not None and target is not None:
                try:
                    self._event(
                        context,
                        {
                            "schema_version": "hosting-relay-event-v1",
                            "sequence": sequence,
                            "endpoint": context.endpoint,
                            "block_id": context.block_id,
                            "outer_attempt": context.outer_attempt,
                            "target_label": target.label,
                            "outstanding_at_submit": outstanding_at_submit,
                            "request_hash": request_hash,
                            "queued_at_utc": context.queued_at_utc,
                            "submitted_at_utc": started_at,
                            "response_completed_at_utc": datetime.now(UTC).isoformat(),
                            "duration_seconds": time.monotonic() - started,
                            "http_status": status,
                            "prompt_tokens_local": prompt_tokens_local,
                            "message_count": message_count,
                            "message_roles": message_roles,
                            "generation_parameters_valid": generation_parameters_valid,
                            "usage": usage,
                            "thinking_content_observed": thinking_content_observed,
                            "error_type": error_type,
                        },
                    )
                finally:
                    self._release_route()
            self._send(handler, status, content_type, response_body)
