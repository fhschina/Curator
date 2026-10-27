# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the real inference boundary without replacing Data Designer or its parser."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

    from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

QWEN_CONTEXT_SIZE = 32768
QWEN_IMAGE_ID = "sha256:3d8ebe07d1b64ceab33030e09a9076e77754d3ed70a19fb0c36b0168d6b94d6f"


def message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def request_pair_id(body: bytes) -> str | None:
    try:
        for message in json.loads(body).get("messages", []):
            content = message_text(message)
            if message.get("role") == "user" and isinstance(content, str) and content.startswith("Audit record: "):
                return content.splitlines()[0].removeprefix("Audit record: ").strip() or None
    except (ValueError, TypeError, AttributeError):
        return None
    return None


@lru_cache(maxsize=32)
def response_recipe(schema: str) -> StructuredResponseRecipe:
    from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

    return StructuredResponseRecipe(json_schema=json.loads(schema))


def audit_attempt(receipt: dict) -> dict:
    """Replay the native parser against the exact schema sent on this HTTP attempt."""
    from data_designer.engine.models.parsers.errors import ParserException

    result = {"pair_id": receipt.get("pair_id"), "column": "unknown", "status": "incomplete", "error": None}
    try:
        request = json.loads(receipt.get("request", "{}"))
        text = "\n".join(message_text(m) for m in request["messages"] if m.get("role") == "user")
        schemas = re.findall(r"<response_schema>\s*(\{.*?\})\s*</response_schema>", text, re.DOTALL)
        if schemas:
            result["column"] = json.loads(schemas[-1]).get("title", "unknown")
    except (ValueError, KeyError, TypeError, AttributeError):
        schemas = []
    if not receipt.get("completed", receipt.get("status") is not None):
        return result
    if receipt.get("transport_error") or receipt.get("status") != HTTPStatus.OK:
        return {
            **result,
            "status": "transport_error",
            "error": receipt.get("transport_error") or str(receipt.get("status")),
        }
    if not schemas:
        return {**result, "status": "missing_schema"}
    try:
        response = json.loads(receipt["response"])
        choice = response["choices"][0]
        if choice.get("finish_reason", "stop") != "stop":
            return {**result, "status": "output_incomplete", "error": str(choice.get("finish_reason"))}
        answer = message_text(choice["message"])
        response_recipe(schemas[-1]).parse(answer)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, ParserException) as error:
        return {**result, "status": "parse_error", "error": str(error)[:400]}
    return {**result, "status": "valid"}


def environment_receipt(repo: Path) -> dict:
    """Reject an installed Curator masking the checkout on either driver or worker."""
    import data_designer.config as dd

    import nemo_curator

    source = Path(nemo_curator.__file__).resolve()
    if not source.is_relative_to(repo.resolve()):
        msg = f"Curator loaded from {source}, expected checkout {repo}"
        raise RuntimeError(msg)
    if version("data-designer") != "0.9.1" or not hasattr(dd, "SkipConfig"):
        msg = "This frozen experiment requires Data Designer 0.9.1 with native SkipConfig"
        raise RuntimeError(msg)
    return {
        "curator_source": str(source),
        "data_designer": version("data-designer"),
        "data_designer_source": str(Path(dd.__file__).resolve()),
        "pid": os.getpid(),
    }


def existing_qwen_service(container: str, endpoint: str) -> dict:
    """Bind the loopback endpoint to the live, pinned one-GPU serving container."""
    live = json.loads(subprocess.check_output([shutil.which("docker"), "inspect", container], text=True))[0]  # noqa: S603
    url = urlparse(endpoint)
    ports = live["NetworkSettings"]["Ports"].get("8000/tcp") or []
    required = {
        "--served-model-name": "Qwen/Qwen3.8-27B-FP8",
        "--tensor-parallel-size": "1",
        "--dtype": "auto",
        "--quantization": "fp8",
        "--max-model-len": "32768",
        "--max-num-seqs": "8",
        "--gpu-memory-utilization": "0.8",
    }
    command = live["Config"]["Cmd"]
    args = dict(pairwise(command))
    mounts = {m["Destination"]: m for m in live["Mounts"]}
    devices = live["HostConfig"]["DeviceRequests"]
    if (
        not live["State"]["Running"]
        or live["Image"] != QWEN_IMAGE_ID
        or command[:2] != ["serve", "/qwen"]
        or any(args.get(k) != v for k, v in required.items())
        or url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or url.path.rstrip("/") != "/v1"
        or not any(p["HostIp"] == "127.0.0.1" and int(p["HostPort"]) == url.port for p in ports)
        or not devices
        or len(devices) != 1
        or len(devices[0].get("DeviceIDs") or []) != 1
        or mounts.get("/qwen", {}).get("Source") != "/raid/hfang/hf_cache/Qwen3.8-27B-FP8"
        or mounts.get("/qwen", {}).get("RW", True)
    ):
        msg = "Existing Qwen service does not match the frozen one-GPU FP8 serving configuration"
        raise ValueError(msg)
    response = requests.get(endpoint.rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    models = response.json()["data"]
    if (
        len(models) != 1
        or models[0]["id"] != required["--served-model-name"]
        or models[0]["max_model_len"] != QWEN_CONTEXT_SIZE
    ):
        msg = "The endpoint model registry disagrees with the serving receipt"
        raise ValueError(msg)
    return {
        "container_id": live["Id"],
        "image_id": live["Image"],
        "image": live["Config"]["Image"],
        "entrypoint": live["Config"]["Entrypoint"],
        "command": command,
        "mounts": live["Mounts"],
        "devices": devices,
        "ports": ports,
        "endpoint": endpoint,
        "models": models,
    }


def service_pool(path: Path, replicas: int) -> list[dict]:
    """Validate every live endpoint before dispatching any judge request."""
    manifest = json.loads(path.read_text())
    if manifest.get("model_revision") != "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a":
        msg = "Service pool checkpoint revision mismatch"
        raise ValueError(msg)
    services = manifest["services"]
    if len(services) != replicas or replicas not in (1, 8):
        msg = "Service pool size must match the requested one or eight replicas"
        raise ValueError(msg)
    verified = [existing_qwen_service(item["container"], item["endpoint"]) for item in services]
    devices = [item["devices"][0]["DeviceIDs"][0] for item in verified]
    if len(set(devices)) != replicas or len({item["endpoint"] for item in verified}) != replicas:
        msg = "Every replica requires a distinct GPU and endpoint"
        raise ValueError(msg)
    return verified


@dataclass
class EnvironmentAuditStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    repo: str
    receipt_dir: str
    name: str = "policy_environment_audit"

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        receipt = environment_receipt(Path(self.repo))
        path = Path(self.receipt_dir) / f"worker-{os.getpid()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return batch


class RecordingRelay:
    """Loopback relay: one receipt for every actual HTTP attempt, with no added retries."""

    def __init__(self, upstream: str | list[str], receipt_dir: Path):
        self.receipt_dir = receipt_dir
        receipt_dir.mkdir(parents=True, exist_ok=True)
        targets = [u.rstrip("/") for u in ([upstream] if isinstance(upstream, str) else upstream)]
        if not targets:
            msg = "At least one upstream is required"
            raise ValueError(msg)
        lock = threading.Lock()
        slots = [threading.BoundedSemaphore(8) for _ in targets]
        active = [0 for _ in targets]
        next_target = 0

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                self.forward()

            def do_POST(self) -> None:
                self.forward()

            def forward(self) -> None:
                nonlocal next_target
                with lock:
                    target_index = next_target % len(targets)
                    next_target += 1
                target = targets[target_index]
                slots[target_index].acquire()
                with lock:
                    active[target_index] += 1
                    concurrency = active[target_index]
                started = time.time()
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                status, response_body = 502, b""
                failure = None
                receipt_path = receipt_dir / f"{uuid.uuid4().hex}.json"
                receipt = {
                    "pair_id": request_pair_id(body),
                    "upstream": target,
                    "replica_index": target_index,
                    "replica_active_requests": concurrency,
                    "started": started,
                    "method": self.command,
                    "path": self.path,
                    "request": body.decode("utf-8", errors="replace"),
                    "status": None,
                    "response": "",
                    "transport_error": None,
                    "completed": False,
                }
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                try:
                    response = requests.request(
                        self.command,
                        target + self.path,
                        data=body or None,
                        headers={k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}},
                        timeout=650,
                    )
                    status, response_body = response.status_code, response.content
                except requests.RequestException as error:
                    failure = str(error)
                    response_body = json.dumps({"error": failure}).encode()
                finally:
                    receipt.update(
                        {
                            "elapsed_s": time.time() - started,
                            "status": status,
                            "response": response_body.decode("utf-8", errors="replace"),
                            "transport_error": failure,
                            "completed": True,
                        }
                    )
                    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                    with lock:
                        active[target_index] -= 1
                    slots[target_index].release()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> str:
        self.thread.start()
        return self.endpoint

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


class RecordedWorkflowMixin:
    """Wrap Sarah's workflow endpoints; keep its native model configs and stages."""

    receipt_dir: Path

    def _build_judge_stages(self, *, endpoint: str) -> list:
        import critics
        import policy_runtime
        import ray

        ray.init(ignore_reinit_error=True, runtime_env={"py_modules": [critics, policy_runtime]})
        self._relay = RecordingRelay(getattr(self, "service_endpoints", None) or endpoint, self.receipt_dir / "http")
        self._relay.start()
        return super()._build_judge_stages(endpoint=self._relay.endpoint)

    def run(self) -> WorkflowRunResult:
        try:
            return super().run()
        finally:
            if relay := getattr(self, "_relay", None):
                relay.close()
