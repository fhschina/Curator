# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the real inference boundary without replacing Data Designer or its parser."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

QWEN_CONTEXT_SIZE = 32768


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

    def __init__(self, upstream: str, receipt_dir: Path):
        self.receipt_dir = receipt_dir
        receipt_dir.mkdir(parents=True, exist_ok=True)
        target = upstream.rstrip("/")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                self.forward()

            def do_POST(self) -> None:
                self.forward()

            def forward(self) -> None:
                started = time.time()
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                status, response_body = 502, b""
                failure = None
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
                    receipt = {
                        "started": started,
                        "elapsed_s": time.time() - started,
                        "method": self.command,
                        "path": self.path,
                        "request": body.decode("utf-8", errors="replace"),
                        "status": status,
                        "response": response_body.decode("utf-8", errors="replace"),
                        "transport_error": failure,
                    }
                    (receipt_dir / f"{uuid.uuid4().hex}.json").write_text(json.dumps(receipt), encoding="utf-8")
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
        self._relay = RecordingRelay(endpoint, self.receipt_dir / "http")
        self._relay.start()
        return super()._build_judge_stages(endpoint=self._relay.endpoint)

    def run(self) -> WorkflowRunResult:
        try:
            return super().run()
        finally:
            if relay := getattr(self, "_relay", None):
                relay.close()
