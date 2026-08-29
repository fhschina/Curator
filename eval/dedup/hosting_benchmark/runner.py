# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Execution and acceptance gates for the paired hosting benchmark."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import yaml

from eval.dedup.config import LocalNddJudgeConfig, TokenizerConfig
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.judging.local_ndd import run_local_ndd_pending
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic
from eval.llm_judge.run_llm_judge import LocalJudgeRuntime

from .artifacts import complete_marker, load_run, next_attempt_root, read_json, read_jsonl
from .config import HostingBenchmarkConfig
from .contracts import execution_contract_digest
from .recovery import validate_recovery_run
from .relay import (
    RelayContext,
    RelayTarget,
    RequestRelay,
    audit_paired_request_events,
    require_no_context_overflow,
    select_successful_initial_request_events,
)


def _persist_idempotent_report(
    path: Path,
    report: dict[str, Any],
    *,
    volatile_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not path.is_file():
        write_json_atomic(path, report)
        return report
    existing = read_json(path)
    stable_existing = {key: value for key, value in existing.items() if key not in volatile_fields}
    stable_report = {key: value for key, value in report.items() if key not in volatile_fields}
    require(
        stable_existing == stable_report,
        "HOSTING_REPORT_CHANGED",
        "an existing benchmark report differs from the current validated state",
        path=str(path),
    )
    return existing


def _persist_dynamic_preflight_report(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    return _persist_idempotent_report(
        root / "dynamic_preflight.json",
        report,
        volatile_fields=frozenset({"checked_at_utc"}),
    )


def _visible_cuda_device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    require(
        len(devices) == 1 and devices[0] != "-1",
        "HOSTING_GPU_SELECTION_MISMATCH",
        "CUDA_VISIBLE_DEVICES must select exactly one B200",
    )
    return devices[0]


def _verify_model(config: HostingBenchmarkConfig) -> dict[str, Any]:
    provision_path = config.model.local_model_path / ".hosting-benchmark-revision.json"
    model_config_path = config.model.local_model_path / "config.json"
    require(provision_path.is_file(), "HOSTING_MODEL_REVISION_UNKNOWN", "local model revision marker is missing")
    provision = read_json(provision_path)
    require(
        provision.get("model_id") == config.model.local_model_id
        and provision.get("revision") == config.model.local_model_revision,
        "HOSTING_MODEL_REVISION_MISMATCH",
        "local model revision marker differs from the frozen checkpoint",
    )
    require(model_config_path.is_file(), "HOSTING_MODEL_MISSING", "local model config is missing")
    model_config = read_json(model_config_path)
    quantization = model_config.get("quantization_config")
    require(
        isinstance(quantization, dict) and quantization.get("quant_method") == "fp8",
        "HOSTING_MODEL_NOT_FP8",
        "local model is not the frozen FP8 variant",
    )
    index_candidates = sorted(config.model.local_model_path.glob("*.index.json"))
    require(index_candidates, "HOSTING_MODEL_INCOMPLETE", "local model has no weight index")
    weight_index = read_json(index_candidates[0])
    weight_map = weight_index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "HOSTING_MODEL_INCOMPLETE", "weight index is empty")
    shards = sorted({str(value) for value in weight_map.values()})
    for shard in shards:
        require(
            (config.model.local_model_path / shard).is_file(),
            "HOSTING_MODEL_INCOMPLETE",
            "a checkpoint shard is missing",
            shard=shard,
        )
    return {
        "model_config_sha256": sha256_file(model_config_path),
        "weight_index_sha256": sha256_file(index_candidates[0]),
        "configured_revision": config.model.local_model_revision,
        "quant_method": quantization["quant_method"],
        "weight_shards": len(shards),
        "weight_bytes": sum((config.model.local_model_path / shard).stat().st_size for shard in shards),
    }


def static_preflight(run_root: str | Path) -> dict[str, Any]:
    root, manifest, config = load_run(run_root)
    host = socket.gethostname()
    require(host.split(".")[0] == "umb-b200-218", "HOSTING_WRONG_HOST", "benchmark is on the wrong host", host=host)
    expected_environment = Path("/raid/hfang/llm_judge_env_pr2324_latest")
    require(
        Path(sys.prefix).resolve() == expected_environment.resolve(),
        "HOSTING_ENVIRONMENT_MISMATCH",
        "benchmark must use the existing locked llm_judge environment",
        executable=sys.executable,
        environment_prefix=sys.prefix,
    )
    for executable in ("etcd", "nats-server"):
        require(
            shutil.which(executable),
            "HOSTING_EXECUTABLE_MISSING",
            "serving dependency is unavailable",
            name=executable,
        )
    storage_free_bytes = shutil.disk_usage(root).free
    require(
        storage_free_bytes >= 100 * 1024**3,
        "HOSTING_STORAGE_LOW",
        "benchmark filesystem has less than 100 GiB free",
        free_bytes=storage_free_bytes,
    )
    for private_root in (config.runs_root, config.ray_temp_dir, config.checkpoint_root):
        private_root.mkdir(parents=True, exist_ok=True)
        private_root.chmod(0o700)
        require(os.access(private_root, os.W_OK), "HOSTING_STORAGE_READ_ONLY", "benchmark path is not writable")
    cuda_device = _visible_cuda_device()
    gpu_query = subprocess.run(  # noqa: S603 - fixed executable; device is a single argv value
        [
            "nvidia-smi",
            "-i",
            cuda_device,
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus = [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()]
    require(len(gpus) == 1 and "B200" in gpus[0], "HOSTING_GPU_MISMATCH", "exactly one visible B200 is required")
    credential = os.environ.get(config.model.api_key_env, "").strip()
    require(credential, "HOSTING_API_KEY_MISSING", "Inference Hub credential is empty")
    repository_root = Path(__file__).resolve().parents[3]
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(current_commit == manifest["git_commit"], "HOSTING_SOURCE_CHANGED", "benchmark code commit changed")
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(not git_status.strip(), "HOSTING_SOURCE_DIRTY", "benchmark checkout has uncommitted changes")
    runner_root = config.runner_config.parent
    expected_resources = manifest["judge_resources"]
    require(
        sha256_file(config.runner_config) == expected_resources["runner_config"],
        "HOSTING_SOURCE_CHANGED",
        "runner config changed",
    )
    runner_config = yaml.safe_load(config.runner_config.read_text(encoding="utf-8"))
    model = runner_config["models"][0]
    engine = model["dynamo_model"]["engine_kwargs"]
    inference = model["inference_parameters"]
    require(
        engine.get("tensor_parallel_size") == 1
        and engine.get("max_model_len") == 32_768
        and engine.get("max_num_seqs") == 8
        and engine.get("gpu_memory_utilization") == 0.8
        and engine.get("enforce_eager") is True,
        "HOSTING_LOCAL_ENGINE_MISMATCH",
        "Sarah local engine settings differ from the frozen single-B200 contract",
    )
    require(
        inference.get("temperature") == 0
        and inference.get("max_tokens") == 4_096
        and inference.get("timeout") == 600
        and inference.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False,
        "HOSTING_GENERATION_CONTRACT_MISMATCH",
        "Sarah generation or thinking contract differs",
    )
    judge = runner_config["execution"]["stages"][0]["judges"][0]
    require(
        sha256_file(runner_root / judge["system_prompt_path"]) == expected_resources["system_prompt"]
        and sha256_file(runner_root / judge["prompt_path"]) == expected_resources["pair_prompt"]
        and sha256_json(judge["scores"]) == expected_resources["rubrics"],
        "HOSTING_SOURCE_CHANGED",
        "Sarah prompt or rubric resources changed",
    )
    source_paths = {
        "candidate_pairs": config.source_run_root / "data" / "candidate_pairs.parquet",
        "pair_provenance": config.source_run_root / "data" / "pair_provenance.parquet",
        "corpus_manifest": config.source_run_root / "manifests" / "corpus_manifest.json",
    }
    for name, path in source_paths.items():
        require(path.is_file(), "HOSTING_SOURCE_MISSING", "source artifact is missing", path=str(path))
        require(
            sha256_file(path) == manifest["source_digests"][name],
            "HOSTING_SOURCE_CHANGED",
            "source artifact changed",
            path=str(path),
        )
    block_rows = []
    for block in manifest["workload"]["blocks"]:
        path = root / block["path"]
        require(path.is_file() and sha256_file(path) == block["sha256"], "HOSTING_WORKLOAD_CHANGED", "block changed")
        rows = read_jsonl(path)
        block_rows.append(
            [
                block["block_id"],
                [[row["canonical_pair_id"], row["candidate"]["judge_payload_hash"]] for row in rows],
            ]
        )
    warmup_path = root / manifest["workload"]["warmup_path"]
    require(
        warmup_path.is_file() and sha256_file(warmup_path) == manifest["workload"]["warmup_sha256"],
        "HOSTING_WORKLOAD_CHANGED",
        "warmup workload changed",
    )
    warmup_rows = read_jsonl(warmup_path)
    workload_digest = sha256_json(
        {
            "warmup": [[row["canonical_pair_id"], row["candidate"]["judge_payload_hash"]] for row in warmup_rows],
            "blocks": block_rows,
        }
    )
    require(workload_digest == manifest["workload_digest"], "HOSTING_WORKLOAD_CHANGED", "workload digest changed")
    identity_digest = sha256_json(
        {
            "config": config.digest,
            "git_commit": current_commit,
            "judge_resources": expected_resources,
            "source": manifest["source_digests"],
            "tokenizer": manifest["tokenizer"],
            "workload": workload_digest,
        }
    )
    require(identity_digest == manifest["identity_digest"], "HOSTING_IDENTITY_CHANGED", "run identity changed")
    report = {
        "schema_version": "hosting-static-preflight-v1",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "host": host,
        "cuda_visible_device": cuda_device,
        "python_executable": sys.executable,
        "storage_free_bytes": storage_free_bytes,
        "gpus": gpus,
        "credential_env": config.model.api_key_env,
        "credential_present": True,
        "git_commit": current_commit,
        "model": _verify_model(config),
        "status": "pass",
    }
    return _persist_idempotent_report(
        root / "static_preflight.json",
        report,
        volatile_fields=frozenset({"checked_at_utc", "storage_free_bytes"}),
    )


def _benchmark_evaluation_config(config: HostingBenchmarkConfig) -> SimpleNamespace:
    judge = LocalNddJudgeConfig(
        backend="local_ndd",
        model=config.model.logical_model,
        model_path=config.model.local_model_path,
        runner_config=config.runner_config,
        ray_temp_dir=config.ray_temp_dir,
        checkpoint_root=config.checkpoint_root,
        num_cpus=None,
        num_gpus=1,
        max_retries=config.max_retries,
        max_visible_tokens=config.payload.max_visible_tokens,
        window_tokens=config.payload.window_tokens,
        window_overlap_tokens=config.payload.window_overlap_tokens,
        prompt_version="dedup-judge-sarah-minhash-v1",
        schema_version="dedup-judge-output-v0",
        visible_payload_version="judge-visible-payload-v2",
    )
    return SimpleNamespace(judge=judge)


class _BorrowedBlockRuntime(AbstractContextManager["_BorrowedBlockRuntime"]):
    def __init__(
        self,
        *,
        runtime: LocalJudgeRuntime,
        relay: RequestRelay,
        target: RelayTarget,
        endpoint: str,
        block_id: str,
        attempt_root: Path,
        concurrency: int,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self.runtime = runtime
        self.relay = relay
        self.target = target
        self.endpoint_label = endpoint
        self.block_id = block_id
        self.attempt_root = attempt_root
        self.concurrency = concurrency
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.outer_attempt = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def run(self, **kwargs: Any) -> Any:
        self.outer_attempt += 1
        events_path = self.attempt_root / "events" / f"outer-attempt-{self.outer_attempt:02d}.jsonl"
        self.relay.set_route(
            target=self.target,
            context=RelayContext(
                endpoint=self.endpoint_label,
                block_id=self.block_id,
                outer_attempt=self.outer_attempt,
                events_path=events_path,
                queued_at_utc=datetime.now(UTC).isoformat(),
            ),
        )
        return self.runtime.run(
            **kwargs,
            inference_parameter_overrides={
                "judge": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": self.max_output_tokens,
                    "timeout": self.timeout_seconds,
                    "max_parallel_requests": self.concurrency,
                }
            },
        )


def run_endpoint_block(
    *,
    run_root: Path,
    config: HostingBenchmarkConfig,
    contract_digest: str,
    runtime: LocalJudgeRuntime,
    relay: RequestRelay,
    target: RelayTarget,
    endpoint: str,
    block_id: str,
    concurrency: int,
    rows: list[dict[str, Any]],
    measured: bool,
) -> dict[str, Any]:
    block_root = run_root / ("measured" if measured else "warmup") / endpoint / block_id
    if marker := complete_marker(block_root):
        return read_json(marker)
    attempt_root = next_attempt_root(block_root)
    terminal_path = attempt_root / "terminal.jsonl"
    pending = [row["candidate"] for row in rows]
    prepared = {row["canonical_pair_id"]: row["payload"] for row in rows}
    persisted: list[dict[str, Any]] = []
    write_lock = threading.Lock()

    def persist(result: dict[str, Any]) -> None:
        record = {**result, "validation_completed_at_utc": datetime.now(UTC).isoformat()}
        encoded = json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        with write_lock, terminal_path.open("a", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
            persisted.append(record)

    borrowed = _BorrowedBlockRuntime(
        runtime=runtime,
        relay=relay,
        target=target,
        endpoint=endpoint,
        block_id=block_id,
        attempt_root=attempt_root,
        concurrency=concurrency,
        max_output_tokens=config.model.max_output_tokens,
        timeout_seconds=config.timeout_seconds,
    )
    execution_started = time.monotonic()
    run_local_ndd_pending(
        _benchmark_evaluation_config(config),
        pending=pending,
        prepared=prepared,
        contract_digest=contract_digest,
        contract_version="hosting-judge-execution-contract-v1",
        work_root=attempt_root / f"ndd-{endpoint}-{block_id}-{attempt_root.name}",
        persist=persist,
        runtime_factory=lambda *_args, **_kwargs: borrowed,
    )
    execution_duration = time.monotonic() - execution_started
    event_paths = sorted((attempt_root / "events").glob("*.jsonl"))
    events = [row for path in event_paths for row in read_jsonl(path)]
    require(events, "HOSTING_REQUEST_EVENTS_MISSING", "block produced no request events")
    require(persisted, "HOSTING_TERMINAL_ROWS_MISSING", "block produced no terminal pair records")
    require(
        Counter(str(row["canonical_pair_id"]) for row in persisted)
        == Counter(str(row["canonical_pair_id"]) for row in rows),
        "HOSTING_TERMINAL_ACCOUNTING_MISMATCH",
        "terminal pair records differ from the immutable block workload",
    )
    require(
        all(event.get("generation_parameters_valid") is True for event in events),
        "HOSTING_GENERATION_CONTRACT_MISMATCH",
        "relay observed a request outside the frozen generation contract",
    )
    require(
        not any(event.get("thinking_content_observed", False) for event in events),
        "HOSTING_THINKING_NOT_DISABLED",
        "provider returned visible reasoning despite thinking-disabled request",
    )
    require_no_context_overflow(events, block_id=block_id)
    first_submitted = min(datetime.fromisoformat(row["submitted_at_utc"]) for row in events)
    last_validated = max(datetime.fromisoformat(row["validation_completed_at_utc"]) for row in persisted)
    measured_duration = (last_validated - first_submitted).total_seconds()
    require(measured_duration > 0, "HOSTING_INVALID_DURATION", "measured block duration is not positive")
    max_observed_outstanding = max(int(row["outstanding_at_submit"]) for row in events)
    require(
        max_observed_outstanding <= concurrency,
        "HOSTING_CONCURRENCY_EXCEEDED",
        "relay observed more outstanding requests than the frozen concurrency",
    )
    status_counts = Counter(int(row["http_status"]) for row in events)
    marker = {
        "schema_version": "hosting-block-complete-v1",
        "endpoint": endpoint,
        "block_id": block_id,
        "concurrency": concurrency,
        "measured": measured,
        "pairs": len(rows),
        "valid": sum(row["record_type"] == "result" for row in persisted),
        "failed": sum(row["record_type"] == "error" for row in persisted),
        "attempts": sum(int(row["attempts"]) for row in persisted),
        "started_at_utc": first_submitted.isoformat(),
        "completed_at_utc": last_validated.isoformat(),
        "duration_seconds": measured_duration,
        "execution_duration_seconds": execution_duration,
        "goodput_pairs_per_second": sum(row["record_type"] == "result" for row in persisted) / measured_duration,
        "http_status_counts": {str(key): value for key, value in sorted(status_counts.items())},
        "request_events": len(events),
        "max_observed_outstanding": max_observed_outstanding,
        "terminal_sha256": sha256_file(terminal_path),
        "event_sha256": sha256_json([sha256_file(path) for path in event_paths]),
        "contract_digest": contract_digest,
        "status": "quota_limited" if status_counts[429] else "complete",
        "attempt_root": str(attempt_root.relative_to(run_root)),
    }
    marker_name = "quota_limited.json" if status_counts[429] else "complete.json"
    write_json_atomic(attempt_root / marker_name, marker)
    return marker


def _initial_events(run_root: Path, marker: dict[str, Any]) -> list[dict[str, Any]]:
    attempt_root = run_root / marker["attempt_root"]
    events = [row for path in sorted((attempt_root / "events").glob("*.jsonl")) for row in read_jsonl(path)]
    return select_successful_initial_request_events(
        events,
        expected_count=int(marker["pairs"]),
        block_id=str(marker["block_id"]),
    )


def assert_paired_requests(run_root: Path, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_events = _initial_events(run_root, left)
    right_events = _initial_events(run_root, right)
    for endpoint, events in ((left["endpoint"], left_events), (right["endpoint"], right_events)):
        for event in events:
            require(
                event.get("generation_parameters_valid") is True,
                "HOSTING_GENERATION_CONTRACT_MISMATCH",
                "relay observed a request outside the frozen generation contract",
                endpoint=endpoint,
            )
            require(
                not event.get("thinking_content_observed", False),
                "HOSTING_THINKING_NOT_DISABLED",
                "provider returned visible reasoning despite thinking-disabled request",
                endpoint=endpoint,
            )
    return audit_paired_request_events(left["endpoint"], left_events, right["endpoint"], right_events)


class _GpuMonitor(AbstractContextManager["_GpuMonitor"]):
    def __init__(self, destination: Path, *, device: str) -> None:
        self.destination = destination
        self.device = device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self.destination.parent.mkdir(parents=True, exist_ok=True)

        def sample() -> None:
            while not self._stop.is_set():
                result = subprocess.run(  # noqa: S603 - fixed executable; device is a single argv value
                    [
                        "nvidia-smi",
                        "-i",
                        self.device,
                        "--query-gpu=index,name,utilization.gpu,memory.used,power.draw,clocks.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                with self.destination.open("a", encoding="utf-8") as file:
                    for line in result.stdout.splitlines():
                        file.write(f"{datetime.now(UTC).isoformat()},{line}\n")
                self._stop.wait(1)

        self._thread = threading.Thread(target=sample, name="hosting-gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _prompt_counter(tokenizer: TokenCounter) -> Any:
    require(tokenizer.tokenizer is not None, "HOSTING_TOKENIZER_INVALID", "Qwen tokenizer is unavailable")

    def count(messages: list[dict[str, Any]]) -> int:
        try:
            encoded = tokenizer.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            encoded = tokenizer.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(encoded)

    return count


def run_benchmark(run_root: str | Path) -> dict[str, Any]:
    root, manifest, config = load_run(run_root)
    completion_path = root / "run_complete.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        require(completion.get("status") == "complete", "HOSTING_RUN_INCOMPLETE", "run completion is not valid")
        return completion
    preflight_path = root / "static_preflight.json"
    require(preflight_path.is_file(), "HOSTING_PREFLIGHT_MISSING", "run static preflight first")
    require(read_json(preflight_path).get("status") == "pass", "HOSTING_PREFLIGHT_FAILED", "preflight did not pass")
    static_preflight(root)
    tokenizer = TokenCounter(
        TokenizerConfig(
            kind="huggingface",
            model_id=config.payload.tokenizer_model_id,
            revision=config.payload.tokenizer_revision,
            cache_root=config.payload.tokenizer_cache_root,
        )
    )
    require(tokenizer.contract() == manifest["tokenizer"], "HOSTING_TOKENIZER_CHANGED", "tokenizer contract changed")
    credential = os.environ[config.model.api_key_env]
    contract_digest = execution_contract_digest(manifest)
    recovery = validate_recovery_run(root, manifest)
    relay = RequestRelay(
        logical_model=config.model.logical_model,
        max_model_len=config.model.max_model_len,
        prompt_token_counter=_prompt_counter(tokenizer),
        upstream_timeout_seconds=config.timeout_seconds,
        expected_generation_parameters={
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.model.max_output_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    relay.start()
    runtime = LocalJudgeRuntime(
        config.runner_config,
        model_overrides={"judge": str(config.model.local_model_path)},
        served_model_overrides={"judge": config.model.logical_model},
        provider_endpoint=relay.endpoint,
        provider_api_key="unused",  # pragma: allowlist secret
        ray_temp_dir=str(config.ray_temp_dir),
        num_gpus=1,
    )
    startup_started = time.monotonic()
    runtime_started = False
    try:
        runtime.start()
        runtime_started = True
        cold_start = time.monotonic() - startup_started
        targets = {
            "local": RelayTarget(
                label="local-b200",
                base_url=runtime.inference_endpoint,
                model=config.model.logical_model,
            ),
            "hub": RelayTarget(
                label="nvidia-inference-hub",
                base_url=config.model.hub_base_url,
                model=config.model.hub_model,
                api_key=credential,
            ),
        }
        warmup_rows = read_jsonl(root / manifest["workload"]["warmup_path"])
        warmup_markers = {}
        with _GpuMonitor(root / "gpu_samples.csv", device=_visible_cuda_device()):
            for endpoint in ("local", "hub"):
                warmup_markers[endpoint] = run_endpoint_block(
                    run_root=root,
                    config=config,
                    contract_digest=contract_digest,
                    runtime=runtime,
                    relay=relay,
                    target=targets[endpoint],
                    endpoint=endpoint,
                    block_id="warmup-c08",
                    concurrency=8,
                    rows=warmup_rows,
                    measured=False,
                )
            require(
                warmup_markers["hub"]["status"] != "quota_limited",
                "HOSTING_HUB_QUOTA_LIMITED",
                "Hub returned 429 during the no-cap quota probe",
            )
            for endpoint, marker in warmup_markers.items():
                require(
                    int(marker["valid"]) == int(marker["pairs"]),
                    "HOSTING_SMOKE_INVALID",
                    "endpoint did not produce schema-valid results for the full warm-up",
                    endpoint=endpoint,
                )
            warmup_audit = assert_paired_requests(root, warmup_markers["local"], warmup_markers["hub"])
            _persist_dynamic_preflight_report(
                root,
                {
                    "schema_version": "hosting-dynamic-preflight-v2",
                    "checked_at_utc": datetime.now(UTC).isoformat(),
                    **warmup_audit,
                    "thinking_disabled_accepted": True,
                    "hub_quota_probe_requests": config.workload.warmup_pairs,
                    "hub_http_429": 0,
                    "status": "pass",
                },
            )
            measured_markers = []
            for block in manifest["workload"]["blocks"]:
                rows = read_jsonl(root / block["path"])
                paired = {}
                for endpoint in block["endpoint_order"]:
                    marker = run_endpoint_block(
                        run_root=root,
                        config=config,
                        contract_digest=contract_digest,
                        runtime=runtime,
                        relay=relay,
                        target=targets[endpoint],
                        endpoint=endpoint,
                        block_id=block["block_id"],
                        concurrency=int(block["concurrency"]),
                        rows=rows,
                        measured=True,
                    )
                    require(
                        marker["status"] != "quota_limited",
                        "HOSTING_HUB_QUOTA_LIMITED",
                        "a measured block returned 429",
                        block_id=block["block_id"],
                    )
                    paired[endpoint] = marker
                    measured_markers.append(marker)
                assert_paired_requests(root, paired["local"], paired["hub"])
        completion = {
            "schema_version": "hosting-run-complete-v1",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "cold_start_seconds": cold_start,
            "contract_digest": contract_digest,
            "recovery": recovery,
            "warmup": warmup_markers,
            "measured": measured_markers,
            "status": "complete",
        }
        write_json_atomic(completion_path, completion)
        return completion
    finally:
        if runtime_started:
            runtime.stop()
        relay.stop()
