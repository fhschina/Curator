# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hub and local-Qwen backends for the Judge v0.7 release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

from eval.dedup.core.validation import require, sha256_file, sha256_json, write_json_atomic
from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.runtime import JUDGE_CONTRACT_VERSION, TOOL_VERSION, contract, release, state, transport

VERSION = TOOL_VERSION
HERE = Path(__file__).resolve()
PROTOCOL = HERE.parents[1] / "release/README.md"
DEFAULT_LOCAL_RUNNER_CONFIG = HERE.parents[1] / "release/local_runner/config.yaml"
LOCAL_MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
LOCAL_MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
LOCAL_ENDPOINT = "local://ray-dynamo-vllm"
LOCAL_BACKEND_SCHEMA = "dedup-local-backend-v1"
LEGACY_LOCAL_BACKEND_SCHEMA = "v07-local-backend-v2"
LOCAL_MAX_SEQS_PER_REPLICA = 8
LOCAL_WORKERS_PER_REPLICA = 8
LOCAL_MIN_INTERVAL_SECONDS = 0.001
BACKENDS = ("hub", "local")


def _backend(manifest: dict) -> str:
    """Treat historical manifests without an explicit backend as Hub runs."""
    value = manifest.get("backend", "hub")
    require(value in BACKENDS, "V07_BACKEND", "supported v0.7 backend")
    return value


def verify(root: Path) -> dict:
    manifest = release.verify(root)
    backend = _backend(manifest)
    if backend == "local":
        local = manifest.get("local_backend")
        require(
            isinstance(local, dict)
            and local.get("schema_version") in {LOCAL_BACKEND_SCHEMA, LEGACY_LOCAL_BACKEND_SCHEMA}
            and manifest.get("endpoint") == LOCAL_ENDPOINT
            and manifest.get("model") == contract.LOGICAL_MODEL
            and local.get("num_gpus") == local.get("replicas") == len(local.get("devices", []))
            and len(local.get("devices", [])) == len(set(local.get("devices", [])))
            and local.get("tensor_parallel_size") == 1
            and local.get("cuda_visible_devices") == ",".join(local.get("devices", []))
            and manifest.get("workers") == local.get("transport", {}).get("workers")
            and local.get("transport", {}).get("workers") == LOCAL_WORKERS_PER_REPLICA * local.get("replicas", 0)
            and manifest.get("min_interval_seconds") == local.get("transport", {}).get("min_interval_seconds"),
            "V07_LOCAL_MANIFEST",
            "frozen local backend contract",
        )
    return manifest


def _model_contract(model_path: Path) -> dict:
    marker_path = model_path / ".hosting-benchmark-revision.json"
    config_path = model_path / "config.json"
    require(marker_path.is_file(), "V07_LOCAL_MODEL_REVISION", "local model revision marker")
    marker = release.read(marker_path)
    require(
        marker.get("model_id") == LOCAL_MODEL_ID and marker.get("revision") == LOCAL_MODEL_REVISION,
        "V07_LOCAL_MODEL_REVISION",
        "pinned local Qwen checkpoint revision",
    )
    require(config_path.is_file(), "V07_LOCAL_MODEL", "local model config")
    model_config = release.read(config_path)
    quantization = model_config.get("quantization_config")
    require(
        isinstance(quantization, dict) and quantization.get("quant_method") == "fp8",
        "V07_LOCAL_MODEL",
        "FP8 local checkpoint",
    )
    indexes = sorted(model_path.glob("*.index.json"))
    require(len(indexes) == 1, "V07_LOCAL_MODEL", "one local weight index")
    weight_index = release.read(indexes[0])
    weight_map = weight_index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "V07_LOCAL_MODEL", "nonempty local weight index")
    shards = sorted({str(value) for value in weight_map.values()})
    require(
        all((model_path / shard).is_file() for shard in shards),
        "V07_LOCAL_MODEL",
        "all local checkpoint shards",
    )
    return {
        "model_id": LOCAL_MODEL_ID,
        "revision": LOCAL_MODEL_REVISION,
        "path": str(model_path),
        "model_config_sha256": sha256_file(config_path),
        "weight_index_sha256": sha256_file(indexes[0]),
        "quant_method": "fp8",
        "weight_shards": len(shards),
        "weight_bytes": sum((model_path / shard).stat().st_size for shard in shards),
    }


def _runner_contract(runner_config: Path) -> dict:
    require(runner_config.is_file(), "V07_LOCAL_CONFIG", "local runner config")
    config = yaml.safe_load(runner_config.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "V07_LOCAL_CONFIG", "local runner mapping")
    models = config.get("models")
    require(isinstance(models, list) and len(models) == 1, "V07_LOCAL_CONFIG", "one local model")
    model = models[0]
    require(isinstance(model, dict) and model.get("alias") == "judge", "V07_LOCAL_CONFIG", "judge model alias")
    dynamo = model.get("dynamo_model")
    require(isinstance(dynamo, dict), "V07_LOCAL_CONFIG", "Dynamo model settings")
    engine = dynamo.get("engine_kwargs")
    expected = {
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "max_num_seqs": LOCAL_MAX_SEQS_PER_REPLICA,
        "gpu_memory_utilization": 0.8,
        "enforce_eager": True,
    }
    require(
        isinstance(engine, dict) and all(engine.get(key) == value for key, value in expected.items()),
        "V07_LOCAL_CONFIG",
        "frozen single-GPU engine settings",
    )
    return {"path": str(runner_config), "sha256": sha256_file(runner_config), "engine": expected}


def _default_runtime_root(root: Path) -> Path:
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    return Path("/tmp") / f"dedup-local-{suffix}"  # noqa: S108 - short Ray socket path is required


def _parse_devices(value: str) -> list[str]:
    devices = [device.strip() for device in value.split(",")]
    require(
        devices and all(devices) and len(devices) == len(set(devices)) and all(device != "-1" for device in devices),
        "V07_LOCAL_DEVICE",
        "one or more unique visible GPUs",
    )
    return devices


def _local_contract(
    root: Path,
    *,
    model_path: Path,
    runner_config: Path,
    runtime_root: Path | None,
    tools_dir: Path,
    devices: str,
    replicas: int | None,
) -> dict:
    device_list = _parse_devices(devices)
    replica_count = len(device_list) if replicas is None else replicas
    require(
        type(replica_count) is int and replica_count == len(device_list),
        "V07_LOCAL_REPLICAS",
        "one tensor-parallel-1 replica per visible GPU",
    )
    runtime_root = (runtime_root or _default_runtime_root(root)).resolve()
    model_path = model_path.resolve()
    runner_config = runner_config.resolve()
    tools_dir = tools_dir.resolve()
    require(
        all((tools_dir / name).is_file() for name in ("etcd", "nats-server")),
        "V07_LOCAL_TOOLS",
        "local serving tools",
    )
    python_bin = Path(sys.executable).parent.absolute()
    require((python_bin / "ray").is_file(), "V07_LOCAL_TOOLS", "Ray CLI in the selected Python environment")
    return {
        "schema_version": LOCAL_BACKEND_SCHEMA,
        "checkpoint": _model_contract(model_path),
        "runner": _runner_contract(runner_config),
        "logical_model": contract.LOGICAL_MODEL,
        "endpoint_kind": LOCAL_ENDPOINT,
        "devices": device_list,
        "cuda_visible_devices": ",".join(device_list),
        "replicas": replica_count,
        "tensor_parallel_size": 1,
        "num_gpus": replica_count,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_bin": str(python_bin),
        "tools_dir": str(tools_dir),
        "runtime_root": str(runtime_root),
        "ray_temp_dir": str(runtime_root / "ray"),
        "uv_cache_dir": str(runtime_root / "uv"),
        "hf_home": str(runtime_root / "huggingface"),
        "generation": contract.GENERATION,
        "transport": {
            "workers": LOCAL_WORKERS_PER_REPLICA * replica_count,
            "workers_per_replica": LOCAL_WORKERS_PER_REPLICA,
            "min_interval_seconds": LOCAL_MIN_INTERVAL_SECONDS,
            "client_rpm_limit": None,
        },
    }


def prepare_local(
    root: Path,
    *,
    source_run: Path,
    model_path: Path,
    runner_config: Path = DEFAULT_LOCAL_RUNNER_CONFIG,
    runtime_root: Path | None = None,
    tools_dir: Path,
    devices: str = "0",
    replicas: int | None = None,
    smoke_only: bool = False,
) -> dict:
    local = _local_contract(
        root,
        model_path=model_path,
        runner_config=runner_config,
        runtime_root=runtime_root,
        tools_dir=tools_dir,
        devices=devices,
        replicas=replicas,
    )
    release.prepare(root, source=source_run, smoke_only=smoke_only)
    manifest = release.read(root / "manifest.json")
    local_sources = [
        HERE,
        PROTOCOL,
        Path(local["runner"]["path"]),
        Path(local["runner"]["path"]).parent / "cutlass_compat/sitecustomize.py",
        HERE.parents[3] / "nemo_curator/eval/llm_judge/workflow.py",
        HERE.parents[3] / "nemo_curator/eval/llm_judge/runtime.py",
    ]
    manifest["sources"].update({str(path): sha256_file(path) for path in local_sources})
    manifest["source_digest"] = sha256_json(manifest["sources"])
    manifest.update(
        backend="local",
        endpoint=LOCAL_ENDPOINT,
        model=contract.LOGICAL_MODEL,
        workers=local["transport"]["workers"],
        min_interval_seconds=local["transport"]["min_interval_seconds"],
        local_backend=local,
    )
    manifest.pop("contract_digest")
    manifest["contract_digest"] = sha256_json(manifest)
    manifest_path = root / "manifest.json"
    replacement = root / ".manifest.local.json"
    write_json_atomic(replacement, manifest)
    os.replace(replacement, manifest_path)
    return {key: value for key, value in manifest.items() if key not in {"sources", "artifacts"}}


def _local_gpus(devices: list[str]) -> list[str]:
    gpus = []
    families = set()
    uuids = set()
    for device in devices:
        gpu = subprocess.run(  # noqa: S603 - fixed executable and frozen device selector
            [
                "nvidia-smi",
                "-i",
                device,
                "--query-gpu=index,name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        rows = list(csv.reader(gpu.splitlines()))
        require(len(rows) == 1 and len(rows[0]) == 4, "V07_LOCAL_GPU", "one GPU per device selector")
        _, name, uuid, _ = (field.strip() for field in rows[0])
        family = next((value for value in ("B200", "H100") if re.search(rf"\b{value}\b", name)), None)
        require(family is not None and bool(uuid), "V07_LOCAL_GPU", "NVIDIA B200 or H100 GPU")
        families.add(family)
        uuids.add(uuid)
        gpus.append(gpu)
    require(
        bool(gpus) and len(uuids) == len(devices) and len(families) == 1,
        "V07_LOCAL_GPU",
        "distinct GPUs from one supported family, one GPU per local replica",
    )
    return gpus


def _local_preflight(manifest: dict) -> dict:
    local = manifest["local_backend"]
    require(
        Path(sys.executable).resolve() == Path(local["python_executable"]).resolve(),
        "V07_LOCAL_ENVIRONMENT",
        "same Python environment used at prepare time",
    )
    require(
        _model_contract(Path(local["checkpoint"]["path"])) == local["checkpoint"],
        "V07_LOCAL_MODEL_CHANGED",
        "unchanged pinned local checkpoint",
    )
    runner = Path(local["runner"]["path"])
    require(
        _runner_contract(runner) == local["runner"],
        "V07_LOCAL_CONFIG_CHANGED",
        "unchanged local runner config",
    )
    tools_dir = Path(local["tools_dir"])
    current_path = os.environ.get("PATH", "")
    path_prefix = os.pathsep.join((str(tools_dir), local["python_bin"]))
    os.environ["PATH"] = path_prefix + (os.pathsep + current_path if current_path else "")
    require(
        all(shutil.which(name) for name in ("etcd", "nats-server", "ray")),
        "V07_LOCAL_TOOLS",
        "etcd, nats-server, and ray on PATH",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = local["cuda_visible_devices"]
    gpus = _local_gpus(local["devices"])
    require(
        len(gpus) == len(set(gpus)) == local["replicas"],
        "V07_LOCAL_GPU",
        "one distinct supported GPU per local replica",
    )
    runtime_root = Path(local["runtime_root"])
    for path in (runtime_root, Path(local["ray_temp_dir"]), Path(local["uv_cache_dir"]), Path(local["hf_home"])):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(os.access(path, os.W_OK), "V07_LOCAL_STORAGE", "writable local runtime storage")
    os.environ["UV_CACHE_DIR"] = local["uv_cache_dir"]
    os.environ["HF_HOME"] = local["hf_home"]
    return {"gpus": gpus, "replicas": local["replicas"], "runtime_root": str(runtime_root)}


@contextmanager
def _local_target(manifest: dict) -> Iterator[tuple[str, str, str]]:
    _local_preflight(manifest)
    local = manifest["local_backend"]
    from nemo_curator.eval.llm_judge import LocalJudgeRuntime

    runtime = LocalJudgeRuntime(
        local["runner"]["path"],
        model_overrides={"judge": local["checkpoint"]["path"]},
        served_model_overrides={"judge": local["logical_model"]},
        ray_temp_dir=local["ray_temp_dir"],
        num_gpus=local["num_gpus"],
    )
    dynamo_model = runtime.models[0].setdefault("dynamo_model", {})
    dynamo_model["num_replicas"] = local["replicas"]
    dynamo_model.setdefault("engine_kwargs", {})["tensor_parallel_size"] = local["tensor_parallel_size"]
    runtime_env = dynamo_model.setdefault("runtime_env", {})
    runtime_env.setdefault("env_vars", {})["UV_CACHE_DIR"] = local["uv_cache_dir"]
    started = False
    try:
        runtime.start()
        started = True
        yield runtime.endpoint, local["logical_model"], "unused"
    finally:
        if started:
            runtime.stop()


def run_local(root: Path, session: Path, *, smoke_only: bool = False) -> None:
    with state.exclusive(root):
        manifest = verify(root)
        release.require_execution_mode(manifest, smoke_only=smoke_only)
        require(_backend(manifest) == "local", "V07_BACKEND", "local run manifest")
        require(not (root / "complete.json").exists(), "V07_COMPLETE", "completed run is immutable")
        require(not (session / "started.json").exists(), "V07_SESSION", "fresh execution session required")
        write_json_atomic(
            session / "started.json",
            {
                "at_utc": state.now(),
                "pid": os.getpid(),
                "process_start": state.process_start(os.getpid()),
                "backend": "local",
            },
        )
        collector = transport.RateLimitCollector(root, session)
        stopped = threading.Event()

        def heartbeat() -> None:
            sequence = 0
            while not stopped.is_set():
                write_json_atomic(
                    session / "heartbeats" / f"{sequence:06d}.json",
                    {
                        "at_utc": state.now(),
                        "completed": len(list((root / "results").glob("*.json"))),
                        "population": manifest["population"],
                        "backend": "local",
                        **collector.heartbeat(),
                    },
                )
                sequence += 1
                stopped.wait(30)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            used = sum(bool(event.get("external_request")) for event in state.events(root / "transport_events.jsonl"))
            budget = manifest["max_external_attempts"] - used
            require(budget > 0, "V07_BUDGET", "bounded external request budget")
            profile = TransportProfile(
                min_interval_seconds=manifest["min_interval_seconds"],
                max_in_flight=manifest["workers"],
                max_attempts=2,
                retry_base_seconds=10,
                retry_cap_seconds=40,
                request_deadline_seconds=640,
                max_external_attempts=budget,
            )
            with (
                _local_target(manifest) as (endpoint, model, credential),
                transport.RateLimitRelay(
                    profile=profile,
                    logical_model=contract.LOGICAL_MODEL,
                    upstream_base_url=endpoint,
                    upstream_model=model,
                    upstream_api_key=credential,
                    timeout_seconds=600,
                    expected_generation_parameters=contract.GENERATION,
                ) as relay,
            ):
                relay.set_context(
                    RelayContext(
                        "v07-local-" + session.name,
                        0,
                        root / "transport_events.jsonl",
                    )
                )
                smoke = release.read(root / "smoke_panel.json")
                smoke_ids = {row["canonical_pair_id"] for row in smoke}
                if not (root / "smoke_complete.json").exists():
                    state.intact({path.stem for path in (root / "results").glob("*.json")} <= smoke_ids)
                    release.collect_rows(
                        root,
                        session,
                        smoke,
                        relay,
                        collector,
                        manifest["workers"],
                        "SMOKE",
                    )
                print(json.dumps({"event": "SMOKE_GATE_PASSED", **release.check_smoke(root, manifest)}), flush=True)
                if not smoke_only:
                    print(
                        json.dumps(
                            {
                                "event": "FULL20K_CONTINUATION",
                                "runtime_version": JUDGE_CONTRACT_VERSION,
                                "backend": "local",
                            }
                        ),
                        flush=True,
                    )
                    release.collect_rows(
                        root,
                        session,
                        release.read(root / "panel_index.json"),
                        relay,
                        collector,
                        manifest["workers"],
                        "FULL20K",
                    )
            if not smoke_only:
                print(json.dumps(audit(root)), flush=True)
            write_json_atomic(
                session / "exit.json",
                {
                    "at_utc": state.now(),
                    "status": "SMOKE_PASSED" if smoke_only else "COMPLETE",
                    "backend": "local",
                    "exit_code": 0,
                },
            )
        except BaseException as exc:
            write_json_atomic(
                session / "exit.json",
                {
                    "at_utc": state.now(),
                    "status": "STOPPED",
                    "backend": "local",
                    "error_code": state.error_code(exc),
                    "exit_code": 1,
                },
            )
            raise
        finally:
            stopped.set()
            thread.join(timeout=5)


def launch_local(root: Path, *, smoke_only: bool = False) -> dict:
    with state.exclusive(root):
        manifest = verify(root)
        release.require_execution_mode(manifest, smoke_only=smoke_only)
        require(_backend(manifest) == "local", "V07_BACKEND", "local launch manifest")
        require(
            not (root / "complete.json").exists() and not release.status(root)["running"],
            "V07_LAUNCH",
            "no duplicate active or completed run",
        )
        if (root / "smoke_complete.json").exists():
            require(
                release.read(root / "smoke_complete.json")["passed"],
                "V07_SMOKE_FAILED",
                "failed gate is immutable",
            )
        session = root / "recovery/sessions" / state.datetime.now(state.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        session.mkdir(parents=True)
        argv = [
            "nohup",
            sys.executable,
            "-u",
            "-m",
            "eval.dedup",
            "run",
            "--root",
            str(root),
            "--backend",
            "local",
            "--session",
            str(session),
        ]
        if smoke_only:
            argv.append("--smoke-only")
        with (session / "run.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(  # noqa: S603 - fixed executable and argv; no shell or credentials
                argv,
                cwd=HERE.parents[3],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        value = {
            "at_utc": state.now(),
            "pid": process.pid,
            "process_start": state.process_start(process.pid),
            "launcher": "nohup",
            "session": str(session),
            "log": str(session / "run.log"),
            "smoke_only": smoke_only,
            "backend": "local",
            "credential_persisted": False,
        }
        write_json_atomic(session / "launch.json", value)
        return value


def status(root: Path) -> dict:
    manifest = release.read(root / "manifest.json")
    return {**release.status(root), "backend": _backend(manifest)}


def audit(root: Path) -> dict:
    manifest = verify(root)
    return {**release.audit(root), "backend": _backend(manifest)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "run", "status", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Frozen 20K source bundle; defaults to CURATOR_V07_SOURCE_RUN.",
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--local-model-path",
        type=Path,
        help="Pinned local Qwen checkpoint; defaults to CURATOR_V07_LOCAL_MODEL_PATH.",
    )
    parser.add_argument("--local-runner-config", type=Path, default=DEFAULT_LOCAL_RUNNER_CONFIG)
    parser.add_argument("--local-runtime-root", type=Path)
    parser.add_argument(
        "--local-tools-dir",
        type=Path,
        help="Directory containing etcd and nats-server; defaults to CURATOR_V07_LOCAL_TOOLS_DIR.",
    )
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--local-device", help="Single-GPU compatibility alias, for example 0.")
    devices.add_argument("--local-devices", help="Comma-separated GPUs, one local replica per device.")
    parser.add_argument("--local-replicas", type=int, help="Must equal the number of local devices.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            source = release._configured_path(args.source_run, "CURATOR_V07_SOURCE_RUN")
            require(source is not None, "V07_SOURCE_RUN", "--source-run or CURATOR_V07_SOURCE_RUN")
            backend = args.backend or "hub"
            if backend == "local":
                model_path = release._configured_path(args.local_model_path, "CURATOR_V07_LOCAL_MODEL_PATH")
                tools_dir = release._configured_path(args.local_tools_dir, "CURATOR_V07_LOCAL_TOOLS_DIR")
                require(
                    model_path is not None,
                    "V07_LOCAL_MODEL",
                    "--local-model-path or CURATOR_V07_LOCAL_MODEL_PATH",
                )
                require(
                    tools_dir is not None,
                    "V07_LOCAL_TOOLS",
                    "--local-tools-dir or CURATOR_V07_LOCAL_TOOLS_DIR",
                )
                value = prepare_local(
                    root,
                    source_run=source,
                    model_path=model_path,
                    runner_config=args.local_runner_config,
                    runtime_root=args.local_runtime_root,
                    tools_dir=tools_dir,
                    devices=args.local_devices or args.local_device or "0",
                    replicas=args.local_replicas,
                    smoke_only=args.smoke_only,
                )
            else:
                value = release.prepare(root, source=source, smoke_only=args.smoke_only)
        else:
            manifest = release.read(root / "manifest.json") if args.command == "status" else verify(root)
            backend = _backend(manifest)
            if args.backend is not None:
                require(args.backend == backend, "V07_BACKEND_MISMATCH", "CLI backend matches frozen manifest")
            if args.command == "launch":
                if backend == "local":
                    value = launch_local(root, smoke_only=args.smoke_only)
                else:
                    require(args.env_file is not None, "V07_ARGS", "existing env file required for Hub")
                    value = release.launch(root, args.env_file.resolve(), smoke_only=args.smoke_only)
            elif args.command == "run":
                require(args.session is not None, "V07_ARGS", "execution session required")
                if backend == "local":
                    run_local(root, args.session.resolve(), smoke_only=args.smoke_only)
                else:
                    require(args.env_file is not None, "V07_ARGS", "existing env file required for Hub")
                    release.run(
                        root,
                        args.env_file.resolve(),
                        args.session.resolve(),
                        smoke_only=args.smoke_only,
                    )
                return 0
            elif args.command == "audit":
                with state.exclusive(root):
                    value = audit(root)
            else:
                value = status(root)
        print(json.dumps(value), flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - do not print potentially credential-bearing exceptions
        print(json.dumps({"error_code": state.error_code(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
