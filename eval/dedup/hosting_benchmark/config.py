# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Strict configuration for the inference-hosting benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.dedup.validation import require, sha256_json


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    seed: int
    concurrencies: tuple[int, ...]
    repeats: int
    pairs_per_block: int
    warmup_pairs: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    logical_model: str
    local_model_id: str
    local_model_revision: str
    local_model_path: Path
    hub_base_url: str
    hub_model: str
    api_key_env: str
    max_model_len: int
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class PayloadConfig:
    tokenizer_model_id: str
    tokenizer_revision: str
    tokenizer_cache_root: Path
    max_visible_tokens: int
    window_tokens: int
    window_overlap_tokens: int


@dataclass(frozen=True, slots=True)
class HostingBenchmarkConfig:
    source_path: Path
    source_run_root: Path
    runs_root: Path
    ray_temp_dir: Path
    checkpoint_root: Path
    runner_config: Path
    model: ModelConfig
    payload: PayloadConfig
    workload: WorkloadConfig
    temperature: float
    top_p: float
    timeout_seconds: float
    max_retries: int

    @property
    def digest(self) -> str:
        return sha256_json(_jsonable(json.loads(self.source_path.read_text(encoding="utf-8"))))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _path(value: object, *, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _exact(value: dict[str, Any], expected: set[str], *, context: str) -> None:
    require(
        set(value) == expected,
        "INVALID_HOSTING_BENCHMARK_CONFIG",
        "hosting benchmark config fields differ",
        context=context,
        expected_fields=sorted(expected),
        actual_fields=sorted(value),
    )


def load_config(path: str | Path) -> HostingBenchmarkConfig:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    require(isinstance(raw, dict), "INVALID_HOSTING_BENCHMARK_CONFIG", "benchmark config must be an object")
    _exact(
        raw,
        {
            "schema_version",
            "source_run_root",
            "runs_root",
            "ray_temp_dir",
            "checkpoint_root",
            "runner_config",
            "model",
            "payload",
            "workload",
            "generation",
        },
        context="root",
    )
    require(raw["schema_version"] == 1, "INVALID_HOSTING_BENCHMARK_CONFIG", "unsupported schema version")
    base = source.parent
    model = raw["model"]
    payload = raw["payload"]
    workload = raw["workload"]
    generation = raw["generation"]
    for value, fields, context in (
        (
            model,
            {
                "logical_model",
                "local_model_id",
                "local_model_revision",
                "local_model_path",
                "hub_base_url",
                "hub_model",
                "api_key_env",
                "max_model_len",
                "max_output_tokens",
            },
            "model",
        ),
        (
            payload,
            {
                "tokenizer_model_id",
                "tokenizer_revision",
                "tokenizer_cache_root",
                "max_visible_tokens",
                "window_tokens",
                "window_overlap_tokens",
            },
            "payload",
        ),
        (workload, {"seed", "concurrencies", "repeats", "pairs_per_block", "warmup_pairs"}, "workload"),
        (generation, {"temperature", "top_p", "timeout_seconds", "max_retries"}, "generation"),
    ):
        require(isinstance(value, dict), "INVALID_HOSTING_BENCHMARK_CONFIG", "section must be an object")
        _exact(value, fields, context=context)
    config = HostingBenchmarkConfig(
        source_path=source,
        source_run_root=_path(raw["source_run_root"], base=base),
        runs_root=_path(raw["runs_root"], base=base),
        ray_temp_dir=_path(raw["ray_temp_dir"], base=base),
        checkpoint_root=_path(raw["checkpoint_root"], base=base),
        runner_config=_path(raw["runner_config"], base=base),
        model=ModelConfig(
            logical_model=str(model["logical_model"]),
            local_model_id=str(model["local_model_id"]),
            local_model_revision=str(model["local_model_revision"]),
            local_model_path=_path(model["local_model_path"], base=base),
            hub_base_url=str(model["hub_base_url"]),
            hub_model=str(model["hub_model"]),
            api_key_env=str(model["api_key_env"]),
            max_model_len=int(model["max_model_len"]),
            max_output_tokens=int(model["max_output_tokens"]),
        ),
        payload=PayloadConfig(
            tokenizer_model_id=str(payload["tokenizer_model_id"]),
            tokenizer_revision=str(payload["tokenizer_revision"]),
            tokenizer_cache_root=_path(payload["tokenizer_cache_root"], base=base),
            max_visible_tokens=int(payload["max_visible_tokens"]),
            window_tokens=int(payload["window_tokens"]),
            window_overlap_tokens=int(payload["window_overlap_tokens"]),
        ),
        workload=WorkloadConfig(
            seed=int(workload["seed"]),
            concurrencies=tuple(int(value) for value in workload["concurrencies"]),
            repeats=int(workload["repeats"]),
            pairs_per_block=int(workload["pairs_per_block"]),
            warmup_pairs=int(workload["warmup_pairs"]),
        ),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        timeout_seconds=float(generation["timeout_seconds"]),
        max_retries=int(generation["max_retries"]),
    )
    require(config.model.max_model_len == 32_768, "INVALID_HOSTING_BENCHMARK_CONFIG", "context is frozen")
    require(config.model.max_output_tokens == 4_096, "INVALID_HOSTING_BENCHMARK_CONFIG", "output cap is frozen")
    require(
        config.model.logical_model == "Qwen/Qwen3.8-27B-FP8"
        and config.model.local_model_id == "Qwen/Qwen3.8-27B-FP8"
        and config.model.local_model_revision == "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
        and config.payload.tokenizer_model_id == config.model.local_model_id
        and config.payload.tokenizer_revision == config.model.local_model_revision,
        "INVALID_HOSTING_BENCHMARK_CONFIG",
        "local model or tokenizer differs from the frozen Qwen checkpoint",
    )
    require(
        config.model.local_model_path == Path("/raid/hfang/hf_cache/Qwen3.8-27B-FP8")
        and config.source_run_root == Path("/raid/hfang/dedup_eval_runs/dedup-full-20260813T220949Z-d4c37bb483/v0_run")
        and config.runs_root == Path("/raid/hfang/ihb/runs")
        and config.ray_temp_dir == Path("/raid/hfang/ihb/ray")
        and config.checkpoint_root == Path("/raid/hfang/ihb/checkpoints")
        and config.runner_config.name == "sarah_minhash_qwen.yaml",
        "INVALID_HOSTING_BENCHMARK_CONFIG",
        "benchmark paths differ from the frozen run protocol",
    )
    require(config.temperature == 0 and config.top_p == 1, "INVALID_HOSTING_BENCHMARK_CONFIG", "sampling differs")
    require(config.timeout_seconds == 600, "INVALID_HOSTING_BENCHMARK_CONFIG", "timeout differs")
    require(config.max_retries == 2, "INVALID_HOSTING_BENCHMARK_CONFIG", "retry budget differs")
    require(
        config.model.hub_base_url == "https://inference-api.nvidia.com/v1"
        and config.model.hub_model == "nvidia/qwen/qwen3.8-27b"
        and config.model.api_key_env == "NVIDIA_API_KEY",
        "INVALID_HOSTING_BENCHMARK_CONFIG",
        "Inference Hub target differs from the frozen target",
    )
    require(
        config.payload.max_visible_tokens == 20_000
        and config.payload.window_tokens == 4_096
        and config.payload.window_overlap_tokens == 512,
        "INVALID_HOSTING_BENCHMARK_CONFIG",
        "payload construction differs from the frozen contract",
    )
    require(
        config.workload.concurrencies == (1, 2, 4, 8)
        and config.workload.seed == 26_082_701
        and config.workload.repeats == 3
        and config.workload.pairs_per_block == 500
        and config.workload.warmup_pairs == 100,
        "INVALID_HOSTING_BENCHMARK_CONFIG",
        "formal workload shape differs",
    )
    require(
        config.payload.window_overlap_tokens < config.payload.window_tokens,
        "INVALID_HOSTING_BENCHMARK_CONFIG",
        "window overlap must be smaller",
    )
    require(
        config.payload.max_visible_tokens + config.model.max_output_tokens < config.model.max_model_len,
        "INVALID_HOSTING_BENCHMARK_CONFIG",
        "payload budget leaves no room for the judge contract",
    )
    return config
