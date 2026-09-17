# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable local and external runtimes for config-driven LLM judges."""

from __future__ import annotations

import copy
import ensurepip
import importlib
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.eval.llm_judge.workflow import (
    DataFormat,
    _build_language_filter_stage,
    _get_data_designer_run_config,
    _get_num_workers,
    _load_yaml,
    _place_filters,
    _start_inference_server,
    _validate_filter_references,
    build_config_builder,
    build_pipeline,
)

if TYPE_CHECKING:
    from types import TracebackType

    from nemo_curator.core.serve import InferenceServer
    from nemo_curator.tasks import Task

_RAY_UNIX_SOCKET_MAX_BYTES = 107
_RAY_SESSION_NAME_PROBE = "session_9999-12-31_23-59-59_999999_9999999999"


def _ensure_pip_for_ray_uv_runtime() -> None:
    """Seed pip when a uv-created driver environment does not include it."""
    if importlib.util.find_spec("pip") is not None:
        return
    ensurepip.bootstrap(upgrade=True)
    importlib.invalidate_caches()
    if importlib.util.find_spec("pip") is None:
        msg = "Ray's uv runtime environment requires pip, but ensurepip did not install it."
        raise RuntimeError(msg)


def _validate_ray_temp_dir(path: str | Path) -> None:
    """Fail before startup when Ray's dashboard socket cannot fit in AF_UNIX."""
    socket_path = Path(path).resolve() / _RAY_SESSION_NAME_PROBE / "sockets" / "dash_MetricsHead"
    path_bytes = len(os.fsencode(socket_path))
    if path_bytes > _RAY_UNIX_SOCKET_MAX_BYTES:
        msg = (
            f"Ray temp path is too long for its dashboard Unix socket ({path_bytes} > "
            f"{_RAY_UNIX_SOCKET_MAX_BYTES} bytes): {socket_path}"
        )
        raise ValueError(msg)


class JudgePipelineRuntime:
    """Ray/Data Designer runtime targeting an OpenAI-compatible endpoint."""

    def __init__(  # noqa: PLR0913
        self,
        config_path: str | Path,
        *,
        endpoint: str | None,
        provider_api_key: str = "unused",  # pragma: allowlist secret
        execution_mode: Literal["single_stage", "multi_stage"] = "single_stage",
        model_overrides: dict[str, str] | None = None,
        served_model_overrides: dict[str, str] | None = None,
        ray_temp_dir: str = "/tmp/ray",  # noqa: S108
        num_cpus: int | None = None,
        num_gpus: int | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        _validate_ray_temp_dir(ray_temp_dir)
        self.execution_mode = execution_mode
        self.provider_api_key = provider_api_key
        self._endpoint = endpoint
        self.config = copy.deepcopy(_load_yaml(self.config_path))
        self.models: list[dict[str, object]] = self.config["models"]  # type: ignore[assignment]
        for model in self.models:
            alias = str(model["alias"])
            if model_overrides and alias in model_overrides:
                model["model"] = model_overrides[alias]
            if served_model_overrides and alias in served_model_overrides:
                model["served_model_name"] = served_model_overrides[alias]
        self.execution: dict[str, object] = self.config["execution"]  # type: ignore[assignment]
        self.data_designer_run_config = _get_data_designer_run_config(self.execution)
        self.configured_stages: list[dict[str, object]] = self.execution["stages"]  # type: ignore[assignment]
        _validate_filter_references(self.config, self.configured_stages)
        self.stage_filters = _place_filters(self.config, self.configured_stages)
        self.client = RayClient(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            include_dashboard=False,
            ray_temp_dir=ray_temp_dir,
        )

    @property
    def endpoint(self) -> str:
        if self._endpoint is None:
            msg = "JudgePipelineRuntime requires an endpoint before it can run a pipeline."
            raise RuntimeError(msg)
        return self._endpoint

    def set_endpoint(self, endpoint: str) -> None:
        if not endpoint:
            message = "judge endpoint must be non-empty"
            raise ValueError(message)
        self._endpoint = endpoint

    def start(self) -> Self:
        _ensure_pip_for_ray_uv_runtime()
        self.client.start()
        return self

    def stop(self) -> None:
        self.client.stop()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def run(  # noqa: PLR0913
        self,
        *,
        input_path: str,
        input_format: DataFormat,
        output_path: str,
        output_format: DataFormat = "jsonl",
        checkpoint_path: str | None = None,
        files_per_partition: int | None = None,
        language: str | None = None,
        fasttext_langid_model_path: str | None = None,
        min_langid_score: float = 0.3,
        language_text_field: str = "raw_text",
        inference_parameter_overrides: dict[str, dict[str, object]] | None = None,
    ) -> list[Task] | None:
        language_filter = _build_language_filter_stage(
            language=language,
            model_path=fasttext_langid_model_path,
            min_score=min_langid_score,
            text_field=language_text_field,
        )
        groups = (
            [("all_judges", [judge for stage in self.configured_stages for judge in stage["judges"]], [])]
            if self.execution_mode == "single_stage"
            else [
                (str(stage["name"]), stage["judges"], filters)
                for stage, filters in zip(self.configured_stages, self.stage_filters, strict=True)
            ]
        )
        judge_stages = []
        for name, judges, filters in groups:
            builder, providers = build_config_builder(
                self.config_path,
                endpoint=self.endpoint,
                models=self.models,
                judges=judges,
                provider_api_key=self.provider_api_key,
                inference_parameter_overrides=inference_parameter_overrides,
            )
            stage_config = (
                self.execution
                if self.execution_mode == "single_stage"
                else next(stage for stage in self.configured_stages if stage["name"] == name)
            )
            effective_filters = (
                [item for stage_filters in self.stage_filters for item in stage_filters]
                if self.execution_mode == "single_stage"
                else filters
            )
            judge_stages.append(
                (
                    name,
                    builder,
                    providers,
                    stage_config.get("runtime_env"),
                    _get_num_workers(stage_config, owner=f"Stage {name!r} num_workers"),
                    self.data_designer_run_config,
                    effective_filters,
                )
            )
        pipeline = build_pipeline(
            input_path=input_path,
            input_format=input_format,
            output_path=output_path,
            output_format=output_format,
            judge_stages=judge_stages,
            language_filter_stage=language_filter,
            files_per_partition=files_per_partition,
        )
        return pipeline.run(executor=RayDataExecutor(), checkpoint_path=checkpoint_path)


class ExternalJudgeRuntime(JudgePipelineRuntime):
    """Runtime for an already-running OpenAI-compatible endpoint."""

    def __init__(self, config_path: str | Path, *, endpoint: str, provider_api_key: str, **kwargs) -> None:
        super().__init__(
            config_path,
            endpoint=endpoint,
            provider_api_key=provider_api_key,
            num_gpus=0,
            **kwargs,
        )


class LocalJudgeRuntime(JudgePipelineRuntime):
    """Runtime that owns both Ray and the configured local Dynamo models."""

    def __init__(self, config_path: str | Path, **kwargs) -> None:
        super().__init__(config_path, endpoint=None, **kwargs)
        self.inference_server: InferenceServer | None = None

    def start(self) -> Self:
        super().start()
        try:
            self.inference_server = _start_inference_server(self.config, self.models, config_path=self.config_path)
        except Exception:
            super().stop()
            raise
        self.set_endpoint(self.inference_server.endpoint)
        return self

    def stop(self) -> None:
        if self.inference_server is not None:
            self.inference_server.stop()
            self.inference_server = None
        self._endpoint = None
        super().stop()
