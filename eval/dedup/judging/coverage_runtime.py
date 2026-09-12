# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native NDD structured-column runtime, separate from immutable score-based runners."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

import yaml

from eval.dedup.judging.coverage_witness import COLUMN, CONTRACT, coverage_schema
from eval.dedup.validation import require


def load_coverage_config(path: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(path.read_text())
    require(
        config["column_name"] == COLUMN and config["output_contract"] == CONTRACT,
        "COVERAGE_CONFIG_INVALID",
        "runner contract differs",
    )
    base = yaml.safe_load((path.parent / config["base_runner_config"]).read_text())
    return config, base


def coverage_record(row: dict, config: dict) -> dict:
    return {
        "payload": row["payload"],
        "repair_feedback": row.get("repair_feedback"),
        "coverage_rubric": json.dumps(config["rubric"], ensure_ascii=False),
    }


def build_coverage_builder(
    path: Path,
    *,
    endpoint: str,
    provider_api_key: str = "unused",
    model_name: str | None = None,
    inference_parameter_overrides: dict | None = None,
) -> tuple[Any, list[Any]]:
    import data_designer.config as dd

    from eval.llm_judge.run_llm_judge import build_config_builder

    config, base = load_coverage_config(path)
    models = deepcopy(base["models"])
    require(
        len(models) == 1 and models[0]["alias"] == "judge", "COVERAGE_MODEL_INVALID", "one judge model is required"
    )
    if model_name is not None:
        models[0]["served_model_name"] = model_name
    builder, providers = build_config_builder(
        path,
        endpoint=endpoint,
        models=models,
        judges=[],
        provider_api_key=provider_api_key,
        inference_parameter_overrides=inference_parameter_overrides,
    )
    prompt = (path.parent / config["prompt_path"]).read_text()
    require(
        prompt.count("{{ coverage_rubric }}") == 1,
        "COVERAGE_RUBRIC_MISSING",
        "pair template requires the synchronized rubric",
    )
    prompt = prompt.replace("{{ coverage_rubric }}", json.dumps(config["rubric"], ensure_ascii=False))
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name=COLUMN,
            model_alias="judge",
            system_prompt=(path.parent / config["system_prompt_path"]).read_text(),
            prompt=prompt,
            output_format=coverage_schema(),
            extract_reasoning_content=False,
            with_trace=dd.TraceType.ALL_MESSAGES,
        )
    )
    return builder, providers


def validate_native_coverage_row(row: dict) -> dict:
    """Reject outputs that NDD's default parser repaired by silently pruning extra fields."""
    from data_designer.config.utils.constants import TRACE_COLUMN_POSTFIX
    from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

    trace = row.get(COLUMN + TRACE_COLUMN_POSTFIX)
    require(isinstance(trace, list) and bool(trace), "COVERAGE_TRACE_MISSING", "raw response trace is required")
    assistant = next((message for message in reversed(trace) if message.get("role") == "assistant"), None)
    content = assistant.get("content") if isinstance(assistant, dict) else None
    if isinstance(content, list):
        require(
            bool(content)
            and all(
                isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
                for block in content
            ),
            "COVERAGE_TRACE_MISSING",
            "raw assistant content must contain only text blocks",
        )
        content = "".join(block["text"] for block in content)
    require(
        isinstance(content, str),
        "COVERAGE_TRACE_MISSING",
        "raw assistant text is required",
    )
    original = StructuredResponseRecipe(coverage_schema(), pruning=False).parse(content)
    value = row.get(COLUMN)
    value = json.loads(value) if isinstance(value, str) else value
    require(original == value, "COVERAGE_NATIVE_OUTPUT_CHANGED", "raw response must equal the native parsed column")
    return original


def coverage_renderer(path: Path) -> Callable[[dict], list[dict]]:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    config, _ = load_coverage_config(path)
    builder, _ = build_coverage_builder(path, endpoint="http://127.0.0.1:1/v1")
    column = builder.get_column_configs()[0]
    renderer = RecordBasedPromptRenderer(create_response_recipe(column))

    def render(row: dict) -> list[dict]:
        record = coverage_record(row, config)
        return [
            {"role": role, "content": renderer.render(prompt_template=template, record=record, prompt_type=kind)}
            for role, template, kind in (
                ("system", column.system_prompt, PromptType.SYSTEM_PROMPT),
                ("user", column.prompt, PromptType.USER_PROMPT),
            )
        ]

    return render


class CoverageWitnessRuntime:
    """Borrow existing Ray lifecycle and pipeline stages without invoking legacy judge columns."""

    def __init__(self, path: Path, *, endpoint: str, model_name: str, ray_temp_dir: str) -> None:
        from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

        self.path = path
        self.config, self.base = load_coverage_config(path)
        self.model_name = model_name
        self.lifecycle = ExternalJudgeRuntime(
            path.parent / self.config["base_runner_config"],
            endpoint=endpoint,
            provider_api_key="unused",
            served_model_overrides={"judge": model_name},
            ray_temp_dir=ray_temp_dir,
        )

    def __enter__(self) -> Self:
        self.lifecycle.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.lifecycle.stop()

    def run(
        self,
        *,
        input_path: str,
        input_format: str,
        output_path: str,
        output_format: str = "jsonl",
        checkpoint_path: str | None = None,
        files_per_partition: int | None = None,
        inference_parameter_overrides: dict | None = None,
    ) -> Any:
        from eval.llm_judge.run_llm_judge import _get_data_designer_run_config, _get_num_workers, build_pipeline
        from nemo_curator.backends.ray_data import RayDataExecutor

        builder, providers = build_coverage_builder(
            self.path,
            endpoint=self.lifecycle.endpoint,
            model_name=self.model_name,
            inference_parameter_overrides=inference_parameter_overrides,
        )
        execution = self.base["execution"]
        pipeline = build_pipeline(
            input_path=input_path,
            input_format=input_format,
            output_path=output_path,
            output_format=output_format,
            judge_stages=[
                (
                    "coverage_witness",
                    builder,
                    providers,
                    execution.get("runtime_env"),
                    _get_num_workers(execution, owner="execution.num_workers"),
                    _get_data_designer_run_config(execution),
                    [],
                )
            ],
            language_filter_stage=None,
            files_per_partition=files_per_partition,
        )
        return pipeline.run(executor=RayDataExecutor(), checkpoint_path=checkpoint_path)
