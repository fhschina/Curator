# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native structured selection requests with schema-generated field guidance."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.judging.coverage_runtime import CoverageWitnessRuntime, coverage_record
from eval.dedup.judging.coverage_selection import COLUMN, CONTRACT, selection_schema
from eval.dedup.validation import require


def load_selection_config(path: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(path.read_text())
    require(
        config["column_name"] == COLUMN and config["output_contract"] == CONTRACT,
        "SELECTION_CONFIG_INVALID",
        "wrong native contract",
    )
    return config, yaml.safe_load((path.parent / config["base_runner_config"]).read_text())


def field_guide() -> str:
    properties = selection_schema()["properties"]
    return json.dumps(
        {
            "root_required_fields": list(properties),
            "each_side_required_fields": properties["a_meaning_in_b"]["required"],
            "record_scope_options": properties["record_scope"]["enum"],
        },
        ensure_ascii=False,
    )


def build_selection_builder(
    path: Path, *, endpoint: str, model_name: str | None = None, inference_parameter_overrides: dict | None = None
) -> tuple[Any, list[Any]]:
    import data_designer.config as dd

    from eval.llm_judge.run_llm_judge import build_config_builder

    config, base = load_selection_config(path)
    models = deepcopy(base["models"])
    require(len(models) == 1 and models[0]["alias"] == "judge", "SELECTION_MODEL_INVALID", "one judge model required")
    if model_name is not None:
        models[0]["served_model_name"] = model_name
    builder, providers = build_config_builder(
        path,
        endpoint=endpoint,
        models=models,
        judges=[],
        provider_api_key="unused",
        inference_parameter_overrides=inference_parameter_overrides,
    )
    prompt = (path.parent / config["prompt_path"]).read_text()
    system = (path.parent / config["system_prompt_path"]).read_text()
    require(
        prompt.count("{{ coverage_rubric }}") == 1 and system.count("{{ coverage_field_guide }}") == 1,
        "SELECTION_GUIDE_MISSING",
        "rubric and schema guide must be embedded statically",
    )
    prompt = prompt.replace("{{ coverage_rubric }}", json.dumps(config["rubric"], ensure_ascii=False))
    system = system.replace("{{ coverage_field_guide }}", field_guide())
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name=COLUMN,
            model_alias="judge",
            system_prompt=system,
            prompt=prompt,
            output_format=selection_schema(),
            extract_reasoning_content=False,
            with_trace=dd.TraceType.ALL_MESSAGES,
        )
    )
    return builder, providers


def selection_renderer(path: Path) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    config, _ = load_selection_config(path)
    builder, _ = build_selection_builder(path, endpoint="http://127.0.0.1:1/v1")
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


class SelectionRuntime(CoverageWitnessRuntime):
    def __init__(self, path: Path, *, endpoint: str, model_name: str, ray_temp_dir: str):
        from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

        self.path = path
        self.config, self.base = load_selection_config(path)
        self.model_name = model_name
        self.lifecycle = ExternalJudgeRuntime(
            path.parent / self.config["base_runner_config"],
            endpoint=endpoint,
            provider_api_key="unused",
            served_model_overrides={"judge": model_name},
            ray_temp_dir=ray_temp_dir,
        )

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

        builder, providers = build_selection_builder(
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
                    "coverage_selection",
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
