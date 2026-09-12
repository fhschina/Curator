# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native typed coverage using the same model setup and frozen proposition policy text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.judging.coverage_runtime import coverage_record
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.proposition_runtime import build_proposition_builder
from eval.dedup.judging.selection_runtime import SelectionRuntime
from eval.dedup.judging.typed_coverage import CONTRACT, typed_coverage_schema
from eval.dedup.validation import require


def load_typed_config(path: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(path.read_text())
    require(
        config["column_name"] == COLUMN and config["output_contract"] == CONTRACT,
        "TYPED_CONFIG",
        "wrong typed contract",
    )
    base_selection = yaml.safe_load((path.parent / config["base_selection_config"]).read_text())
    require(
        config["base_runner_config"] == base_selection["base_runner_config"],
        "TYPED_BASE",
        "model and lifecycle must share the same base",
    )
    return config, yaml.safe_load((path.parent / config["base_runner_config"]).read_text())


def semantic_policy(path: Path) -> str:
    config, _ = load_typed_config(path)
    old = yaml.safe_load((path.parent / config["base_selection_config"]).read_text())
    text = (path.parent / old["system_prompt_path"]).read_text()
    start, stop = "ESTABLISH THE ACTUAL RETAINED SUBJECT", "DIRECTIONS AND STRICT OUTPUT"
    require(text.count(start) == text.count(stop) == 1, "TYPED_POLICY", "frozen semantic block boundaries required")
    return start + text.split(start, 1)[1].split(stop, 1)[0]


def build_typed_builder(path: Path, **kwargs: Any) -> tuple[Any, list[Any]]:
    config, _ = load_typed_config(path)
    builder, providers = build_proposition_builder(path.parent / config["base_selection_config"], **kwargs)
    column = builder.get_column_config(COLUMN)
    schema = typed_coverage_schema()
    guide = json.dumps(
        {
            "root_required_fields": schema["required"],
            "side_branches": {name: d["required"] for name, d in schema["$defs"].items()},
            "record_scope_options": schema["properties"]["record_scope"]["enum"],
        },
        ensure_ascii=False,
    )
    system = (path.parent / config["system_prompt_path"]).read_text()
    prompt = (path.parent / config["prompt_path"]).read_text()
    require(
        system.count("{{ semantic_policy }}")
        == system.count("{{ typed_field_guide }}")
        == prompt.count("{{ coverage_rubric }}")
        == 1,
        "TYPED_GUIDE",
        "policy, field guide and rubric must be static configuration inputs",
    )
    system = system.replace("{{ semantic_policy }}", semantic_policy(path)).replace("{{ typed_field_guide }}", guide)
    prompt = prompt.replace("{{ coverage_rubric }}", json.dumps(config["rubric"], ensure_ascii=False))
    updated = column.model_copy(update={"output_format": schema, "system_prompt": system, "prompt": prompt})
    builder.delete_column(COLUMN).add_column(updated)
    return builder, providers


def typed_renderer(path: Path) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    config, _ = load_typed_config(path)
    builder, _ = build_typed_builder(path, endpoint="http://127.0.0.1:1/v1")
    column = builder.get_column_config(COLUMN)
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


class TypedCoverageRuntime(SelectionRuntime):
    def __init__(self, path: Path, *, endpoint: str, model_name: str, ray_temp_dir: str):
        from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

        self.path = path
        self.config, self.base = load_typed_config(path)
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

        builder, providers = build_typed_builder(
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
                    "coverage_typed",
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
