# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Document-owned binding instructions; unchanged V5 semantics and model settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.judging.context_coverage import CONTRACT
from eval.dedup.judging.context_runtime import CONFIG as BASE
from eval.dedup.judging.context_runtime import VERSION as BASE_VERSION
from eval.dedup.judging.context_runtime import ContextRuntime, build_context_builder, context_record
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.directional_witness import FEEDBACK_CONTRACT
from eval.dedup.validation import require

VERSION = "dedup-judge-hs-v0.6.2.32"
CONFIG = BASE.parent / "hs_v06232_directional_qwen_c16.yaml"


def load_directional_config(path: Path = CONFIG) -> dict:
    config = yaml.safe_load(path.read_text())
    require(
        set(config)
        == {
            "judge_version",
            "base_context_config",
            "output_contract",
            "feedback_contract",
            "binding_instructions_path",
            "binding_pair_path",
            "binding_rule",
        }
        and config["judge_version"] == VERSION
        and config["output_contract"] == CONTRACT
        and config["feedback_contract"] == FEEDBACK_CONTRACT
        and (path.parent / config["base_context_config"]).resolve() == BASE,
        "DIRECTIONAL_CONFIG",
        "versioned binding-only follow-up of the known V5 baseline required",
    )
    return config


def build_directional_builder(path: Path = CONFIG, **kwargs: Any) -> tuple[Any, list[Any]]:
    config = load_directional_config(path)
    builder, providers = build_context_builder(BASE, **kwargs)
    column = builder.get_column_config(COLUMN)
    require(column.system_prompt.count(BASE_VERSION) == 1, "DIRECTIONAL_ROLE_VERSION", "one role version required")
    system = column.system_prompt.replace(BASE_VERSION, VERSION) + "\n\n" + config["binding_rule"] + "\n\n"
    system += (path.parent / config["binding_instructions_path"]).read_text()
    pair = (
        column.prompt
        + "\n<directional_binding_rubric>"
        + json.dumps(config["binding_rule"])
        + "</directional_binding_rubric>\n"
    )
    pair += (path.parent / config["binding_pair_path"]).read_text()
    builder.delete_column(COLUMN).add_column(column.model_copy(update={"system_prompt": system, "prompt": pair}))
    return builder, providers


def directional_renderer(path: Path = CONFIG, *, stage: str) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    builder, _ = build_directional_builder(path, stage=stage, endpoint="http://127.0.0.1:1/v1")
    column = builder.get_column_config(COLUMN)
    renderer = RecordBasedPromptRenderer(create_response_recipe(column))

    def render(row: dict) -> list[dict]:
        record = context_record(row, stage)
        return [
            {"role": role, "content": renderer.render(prompt_template=template, record=record, prompt_type=kind)}
            for role, template, kind in (
                ("system", column.system_prompt, PromptType.SYSTEM_PROMPT),
                ("user", column.prompt, PromptType.USER_PROMPT),
            )
        ]

    return render


class DirectionalRuntime(ContextRuntime):
    def __init__(self, path: Path = CONFIG, **kwargs: Any):
        load_directional_config(path)
        self.directional_path = path
        super().__init__(BASE, **kwargs)

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

        builder, providers = build_directional_builder(
            self.directional_path,
            stage=self.stage,
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
                    f"directional_{self.stage}",
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
