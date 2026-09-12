# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Format-only composite follow-up; frozen semantic policy and response schema are reused."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.judging.composite_coverage import CONTRACT
from eval.dedup.judging.composite_runtime import CONFIG as BASE
from eval.dedup.judging.composite_runtime import VERSION as BASE_VERSION
from eval.dedup.judging.composite_runtime import CompositeRuntime, build_composite_builder, composite_record
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.validation import require

VERSION = "dedup-judge-hs-v0.6.2.30"
CONFIG = BASE.parent / "hs_v06230_format_qwen_c16.yaml"


def load_format_config(path: Path = CONFIG) -> dict:
    config = yaml.safe_load(path.read_text())
    require(
        set(config)
        == {
            "judge_version",
            "base_composite_config",
            "output_contract",
            "format_instructions_path",
            "response_encoding",
        }
        and config["judge_version"] == VERSION
        and config["output_contract"] == CONTRACT
        and (path.parent / config["base_composite_config"]).resolve() == BASE
        and isinstance(config["response_encoding"], str),
        "COMPOSITE_FORMAT_CONFIG",
        "only versioned response-encoding instructions may vary",
    )
    return config


def format_blocks(path: Path = CONFIG) -> tuple[str, str]:
    config = load_format_config(path)
    return (
        "\n\n" + (path.parent / config["format_instructions_path"]).read_text().rstrip(),
        "\n\n<response_encoding>"
        + json.dumps(config["response_encoding"], ensure_ascii=False)
        + "</response_encoding>\n",
    )


def build_format_builder(path: Path = CONFIG, **kwargs: Any) -> tuple[Any, list[Any]]:
    system_block, pair_block = format_blocks(path)
    builder, providers = build_composite_builder(BASE, **kwargs)
    column = builder.get_column_config(COLUMN)
    require(column.system_prompt.count(BASE_VERSION) == 1, "COMPOSITE_FORMAT_VERSION", "one role version required")
    updated = column.model_copy(
        update={
            "system_prompt": column.system_prompt.rstrip().replace(BASE_VERSION, VERSION) + system_block,
            "prompt": column.prompt + pair_block,
        }
    )
    builder.delete_column(COLUMN).add_column(updated)
    return builder, providers


def format_renderer(path: Path = CONFIG, *, stage: str) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    builder, _ = build_format_builder(path, stage=stage, endpoint="http://127.0.0.1:1/v1")
    column = builder.get_column_config(COLUMN)
    renderer = RecordBasedPromptRenderer(create_response_recipe(column))

    def render(row: dict) -> list[dict]:
        record = composite_record(row, stage)
        return [
            {"role": role, "content": renderer.render(prompt_template=template, record=record, prompt_type=kind)}
            for role, template, kind in (
                ("system", column.system_prompt, PromptType.SYSTEM_PROMPT),
                ("user", column.prompt, PromptType.USER_PROMPT),
            )
        ]

    return render


class FormatRuntime(CompositeRuntime):
    def __init__(self, path: Path = CONFIG, **kwargs: Any):
        load_format_config(path)
        self.format_path = path
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

        builder, providers = build_format_builder(
            self.format_path,
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
                    f"composite_format_{self.stage}",
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
