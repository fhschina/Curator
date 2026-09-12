# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static semantic-block ablation on the unchanged typed coverage task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eval.dedup.judging.coverage_runtime import coverage_record
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.typed_coverage_runtime import (
    TypedCoverageRuntime,
    build_typed_builder,
    load_typed_config,
    semantic_policy,
)
from eval.dedup.validation import require, sha256_file, sha256_json

RESOURCES = Path(__file__).resolve().parents[1] / "resources/local_ndd"
BASE = RESOURCES / "hs_v06226_typed_qwen_c16.yaml"
POLICIES = {
    "concise_boundaries": (
        RESOURCES / "hs_v06223_selection_system.jinja",
        "NONEMPTY RECORD ANCHORS",
        "SCOPE AND LOSS TYPES",
    ),
    "complete_propositions": (
        RESOURCES / "hs_v06225_proposition_system.jinja",
        "ESTABLISH THE ACTUAL RETAINED SUBJECT",
        "DIRECTIONS AND STRICT OUTPUT",
    ),
}


def extract_block(text: str, start: str, stop: str) -> str:
    require(
        text.count(start) == text.count(stop) == 1 and text.index(start) < text.index(stop),
        "POLICY_BLOCK_BOUNDARY",
        "one ordered pair of static block boundaries required",
    )
    block = text[text.index(start) : text.index(stop)]
    require("{{" not in block and "{%" not in block, "POLICY_BLOCK_TEMPLATE", "policy cannot contain row variables")
    return block


def policy_description(name: str) -> dict:
    require(name in POLICIES, "POLICY_NAME", "unknown semantic policy")
    path, start, stop = POLICIES[name]
    block = extract_block(path.read_text(), start, stop)
    return {
        "name": name,
        "source": str(path),
        "source_sha256": sha256_file(path),
        "start_inclusive": start,
        "stop_exclusive": stop,
        "block": block,
        "block_sha256": sha256_json(block),
    }


def load_policy_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    require(
        isinstance(config, dict)
        and set(config) == {"base_typed_config", "semantic_policy"}
        and (path.parent / config["base_typed_config"]).resolve() == BASE
        and config["semantic_policy"] in POLICIES,
        "POLICY_CONFIG",
        "only a declared semantic block may vary on the fixed typed task",
    )
    return config


def build_policy_builder(path: Path, **kwargs: Any) -> tuple[Any, list[Any]]:
    config = load_policy_config(path)
    builder, providers = build_typed_builder(BASE, **kwargs)
    column = builder.get_column_config(COLUMN)
    old, new = semantic_policy(BASE), policy_description(config["semantic_policy"])["block"]
    require(column.system_prompt.count(old) == 1, "POLICY_SYSTEM_BOUNDARY", "replace exactly the declared block")
    updated = column.model_copy(update={"system_prompt": column.system_prompt.replace(old, new)})
    builder.delete_column(COLUMN).add_column(updated)
    return builder, providers


def policy_renderer(path: Path) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    config, _ = load_typed_config(BASE)
    builder, _ = build_policy_builder(path, endpoint="http://127.0.0.1:1/v1")
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


class PolicyCoverageRuntime(TypedCoverageRuntime):
    def __init__(self, path: Path, **kwargs: Any):
        load_policy_config(path)
        self.policy_path = path
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

        builder, providers = build_policy_builder(
            self.policy_path,
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
                    "coverage_policy",
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
