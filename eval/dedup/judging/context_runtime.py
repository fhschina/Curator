# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fresh V5 runtime with paired context witnesses and whole-pair translation instructions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.judging.composite_format_runtime import CONFIG as BASE
from eval.dedup.judging.composite_format_runtime import VERSION as BASE_VERSION
from eval.dedup.judging.composite_format_runtime import FormatRuntime, build_format_builder
from eval.dedup.judging.context_coverage import CONTRACT, context_schema, critic_route
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.payload import assert_blind_payload
from eval.dedup.validation import require

VERSION = "dedup-judge-hs-v0.6.2.31"
CONFIG = BASE.parent / "hs_v06231_context_qwen_c16.yaml"
MAIN_PREFIX = "Validated V5 main claim: "
OLD_WITNESS_RULE = "UNCOVERED needs\none own unique decisive loss ID and its closest actual counterpart if one exists."


def encode_main_review(value: dict) -> str:
    return MAIN_PREFIX + json.dumps(value, ensure_ascii=False, sort_keys=True)


def context_record(row: dict, stage: str) -> dict:
    require(stage in {"main", "critic"}, "COMPOSITE_STAGE", "unknown stage")
    assert_blind_payload(row["payload"])
    record = {"payload": row["payload"], "repair_feedback": row.get("repair_feedback", "")}
    if stage == "main":
        require(not ({"proposed_main", "proposed_main_text"} & row.keys()), "COMPOSITE_MAIN_LEAK", "fresh main only")
    else:
        text = row.get("proposed_main_text")
        require(
            isinstance(text, str) and text.startswith(MAIN_PREFIX), "COMPOSITE_MAIN_MISSING", "fresh V5 claim required"
        )
        value = json.loads(text[len(MAIN_PREFIX) :])
        require(text == encode_main_review(value), "COMPOSITE_MAIN_BINDING", "canonical main text required")
        require(
            critic_route(value, row["payload"])[1] is None,
            "COMPOSITE_MAIN_NOT_POSITIVE",
            "only fresh positives routed",
        )
        record["proposed_main_text"] = text
    return record


def load_context_config(path: Path = CONFIG) -> dict:
    config = yaml.safe_load(path.read_text())
    require(
        set(config)
        == {
            "judge_version",
            "base_format_config",
            "output_contract",
            "context_instructions_path",
            "context_witness_rule",
            "translation_rule",
        }
        and config["judge_version"] == VERSION
        and config["output_contract"] == CONTRACT
        and (path.parent / config["base_format_config"]).resolve() == BASE,
        "CONTEXT_CONFIG",
        "versioned contextual contract and known baseline required",
    )
    return config


def build_context_builder(path: Path = CONFIG, **kwargs: Any) -> tuple[Any, list[Any]]:
    config = load_context_config(path)
    builder, providers = build_format_builder(BASE, **kwargs)
    column = builder.get_column_config(COLUMN)
    require(
        column.system_prompt.count(BASE_VERSION) == column.system_prompt.count(OLD_WITNESS_RULE) == 1,
        "CONTEXT_PROMPT_BOUNDARY",
        "replace the old source restriction, not a conflicting appended override",
    )
    system = column.system_prompt.replace(BASE_VERSION, VERSION).replace(
        OLD_WITNESS_RULE, config["context_witness_rule"]
    )
    system += (
        "\n\n" + config["translation_rule"] + "\n\n" + (path.parent / config["context_instructions_path"]).read_text()
    )
    rubric = {k: config[k] for k in ("context_witness_rule", "translation_rule")}
    updated = column.model_copy(
        update={
            "system_prompt": system,
            "prompt": column.prompt
            + "\n<context_contract_rubric>"
            + json.dumps(rubric, ensure_ascii=False)
            + "</context_contract_rubric>\n",
            "output_format": context_schema(),
        }
    )
    builder.delete_column(COLUMN).add_column(updated)
    return builder, providers


def context_renderer(path: Path = CONFIG, *, stage: str) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    builder, _ = build_context_builder(path, stage=stage, endpoint="http://127.0.0.1:1/v1")
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


class ContextRuntime(FormatRuntime):
    def __init__(self, path: Path = CONFIG, **kwargs: Any):
        load_context_config(path)
        self.context_path = path
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

        builder, providers = build_context_builder(
            self.context_path,
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
                    f"context_{self.stage}",
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
