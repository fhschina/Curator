# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in fresh main and critic stages sharing the composite policy and strict V4 schema."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.judging.composite_coverage import CONTRACT, POLICY, composite_schema, critic_route
from eval.dedup.judging.coverage_runtime import CoverageWitnessRuntime
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.payload import assert_blind_payload
from eval.dedup.validation import require

VERSION = "dedup-judge-hs-v0.6.2.29"
CONFIG = Path(__file__).resolve().parents[1] / "resources/local_ndd/hs_v06229_composite_qwen_c16.yaml"
MAIN_PREFIX = "Validated V4 main claim: "


def encode_main_review(value: dict) -> str:
    return MAIN_PREFIX + json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_composite_config(path: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(path.read_text())
    require(
        config["judge_version"] == VERSION
        and config["output_contract"] == CONTRACT
        and config["policy_version"] == POLICY
        and config["public_contract"] == "dedup-judge-output-v3"
        and config["column_name"] == COLUMN
        and set(config["system_prompt_paths"]) == {"main", "critic"},
        "COMPOSITE_CONFIG",
        "versioned main/critic policy and response contract required",
    )
    return config, yaml.safe_load((path.parent / config["base_runner_config"]).read_text())


def composite_record(row: dict, stage: str) -> dict:
    require(stage in {"main", "critic"}, "COMPOSITE_STAGE", "unknown stage")
    assert_blind_payload(row["payload"])
    record = {"payload": row["payload"], "repair_feedback": row.get("repair_feedback", "")}
    if stage == "main":
        require(
            not ({"proposed_main", "proposed_main_text"} & row.keys()),
            "COMPOSITE_MAIN_LEAK",
            "fresh main cannot observe prior predictions",
        )
    else:
        text = row.get("proposed_main_text")
        require(
            isinstance(text, str) and text.startswith(MAIN_PREFIX),
            "COMPOSITE_MAIN_MISSING",
            "critic requires its fresh main certificate",
        )
        proposed = json.loads(text[len(MAIN_PREFIX) :])
        require(
            text == encode_main_review(proposed), "COMPOSITE_MAIN_ENCODING", "canonical prefixed main text required"
        )
        _, owned = critic_route(proposed, row["payload"])
        require(owned is None, "COMPOSITE_ROUTING", "only new positive main decisions enter the critic")
        record["proposed_main_text"] = text
    return record


def build_composite_builder(
    path: Path,
    *,
    stage: str,
    endpoint: str,
    model_name: str | None = None,
    provider_api_key: str = "unused",
    inference_parameter_overrides: dict | None = None,
) -> tuple[Any, list[Any]]:
    import data_designer.config as dd

    from eval.llm_judge.run_llm_judge import build_config_builder

    require(stage in {"main", "critic"}, "COMPOSITE_STAGE", "unknown stage")
    config, base = load_composite_config(path)
    models = deepcopy(base["models"])
    require(len(models) == 1 and models[0]["alias"] == "judge", "COMPOSITE_MODEL", "single shared model required")
    if model_name is not None:
        models[0]["served_model_name"] = model_name
    models[0]["inference_parameters"] = deepcopy(config["inference_parameters"])
    builder, providers = build_config_builder(
        path,
        endpoint=endpoint,
        models=models,
        judges=[],
        provider_api_key=provider_api_key,
        inference_parameter_overrides=inference_parameter_overrides,
    )
    schema = composite_schema()
    guide = json.dumps(
        {"root_fields": schema["required"], "side_branches": {k: v["required"] for k, v in schema["$defs"].items()}},
        ensure_ascii=False,
    )
    system = (path.parent / config["system_prompt_paths"][stage]).read_text()
    prompt = (path.parent / config["prompt_path"]).read_text()
    require(
        system.count("{{ semantic_policy }}")
        == system.count("{{ composite_field_guide }}")
        == prompt.count("{{ composite_rubric }}")
        == prompt.count("{{ stage_review }}")
        == 1,
        "COMPOSITE_PROMPT_BOUNDARY",
        "policy/field guide/rubric/role must be static version inputs",
    )
    system = system.replace("{{ semantic_policy }}", (path.parent / config["policy_path"]).read_text()).replace(
        "{{ composite_field_guide }}", guide
    )
    prompt = prompt.replace("{{ composite_rubric }}", json.dumps(config["rubric"], ensure_ascii=False)).replace(
        "{{ stage_review }}",
        "<proposed_main_claim>{{ proposed_main_text }}</proposed_main_claim>"
        if stage == "critic"
        else "Make a fresh main decision from the original texts above.",
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name=COLUMN,
            model_alias="judge",
            system_prompt=system,
            prompt=prompt,
            output_format=schema,
            extract_reasoning_content=False,
            with_trace=dd.TraceType.ALL_MESSAGES,
        )
    )
    return builder, providers


def composite_renderer(path: Path = CONFIG, *, stage: str) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    builder, _ = build_composite_builder(path, stage=stage, endpoint="http://127.0.0.1:1/v1")
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


class CompositeRuntime(CoverageWitnessRuntime):
    def __init__(self, path: Path = CONFIG, *, stage: str, endpoint: str, model_name: str, ray_temp_dir: str):
        from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

        require(stage in {"main", "critic"}, "COMPOSITE_STAGE", "unknown stage")
        self.path, self.stage = path, stage
        self.config, self.base = load_composite_config(path)
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

        builder, providers = build_composite_builder(
            self.path,
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
                    f"composite_{self.stage}",
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
