# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Present the unchanged typed contract with side comparisons before record binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.judging.coverage_runtime import coverage_record
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.policy_runtime import BASE, RESOURCES, PolicyCoverageRuntime, build_policy_builder
from eval.dedup.judging.typed_coverage import typed_coverage_schema
from eval.dedup.judging.typed_coverage_runtime import load_typed_config
from eval.dedup.validation import require

CONTROL = RESOURCES / "hs_v06227_proposition_qwen_c16.yaml"


def normalize_schema_order(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: normalize_schema_order(v, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return sorted(value) if key == "required" else [normalize_schema_order(v) for v in value]
    return value


def sequencing_schema() -> dict:
    schema = typed_coverage_schema()
    orders = [
        (
            schema,
            [
                "contract_version",
                "input_status",
                "a_meaning_in_b",
                "b_meaning_in_a",
                "anchor_a_ids",
                "anchor_b_ids",
                "scope_explanation",
                "record_scope",
            ],
        )
    ]
    for branch in schema["$defs"].values():
        common = ["reviewed_unique_ids", "coverage_explanation", "status"]
        orders.append((branch, common + [k for k in branch["properties"] if k not in common]))
    for obj, order in orders:
        require(
            len(order) == len(obj["required"]) == len(set(order))
            and set(order) == set(obj["properties"]) == set(obj["required"]),
            "SEQUENCING_FIELDS",
            "reordering cannot change required fields",
        )
        obj["properties"] = {k: obj["properties"][k] for k in order}
        obj["required"] = order
    require(
        normalize_schema_order(schema) == normalize_schema_order(typed_coverage_schema()),
        "SEQUENCING_SCHEMA",
        "all schema constraints must remain identical",
    )
    return schema


def field_guide(schema: dict) -> str:
    return json.dumps(
        {
            "root_required_fields": schema["required"],
            "side_branches": {name: d["required"] for name, d in schema["$defs"].items()},
            "record_scope_options": schema["properties"]["record_scope"]["enum"],
        },
        ensure_ascii=False,
    )


def load_sequencing_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    require(
        config["response_field_order"] == "compare_then_bind"
        and (path.parent / config["base_policy_config"]).resolve() == CONTROL
        and set(config["rubric"]) == {"generation_sequence"},
        "SEQUENCING_CONFIG",
        "only the declared sequence changes on the complete-proposition typed task",
    )
    return config


def build_sequencing_builder(path: Path, **kwargs: Any) -> tuple[Any, list[Any]]:
    config = load_sequencing_config(path)
    builder, providers = build_policy_builder(CONTROL, **kwargs)
    column = builder.get_column_config(COLUMN)
    base, _ = load_typed_config(BASE)
    additions = [(path.parent / config[k]).read_text() for k in ("system_instruction_path", "pair_instruction_path")]
    require(
        all("{{" not in s and "{%" not in s for s in additions),
        "SEQUENCING_STATIC",
        "sequence instructions cannot read row variables",
    )
    guide, schema = field_guide(typed_coverage_schema()), sequencing_schema()
    rubric = json.dumps(base["rubric"], ensure_ascii=False)
    require(
        column.system_prompt.count(guide)
        == column.system_prompt.count("DIRECTION AND TYPED SIDE RESULTS")
        == column.prompt.count(rubric)
        == column.prompt.count("Input truncated:")
        == 1,
        "SEQUENCING_BOUNDARY",
        "replace only declared static guide, rubric and instruction boundaries",
    )
    system = column.system_prompt.replace(guide, field_guide(schema)).replace(
        "DIRECTION AND TYPED SIDE RESULTS", additions[0] + "\nDIRECTION AND TYPED SIDE RESULTS"
    )
    prompt = column.prompt.replace(rubric, json.dumps(base["rubric"] | config["rubric"], ensure_ascii=False)).replace(
        "Input truncated:", additions[1] + "\nInput truncated:"
    )
    updated = column.model_copy(update={"system_prompt": system, "prompt": prompt, "output_format": schema})
    builder.delete_column(COLUMN).add_column(updated)
    return builder, providers


def sequencing_renderer(path: Path) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    config, _ = load_typed_config(BASE)
    builder, _ = build_sequencing_builder(path, endpoint="http://127.0.0.1:1/v1")
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


class SequencingRuntime(PolicyCoverageRuntime):
    def __init__(self, path: Path, **kwargs: Any):
        load_sequencing_config(path)
        self.sequencing_path = path
        super().__init__(CONTROL, **kwargs)

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

        builder, providers = build_sequencing_builder(
            self.sequencing_path,
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
                    "coverage_sequenced",
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
