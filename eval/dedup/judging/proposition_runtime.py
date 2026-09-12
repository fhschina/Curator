# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Evidence-first presentation of the unchanged V2 semantic output contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.dedup.judging.coverage_runtime import coverage_record
from eval.dedup.judging.coverage_selection import COLUMN, selection_schema
from eval.dedup.judging.selection_runtime import (
    SelectionRuntime,
    build_selection_builder,
    field_guide,
    load_selection_config,
)
from eval.dedup.validation import require


def proposition_schema() -> dict:
    schema = selection_schema()
    root_order = (
        "contract_version",
        "input_status",
        "scope_explanation",
        "anchor_a_ids",
        "anchor_b_ids",
        "record_scope",
        "a_meaning_in_b",
        "b_meaning_in_a",
    )
    side_order = (
        "reviewed_unique_ids",
        "coverage_explanation",
        "coverage_counterpart_ids",
        "source_span_id",
        "counterpart_span_id",
        "retention_consequence",
        "uncovered_type",
        "counterpart_relation",
        "coverage_mode",
        "status",
    )
    for obj, order in [(schema, root_order)] + [
        (schema["properties"][field], side_order) for field in ("a_meaning_in_b", "b_meaning_in_a")
    ]:
        require(
            set(order) == set(obj["properties"]), "PROPOSITION_FIELD_DRIFT", "reordering cannot add or drop fields"
        )
        obj["properties"] = {k: obj["properties"][k] for k in order}
        obj["required"] = list(order)
    return schema


def build_proposition_builder(path: Path, **kwargs: Any) -> tuple[Any, list[Any]]:
    config, _ = load_selection_config(path)
    require(
        config.get("response_field_order") == "evidence_then_decision", "PROPOSITION_ORDER", "explicit order required"
    )
    builder, providers = build_selection_builder(path, **kwargs)
    column = builder.get_column_config(COLUMN)
    schema = proposition_schema()
    props = schema["properties"]
    guide = json.dumps(
        {
            "root_required_fields": list(props),
            "each_side_required_fields": props["a_meaning_in_b"]["required"],
            "record_scope_options": props["record_scope"]["enum"],
        },
        ensure_ascii=False,
    )
    require(
        column.system_prompt.count(field_guide()) == 1, "PROPOSITION_GUIDE", "replace only the generated schema guide"
    )
    updated = column.model_copy(
        update={"output_format": schema, "system_prompt": column.system_prompt.replace(field_guide(), guide)}
    )
    builder.delete_column(COLUMN).add_column(updated)
    return builder, providers


def proposition_renderer(path: Path) -> Any:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    config, _ = load_selection_config(path)
    builder, _ = build_proposition_builder(path, endpoint="http://127.0.0.1:1/v1")
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


class PropositionRuntime(SelectionRuntime):
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

        builder, providers = build_proposition_builder(
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
                    "coverage_proposition",
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
