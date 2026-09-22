# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import data_designer.config as dd
import pytest
import yaml
from jinja2 import Environment, StrictUndefined
from pydantic import ValidationError

from nemo_curator.eval.llm_judge import workflow as subject

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest_httpserver

EXAMPLE_DIR = Path(__file__).parents[3] / "tutorials" / "eval" / "llm_judge" / "cc_extract_example"


def _config_with_filters() -> tuple[dict[str, object], list[dict[str, object]]]:
    stages: list[dict[str, object]] = [
        {
            "name": "quality",
            "judges": [{"name": "quality_judge", "scores": [{"name": "quality"}]}],
            "filters": [{"judge": "quality_judge", "score": "quality", "operator": "gte", "value": 4}],
        },
        {
            "name": "safety",
            "judges": [{"name": "safety_judge", "scores": [{"name": "safe"}]}],
        },
    ]
    return ({"filters": [{"judge": "safety_judge", "score": "safe", "operator": "eq", "value": "yes"}]}, stages)


def test_load_yaml_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    config_path = tmp_path / "judge.yaml"
    config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must contain a mapping"):
        subject._load_yaml(config_path)


@pytest.mark.parametrize(
    ("filename", "models", "stages", "judges"),
    [
        ("text_extraction_qwen_judge.yaml", 1, 2, 2),
        ("text_extraction_qwen_gemma_judges.yaml", 2, 4, 4),
    ],
)
def test_example_configs_and_templates_are_valid(filename: str, models: int, stages: int, judges: int) -> None:
    config_path = EXAMPLE_DIR / filename
    config = subject._load_yaml(config_path)
    configured_stages = config["execution"]["stages"]

    assert len(config["models"]) == models
    assert len(configured_stages) == stages
    assert sum(len(stage["judges"]) for stage in configured_stages) == judges
    subject._validate_filter_references(config, configured_stages)

    aliases = {model["alias"] for model in config["models"]}
    environment = Environment(undefined=StrictUndefined)  # noqa: S701
    for stage in configured_stages:
        for judge in stage["judges"]:
            assert judge["model_alias"] in aliases
            prompt = subject._read_template(judge["prompt_path"], config_path=config_path)
            assert environment.from_string(prompt).render(
                raw_text=None, justext_text="clean text", trafilatura_text=None
            )
            system_prompt = subject._read_template(judge["system_prompt_path"], config_path=config_path)
            assert environment.from_string(system_prompt).render(
                raw_text=None, justext_text="clean text", trafilatura_text=None
            )


def test_place_filters_after_their_producing_stage() -> None:
    config, stages = _config_with_filters()

    placed = subject._place_filters(config, stages)

    assert [[item["judge"] for item in filters] for filters in placed] == [
        ["quality_judge"],
        ["safety_judge"],
    ]


@pytest.mark.parametrize(
    ("filter_config", "message"),
    [
        ({"judge": "missing", "score": "quality"}, "unknown judge output column"),
        ({"judge": "quality_judge", "score": "missing"}, "unknown score"),
    ],
)
def test_filter_validation_rejects_unknown_references(filter_config: dict[str, object], message: str) -> None:
    _config, stages = _config_with_filters()
    with pytest.raises(ValueError, match=message):
        subject._validate_filter_references({"filters": [filter_config]}, stages)


def test_filter_validation_rejects_stage_local_filter_for_later_judge() -> None:
    config, stages = _config_with_filters()
    stages[0]["filters"] = [{"judge": "safety_judge", "score": "safe", "operator": "eq", "value": "yes"}]

    with pytest.raises(ValueError, match="produced by a later stage"):
        subject._validate_filter_references(config, stages)


@pytest.mark.parametrize(
    ("judge_result", "score_name", "operator", "expected", "keep"),
    [
        ({"quality": {"score": 4}}, "quality", "eq", 4, True),
        ({"quality": {"score": 4}}, "quality", "ne", 4, False),
        ({"quality": {"score": 4}}, "quality", "gt", 3, True),
        ({"quality": {"score": 4}}, "quality", "gte", 4, True),
        ({"quality": {"score": 4}}, "quality", "lt", 5, True),
        ({"quality": {"score": 4}}, "quality", "lte", 4, True),
        ({"quality": {"score": "good"}}, "quality", "in", ["good", "bad"], True),
        ({"quality": {"score": "good"}}, "quality", "not_in", ["bad"], True),
        ({}, "quality", "eq", 4, False),
        (None, "quality", "eq", 4, False),
        (None, "quality", "ne", 4, False),
        ({"quality": {"score": "four"}}, "quality", "gt", 3, False),
    ],
)
def test_keep_judge_score(
    judge_result: object, score_name: str, operator: subject.FilterOperator, expected: object, keep: bool
) -> None:
    assert subject._keep_judge_score(judge_result, score_name=score_name, operator=operator, expected=expected) is keep


def test_start_inference_server_forwards_dynamo_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeModelConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.setdefault("models", []).append(kwargs)

    class FakeServerConfig:
        def __init__(self, **kwargs: object) -> None:
            captured["backend"] = kwargs

    class FakeServer:
        def __init__(self, **kwargs: object) -> None:
            captured["server"] = kwargs

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(subject, "DynamoVLLMModelConfig", FakeModelConfig)
    monkeypatch.setattr(subject, "DynamoServerConfig", FakeServerConfig)
    monkeypatch.setattr(subject, "InferenceServer", FakeServer)
    config = {"dynamo_server": {"subprocess_env": {"PYTHONPATH": "patches"}, "port": 9000}}
    models = [{"model": "local-weights", "served_model_name": "served-name", "dynamo_model": {"num_replicas": 2}}]

    server = subject._start_inference_server(config, models, config_path=tmp_path / "judge.yaml")

    assert isinstance(server, FakeServer)
    assert captured["models"] == [
        {"model_identifier": "local-weights", "model_name": "served-name", "num_replicas": 2}
    ]
    assert captured["backend"] == {
        "subprocess_env": {"PYTHONPATH": str((tmp_path / "patches").resolve())},
        "port": 9000,
    }
    assert captured["started"] is True


def test_build_pipeline_orders_reader_judges_filters_and_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject.DataDesignerStage, "_init_data_designer", lambda self: None)  # noqa: ARG005
    judge_stages = [
        (
            "quality",
            object(),
            [],
            {"env": "one"},
            1,
            [{"judge": "quality", "score": "score", "operator": "eq", "value": 1}],
        ),
        ("safety", object(), [], {"env": "two"}, 2, []),
    ]

    pipeline = subject.build_pipeline(
        input_path="input.jsonl",
        input_format="jsonl",
        output_path="output",
        output_format="jsonl",
        judge_stages=judge_stages,
        language_filter_stage=None,
        files_per_partition=4,
    )

    assert [stage.name for stage in pipeline.stages] == [
        "jsonl_reader",
        "ndd_quality",
        "judge_filter_quality_01",
        "ndd_safety",
        "jsonl_writer",
    ]
    assert isinstance(pipeline.stages[0], subject.JsonlReader)
    assert isinstance(pipeline.stages[2], subject.Filter)
    assert isinstance(pipeline.stages[-1], subject.JsonlWriter)
    ndd_stages = [stage for stage in pipeline.stages if stage.name.startswith("ndd_")]
    assert [stage.runtime_env for stage in ndd_stages] == [{"env": "one"}, {"env": "two"}]
    assert [stage.num_workers() for stage in ndd_stages] == [1, 2]


def test_build_language_filter_stage_returns_none_when_language_not_set() -> None:
    assert (
        subject._build_language_filter_stage(language=None, model_path=None, min_score=0.3, text_field="raw_text")
        is None
    )


@pytest.mark.parametrize(
    ("model_path", "min_score", "message"),
    [
        (None, 0.3, "fasttext_langid_model_path is required"),
        ("/path/to/model", -0.1, "min_langid_score must be between 0 and 1"),
        ("/path/to/model", 1.1, "min_langid_score must be between 0 and 1"),
    ],
)
def test_build_language_filter_stage_rejects_invalid_config(
    model_path: str | None, min_score: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        subject._build_language_filter_stage(
            language="en", model_path=model_path, min_score=min_score, text_field="raw_text"
        )


def test_build_language_filter_stage_builds_score_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

    class FakeFastTextLangId(DocumentFilter):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs

        def score_document(self, text: str) -> float:
            return 1.0

        def keep_document(self, scores: float) -> bool:
            return True

    monkeypatch.setattr("nemo_curator.stages.text.filters.fasttext.FastTextLangId", FakeFastTextLangId)

    stage = subject._build_language_filter_stage(
        language="en", model_path="/path/to/model", min_score=0.5, text_field="raw_text"
    )

    assert isinstance(stage, subject.ScoreFilter)
    assert stage.name == "fasttext_language_filter"
    assert isinstance(stage.filter_obj, list)
    assert len(stage.filter_obj) == 1
    assert isinstance(stage.filter_obj[0], FakeFastTextLangId)
    assert stage.filter_obj[0].kwargs == {"model_path": "/path/to/model", "min_langid_score": 0.5, "lang": "en"}
    assert stage.text_field == ["raw_text"]


def test_workflow_run_builds_pipeline_and_returns_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured, result = _run_workflow_with_fakes(monkeypatch, tmp_path, output_tasks=["task-a", "task-b"])

    assert captured["builder_judges"] == [["quality_judge"], ["safety_judge"]]
    stage_details = [
        (stage[0], stage[3], stage[4], [item["judge"] for item in stage[5]]) for stage in captured["judge_stages"]
    ]
    assert stage_details == [
        ("quality", {"env": "quality"}, 1, ["quality_judge"]),
        ("safety", {"env": "safety"}, 2, ["safety_judge"]),
    ]
    assert captured["build_pipeline_kwargs"]["language_filter_stage"] is None
    assert captured["server_stopped"] is True
    assert captured["run_kwargs"]["checkpoint_path"] == "checkpoint"
    assert result.workflow_name == "llm_judge"
    assert result.pipeline_tasks == {"llm_judge": ["task-a", "task-b"]}
    assert result.get_metadata("total_time") >= 0


def test_workflow_run_builds_language_filter_stage_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel_stage = object()
    monkeypatch.setattr(subject, "_build_language_filter_stage", lambda **kwargs: sentinel_stage)  # noqa: ARG005

    captured, _result = _run_workflow_with_fakes(
        monkeypatch,
        tmp_path,
        workflow_kwargs={
            "language": "en",
            "fasttext_langid_model_path": "/path/to/model",
        },
    )

    assert captured["build_pipeline_kwargs"]["language_filter_stage"] is sentinel_stage


def test_workflow_run_stops_inference_server_even_when_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="pipeline exploded"):
        _run_workflow_with_fakes(monkeypatch, tmp_path, pipeline_run_error=RuntimeError("pipeline exploded"))


def test_workflow_post_init_validates_filters_eagerly(tmp_path: Path) -> None:
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(
        """
models:
  - alias: judge
    model: model
execution:
  stages:
    - name: quality
      judges:
        - name: quality_judge
          scores:
            - name: quality
      filters:
        - judge: missing_judge
          score: quality
          operator: gte
          value: 4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown judge output column"):
        subject.LLMJudgeWorkflow(judge_config=config_path, input_path="input.jsonl", output_path="output")


def _run_workflow_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    output_tasks: list[object] | None = None,
    pipeline_run_error: Exception | None = None,
    workflow_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], subject.WorkflowRunResult]:
    config, stages = _config_with_filters()
    config.update(
        {
            "models": [{"alias": "judge", "model": "model"}],
            "execution": {"stages": stages},
        }
    )
    stages[0]["runtime_env"] = {"env": "quality"}
    stages[0]["num_workers"] = 1
    stages[1]["runtime_env"] = {"env": "safety"}
    stages[1]["num_workers"] = 2
    captured: dict[str, Any] = {"builder_judges": []}

    class FakeServer:
        endpoint = "http://judge"

        def stop(self) -> None:
            captured["server_stopped"] = True

    class FakePipeline:
        def run(self, **kwargs: object) -> list[object]:
            captured["run_kwargs"] = kwargs
            if pipeline_run_error is not None:
                raise pipeline_run_error
            return output_tasks if output_tasks is not None else []

    def fake_builder(*args: object, **kwargs: object) -> tuple[str, list[str]]:  # noqa: ARG001
        captured["builder_judges"].append([judge["name"] for judge in kwargs["judges"]])
        return "builder", ["provider"]

    def fake_build_pipeline(**kwargs: object) -> FakePipeline:
        captured["judge_stages"] = kwargs["judge_stages"]
        captured["build_pipeline_kwargs"] = kwargs
        return FakePipeline()

    monkeypatch.setattr(subject, "_load_yaml", lambda path: config)  # noqa: ARG005
    monkeypatch.setattr(subject, "_start_inference_server", lambda *args, **kwargs: FakeServer())  # noqa: ARG005
    monkeypatch.setattr(subject, "build_config_builder", fake_builder)
    monkeypatch.setattr(subject, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(subject, "RayDataExecutor", lambda: "executor")

    config_path = tmp_path / "judge.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    workflow = subject.LLMJudgeWorkflow(
        judge_config=config_path,
        input_path="input.jsonl",
        output_path="output",
        checkpoint_path="checkpoint",
        **(workflow_kwargs or {}),
    )
    try:
        result = workflow.run()
    finally:
        assert captured.get("server_stopped") is True
    return captured, result


def _conditional_judge(name: str = "quality_judge", **options: object) -> dict[str, object]:
    return {
        "name": name,
        "prompt_path": f"{name}.jinja",
        "scores": [{"name": "quality", "description": "Is the text readable?", "options": {0: "No", 1: "Yes"}}],
        **options,
    }


def _write_judge_config(tmp_path: Path, stages: list[dict[str, object]]) -> Path:
    for stage in stages:
        for judge in stage["judges"]:
            (tmp_path / judge["prompt_path"]).write_text(
                f"Judge {judge['name']} for {{{{ id }}}}: {{{{ text }}}}", encoding="utf-8"
            )
    config = {
        "models": [
            {
                "alias": "judge",
                "model": "mock-model",
                "skip_health_check": True,
                "inference_parameters": {"max_tokens": 256, "max_parallel_requests": 2},
            }
        ],
        "execution": {"stages": stages},
    }
    path = tmp_path / "judge.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("options", "expected_skip", "propagate_skip"),
    [
        ({}, None, True),
        ({"skip": None}, None, True),
        ({"skip": {"when": "{{ not should_run }}"}}, {"when": "{{ not should_run }}", "value": None}, True),
        (
            {"skip": {"when": "{{ not should_run }}", "value": 0}, "propagate_skip": False},
            {"when": "{{ not should_run }}", "value": 0},
            False,
        ),
        ({"propagate_skip": False}, None, False),
    ],
)
def test_build_config_builder_forwards_skip_options(
    tmp_path: Path, options: dict[str, object], expected_skip: dict[str, object] | None, propagate_skip: bool
) -> None:
    config_path = _write_judge_config(tmp_path, [{"name": "quality", "judges": [_conditional_judge(**options)]}])
    workflow = subject.LLMJudgeWorkflow(judge_config=config_path, input_path="unused", output_path="unused")

    builder = workflow._build_judge_stages(endpoint="http://unused/v1")[0][1]
    column = builder.get_column_config("quality_judge")

    assert isinstance(column, dd.LLMJudgeColumnConfig)
    assert (column.skip.model_dump() if column.skip is not None else None) == expected_skip
    assert column.propagate_skip is propagate_skip


@pytest.mark.parametrize(
    "options",
    [
        {"skip": {}},
        {"skip": False},
        {"skip": {"when": "not should_run"}},
        {"skip": {"when": "{{ not should_run"}},
        {"skip": {"when": "{{ not should_run }}", "value": {"quality": {"score": 0}}}},
        {"propagate_skip": "sometimes"},
    ],
)
def test_build_config_builder_rejects_invalid_skip_options(tmp_path: Path, options: dict[str, object]) -> None:
    config_path = _write_judge_config(tmp_path, [{"name": "quality", "judges": [_conditional_judge(**options)]}])
    workflow = subject.LLMJudgeWorkflow(judge_config=config_path, input_path="unused", output_path="unused")

    with pytest.raises(ValidationError):
        workflow._build_judge_stages(endpoint="http://unused/v1")


@pytest.fixture
def judge_endpoint(
    monkeypatch: pytest.MonkeyPatch, httpserver: pytest_httpserver.HTTPServer
) -> Iterator[pytest_httpserver.HTTPServer]:
    completion = {
        "id": "judge-completion",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '```json\n{"quality": {"score": 1, "reasoning": "Readable text."}}\n```',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(completion)
    server = SimpleNamespace(endpoint=httpserver.url_for("/v1"), stop=Mock())
    start_server = Mock(return_value=server)
    monkeypatch.setattr(subject, "_start_inference_server", start_server)
    yield httpserver
    start_server.assert_called_once()
    server.stop.assert_called_once()


def _run_conditional_workflow(tmp_path: Path, config_path: Path, flags: list[bool]) -> dict[str, dict]:
    seeds = [
        {"id": f"row_{index}", "text": f"Example text {index}.", "should_run": flag}
        for index, flag in enumerate(flags)
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in seeds), encoding="utf-8")
    workflow = subject.LLMJudgeWorkflow(
        judge_config=config_path, input_path=str(input_path), output_path=str(tmp_path / "output")
    )

    result = workflow.run()
    records = [
        json.loads(line)
        for task in result.pipeline_tasks["llm_judge"]
        for path in task.data
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(row["id"] for row in records) == [row["id"] for row in seeds]
    by_id = {row["id"]: row for row in records}
    for seed in seeds:
        assert {key: by_id[seed["id"]][key] for key in seed} == seed
    return by_id


@pytest.mark.parametrize(
    ("options", "flags", "expected_calls"),
    [
        ({"skip": {"when": "{{ not should_run }}"}}, [True, False, True, False], 2),
        ({"skip": {"when": "{{ not should_run }}"}}, [False] * 4, 0),
        ({"skip": {"when": "{{ not should_run }}"}}, [True] * 4, 4),
        ({}, [True, False, True, False], 4),
        ({"skip": None}, [True, False, True, False], 4),
    ],
    ids=["mixed", "all-skipped", "all-selected", "no-skip-config", "null-skip-config"],
)
def test_workflow_conditional_judge_preserves_rows(
    tmp_path: Path,
    judge_endpoint: pytest_httpserver.HTTPServer,
    options: dict[str, object],
    flags: list[bool],
    expected_calls: int,
) -> None:
    config_path = _write_judge_config(tmp_path, [{"name": "quality", "judges": [_conditional_judge(**options)]}])

    records = _run_conditional_workflow(tmp_path, config_path, flags)

    assert len(judge_endpoint.log) == expected_calls
    for record in records.values():
        if options.get("skip") is not None and not record["should_run"]:
            assert record["quality_judge"] is None
        else:
            assert record["quality_judge"]["quality"] == {"score": 1, "reasoning": "Readable text."}


@pytest.mark.parametrize("propagate_skip", [True, False])
def test_workflow_propagates_skips_within_one_stage(
    tmp_path: Path, judge_endpoint: pytest_httpserver.HTTPServer, propagate_skip: bool
) -> None:
    producer = _conditional_judge(skip={"when": "{{ not should_run }}"})
    consumer = _conditional_judge("followup")
    if not propagate_skip:
        consumer["propagate_skip"] = False
    config_path = _write_judge_config(tmp_path, [{"name": "quality", "judges": [producer, consumer]}])
    (tmp_path / "followup.jinja").write_text(
        "Review {{ id }}: {{ text }}. Earlier result: "
        '{{ quality_judge if quality_judge is not none else "Not evaluated" }}',
        encoding="utf-8",
    )

    records = _run_conditional_workflow(tmp_path, config_path, [True, False, True, False])

    assert len(judge_endpoint.log) == (4 if propagate_skip else 6)
    for record in records.values():
        if not record["should_run"]:
            assert record["quality_judge"] is None
        if propagate_skip and not record["should_run"]:
            assert record["followup"] is None
        else:
            assert record["followup"]["quality"]["score"] == 1


def test_workflow_gates_later_stage_with_explicit_null_condition(
    tmp_path: Path, judge_endpoint: pytest_httpserver.HTTPServer
) -> None:
    config_path = _write_judge_config(
        tmp_path,
        [
            {"name": "quality", "judges": [_conditional_judge(skip={"when": "{{ not should_run }}"})]},
            {
                "name": "followup",
                "judges": [_conditional_judge("followup", skip={"when": "{{ quality_judge is none }}"})],
            },
        ],
    )

    records = _run_conditional_workflow(tmp_path, config_path, [True, False, True, False])

    assert len(judge_endpoint.log) == 4
    for record in records.values():
        if record["should_run"]:
            assert record["followup"]["quality"]["score"] == 1
        else:
            assert record["quality_judge"] is None
            assert record["followup"] is None
