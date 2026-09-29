# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from nemo_curator.eval.llm_judge import workflow
from tutorials.eval.dedup.critics.dedup_adapter import DECISION_OPTIONS

if TYPE_CHECKING:
    from pathlib import Path

runner = import_module("tutorials.eval.dedup.6_run_critics")


def test_critic_configuration_is_validated_before_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [{"alias": "judge", "model": "test-model"}],
                "execution": {"stages": [{"name": "main", "judges": [{"name": "main", "scores": []}]}]},
            }
        ),
        encoding="utf-8",
    )
    kwargs = {"judge_config": config_path, "input_path": "input", "output_path": "output", "source_judge": "main"}
    with pytest.raises(ValueError, match="model alias"):
        runner.CoverageWorkflow(**kwargs, model_alias="missing")
    with pytest.raises(ValueError, match="source judge"):
        runner.CoverageWorkflow(**{**kwargs, "source_judge": "missing"})
    monkeypatch.setattr(runner, "_PROMPT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.CoverageWorkflow(**kwargs)


@pytest.mark.parametrize("all_skipped", [False, True])
@pytest.mark.usefixtures("shared_ray_client")
def test_workflow_runs_real_conditional_critic_pipeline(  # noqa: C901, PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pair: dict[str, Any],
    review: dict[str, Any],
    all_skipped: bool,
) -> None:
    requests: list[str] = []
    coverage_prompts: list[str] = []
    lifecycle: list[str] = []
    main = deepcopy(pair["pair_semantic_judgment"])
    negative = deepcopy(main)
    for key, value in {
        "a_can_replace_b": "no",
        "b_can_replace_a": "no",
        "relation_type": "related_non_duplicate",
        "material_difference": "major",
        "primary_material_difference": "other_material",
    }.items():
        negative[key]["score"] = value
    review["a_loss_span_id"] = "A001"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = request["messages"][-1]["content"]
            if isinstance(prompt, list):
                prompt = "\n".join(block.get("text", "") for block in prompt)
            is_main = "MAIN " in prompt
            requests.append("main" if is_main else "coverage")
            if not is_main:
                coverage_prompts.append(prompt)
            content = (negative if "skip" in prompt else main) if is_main else review
            response = {
                "id": "test-response",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "```json\n" + json.dumps(content) + "\n```"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class ModelServer:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"

        def stop(self) -> None:
            lifecycle.append("stop")

    def start_server(*_args: object, **_kwargs: object) -> ModelServer:
        lifecycle.append("start")
        return ModelServer()

    monkeypatch.setattr(workflow, "_start_inference_server", start_server)
    (tmp_path / "main.jinja").write_text("MAIN {{ pair_id }}", encoding="utf-8")
    records = [
        {**deepcopy(pair), "pair_id": f"{'skip' if all_skipped or index % 2 else 'run'}-{index}"} for index in range(4)
    ]
    for record in records:
        record["pair_semantic_judgment"] = deepcopy(negative if record["pair_id"].startswith("skip") else main)
    input_path = tmp_path / "pairs.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    config = {
        "models": [{"alias": "judge", "model": "test-model", "skip_health_check": True}],
        "execution": {
            "stages": [
                {
                    "name": "main",
                    "num_workers": 1,
                    "judges": [
                        {
                            "name": "pair_semantic_judgment",
                            "prompt_path": "main.jinja",
                            "model_alias": "judge",
                            "scores": [
                                {
                                    "name": name,
                                    "description": name,
                                    "options": dict.fromkeys(options, "An allowed value."),
                                }
                                for name, options in DECISION_OPTIONS.items()
                            ],
                        }
                    ],
                }
            ],
        },
    }
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    try:
        runner.CoverageWorkflow(
            judge_config=config_path, input_path=str(input_path), output_path=str(tmp_path / "output")
        ).run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert lifecycle == ["start", "stop"]
    assert requests.count("main") == 0
    assert requests.count("coverage") == (0 if all_skipped else 2)
    for prompt in coverage_prompts:
        assert "A001" in prompt
        assert "apples" in prompt
        assert "pears" in prompt
        assert "Original main reasoning" not in prompt
    rows = [
        json.loads(line) for path in (tmp_path / "output").glob("*.jsonl") for line in path.read_text().splitlines()
    ]
    assert len(rows) == len(records)
    by_id = {row["pair_id"]: row for row in rows}
    assert len(by_id) == len(records)
    for original in records:
        row = by_id[original["pair_id"]]
        for key in ("text_a", "text_b", "truncated", "pair_semantic_judgment"):
            assert row[key] == original[key]
        skipped = original["pair_id"].startswith("skip")
        assert row["coverage_should_run"] is not skipped
        assert row["coverage_action"] == ("SKIP" if skipped else "REJECT_B_REPLACES_A")
        if skipped:
            assert row["coverage_review"] is None
            assert row["final_decision"] == {key: value["score"] for key, value in negative.items()}
        else:
            assert row["coverage_review"] == review
            assert row["final_decision"]["a_can_replace_b"] == "yes"
            assert row["final_decision"]["b_can_replace_a"] == "no"
