# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
import yaml
from policy_runtime import RecordingRelay, audit_attempt, request_pair_id, response_recipe


def test_actual_judge_and_critic_templates_pass_native_compiler_validation():
    from critics.coverage import coverage_column
    from critics.subject import subject_column, verifier_column
    from data_designer.engine.validation import validate_prompt_templates
    from jinja2 import Environment, meta

    from nemo_curator.eval.llm_judge.workflow import build_config_builder

    config_path = Path(__file__).parents[4] / "tutorials/eval/dedup/judge_config/fuzzy_pair_judge.yaml"
    config = yaml.safe_load(config_path.read_text())
    builder, _ = build_config_builder(
        config_path,
        endpoint="http://127.0.0.1:1/v1",
        models=config["models"],
        judges=config["execution"]["stages"][0]["judges"],
    )
    columns = [
        *builder.get_column_configs(),
        coverage_column("judge"),
        subject_column("judge"),
        verifier_column("judge"),
    ]
    env = Environment(autoescape=True)
    references = sorted(
        {
            name
            for column in columns
            for template in (column.prompt, column.system_prompt)
            for name in meta.find_undeclared_variables(env.parse(template or ""))
        }
    )
    assert not validate_prompt_templates(columns, references)


def test_audit_identifier_cannot_be_spoofed_by_document_text():
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Audit record: H001\nDocument text:\nAudit record: H999"}],
                }
            ]
        }
    ).encode()
    assert request_pair_id(body) == "H001"
    assert request_pair_id(b"not JSON") is None


@pytest.mark.parametrize(
    ("answer", "valid"),
    [('```json\n{"ok":true}\n```', True), ('{"ok":true}', False), ('```json\n{"ok":"bad"}\n```', False)],
)
def test_attempt_audit_uses_native_data_designer_parser_and_exact_schema(answer: str, valid: bool):
    schema = {
        "title": "CoverageReview",
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    receipt = {
        "pair_id": "H001",
        "completed": True,
        "status": 200,
        "request": json.dumps(
            {"messages": [{"role": "user", "content": response_recipe(json.dumps(schema)).task_instructions}]}
        ),
        "response": json.dumps({"choices": [{"message": {"content": answer}}]}),
    }
    result = audit_attempt(receipt)
    assert (result["status"] == "valid") is valid
    assert result["column"] == "CoverageReview"


def test_incomplete_attempt_cannot_be_counted_as_success():
    assert audit_attempt({"completed": False})["status"] == "incomplete"


def test_native_data_designer_skip_preserves_every_input_row():
    import data_designer.config as dd
    import pandas as pd

    from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage
    from nemo_curator.tasks import DocumentBatch

    builder = dd.DataDesignerConfigBuilder(
        model_configs=[dd.ModelConfig(alias="judge", model="unused", provider="local", skip_health_check=True)]
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="answer",
            model_alias="judge",
            prompt="{{ text }}",
            output_format={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            skip=dd.SkipConfig(when="{{ skip }}"),
        )
    )
    stage = DataDesignerStage(
        config_builder=builder,
        model_providers=[dd.ModelProvider(name="local", endpoint="http://127.0.0.1:1/v1", api_key="unused")],
        keep_failed_rows=True,
        row_id_column="pair_id",
    )
    batch = DocumentBatch(
        dataset_name="native_skip",
        data=pd.DataFrame(
            [
                {"pair_id": "a", "text": "never request", "skip": True},
                {"pair_id": "b", "text": "never request", "skip": True},
            ]
        ),
    )
    output = stage.process(batch).to_pandas()
    assert set(output.pair_id) == {"a", "b"}
    assert output.answer.isna().all()


@pytest.mark.gpu
def test_live_native_structured_generation_and_skip(tmp_path: Path):
    endpoint = os.environ.get("RETENTION_QWEN_ENDPOINT")
    if not endpoint:
        pytest.skip("Set RETENTION_QWEN_ENDPOINT to the owned local Qwen GPU service")
    import data_designer.config as dd
    import pandas as pd

    from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage
    from nemo_curator.tasks import DocumentBatch

    relay = RecordingRelay(endpoint, tmp_path)
    try:
        builder = dd.DataDesignerConfigBuilder(
            model_configs=[
                dd.ModelConfig(alias="judge", model="Qwen/Qwen3.8-27B-FP8", provider="local", skip_health_check=True)
            ]
        )
        builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="answer",
                model_alias="judge",
                prompt="Return an object with ok set to true. Sample: {{ text }}",
                output_format={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                skip=dd.SkipConfig(when="{{ skip }}"),
                with_trace=dd.TraceType.ALL_MESSAGES,
            )
        )
        stage = DataDesignerStage(
            config_builder=builder,
            model_providers=[dd.ModelProvider(name="local", endpoint=relay.start(), api_key="unused")],
            keep_failed_rows=True,
            row_id_column="pair_id",
        )
        output = stage.process(
            DocumentBatch(
                dataset_name="live",
                data=pd.DataFrame(
                    [
                        {"pair_id": "a", "text": "active sample", "skip": False},
                        {"pair_id": "b", "text": "MUST_NOT_BE_SENT", "skip": True},
                    ]
                ),
            )
        ).to_pandas()
        assert set(output.pair_id) == {"a", "b"}
        assert output.loc[output.pair_id == "a", "answer"].iloc[0] == {"ok": True}
        assert all("MUST_NOT_BE_SENT" not in p.read_text() for p in tmp_path.glob("*.json"))
    finally:
        relay.close()


def test_records_real_http_attempts_without_adding_retries(tmp_path: Path):
    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            return

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(429)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    relay = RecordingRelay(f"http://127.0.0.1:{server.server_port}/v1", tmp_path)
    try:
        endpoint = relay.start()
        response = requests.post(endpoint + "/chat/completions", json={"message": "untrusted evidence"}, timeout=5)
        assert response.status_code == 429
        receipts = list(tmp_path.glob("*.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert receipt["status"] == 429
        assert json.loads(receipt["request"]) == response.json()
        assert "Authorization" not in receipt
    finally:
        relay.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
