# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fresh execution of the immutable exp1 component composition."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import cache
from pathlib import Path

import requests
import yaml

from eval.dedup.analysis import critic_scope_experiment as common
from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.judging import critic_retention_v4 as coverage
from eval.dedup.judging import critic_schema_transport as transport
from eval.dedup.judging import critic_subject_binding as subject
from eval.dedup.judging import critic_subject_proof_verifier as verifier
from eval.dedup.judging import critic_subject_scope as scope
from eval.dedup.judging import local_ndd
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.schema_v3 import unresolved_judge_output_v3, validate_judge_output_v3
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic

VERSION = "v0.6.2.33-exp1"
RESOURCES = Path(__file__).resolve().parents[1] / "resources/local_ndd"
MAIN_CONFIG = RESOURCES / "hs_v06212_qwen_c64.yaml"
GENERATION = common.GENERATION
LOGICAL_MODEL = "Qwen/Qwen3.8-27B-FP8"


@cache
def main_toolchain() -> tuple:
    from data_designer.engine.column_generators.utils.prompt_renderer import create_response_recipe

    from eval.llm_judge.run_llm_judge import build_config_builder

    config = yaml.safe_load(MAIN_CONFIG.read_text())
    builder, _ = build_config_builder(
        MAIN_CONFIG,
        endpoint="http://127.0.0.1:1/v1",
        models=config["models"],
        judges=config["execution"]["stages"][0]["judges"][:1],
    )
    column = builder.get_column_configs()[0]
    return column, create_response_recipe(column)


def main_messages(payload: dict, feedback: dict | None = None) -> list[dict]:
    from data_designer.engine.column_generators.utils.prompt_renderer import PromptType, RecordBasedPromptRenderer
    from data_designer.engine.models.utils import ChatMessage

    column, recipe = main_toolchain()
    renderer = RecordBasedPromptRenderer(recipe)
    # The historical dataframe converted missing feedback to NaN before rendering.
    record = {"payload": payload, "repair_feedback": float("nan") if feedback is None else feedback}
    return [
        ChatMessage(
            role=role,
            content=renderer.render(prompt_template=template, record=record, prompt_type=kind),
        ).to_dict()
        for role, template, kind in (
            ("system", column.system_prompt, PromptType.SYSTEM_PROMPT),
            ("user", column.prompt, PromptType.USER_PROMPT),
        )
    ]


def body(messages: list[dict], schema: dict | None = None) -> dict:
    result = {"model": LOGICAL_MODEL, **deepcopy(GENERATION), "messages": messages}
    if schema is not None:
        result["response_format"] = transport.response_format(schema, "without_unique_items")
    return result


def coverage_renderer() -> Callable[[dict], list[dict]]:
    return selected.renderer(
        {
            "adapter": coverage.__name__,
            "config": str(RESOURCES / "v06212_retention_v4.yaml"),
            "system": str(RESOURCES / "v06212_retention_v4_system.jinja"),
            "pair": str(RESOURCES / "v06212_retention_v4_pair.jinja"),
        }
    )


def collect(root: Path, key: str, request: dict, endpoint: str) -> dict:
    require(not (root / "requests" / (key + ".json")).exists(), "EXP1_REUSE", "fresh receipt required")
    digest = sha256_json(request)
    write_json_atomic(root / "requests" / (key + ".json"), {"body": request, "request_sha256": digest})
    receipt = {"request_sha256": digest, "status": "FAILURE"}
    try:
        response = requests.post(endpoint + "/chat/completions", json=request, timeout=660)
        receipt["http_status"] = response.status_code
        require(response.status_code == 200, "EXP1_HTTP", "request failed; retain the failure")
        receipt.update(status="RECEIVED", raw_response=response.json())
    except DedupEvaluationError as exc:
        receipt["error_code"] = exc.issue.code
    except Exception as exc:  # noqa: BLE001 - exception messages can contain credentials
        receipt["error_code"] = type(exc).__name__
    write_json_atomic(root / "responses" / (key + ".json"), receipt)
    return receipt


def generate_main(messages: list[dict], call: Callable[[dict], dict]) -> dict:
    from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
    from data_designer.engine.model_provider import ModelProviderRegistry
    from data_designer.engine.models.clients.adapters.openai_compatible import translate_openai_compatible_messages
    from data_designer.engine.models.clients.parsing import parse_chat_completion_response
    from data_designer.engine.models.clients.types import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        TransportKwargs,
    )
    from data_designer.engine.models.facade import ModelFacade

    class RecordedClient:
        def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
            transport_args = TransportKwargs.from_request(request)
            rendered = translate_openai_compatible_messages(
                request.messages, provider_name="local-judge", model_name=LOGICAL_MODEL
            )
            return parse_chat_completion_response(
                call({"model": request.model, "messages": rendered, **transport_args.body})
            )

    model = ModelFacade(
        ModelConfig(
            alias="judge",
            model=LOGICAL_MODEL,
            provider="local-judge",
            skip_health_check=True,
            inference_parameters=ChatCompletionInferenceParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=4096,
                timeout=600,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
        ),
        ModelProviderRegistry(
            providers=[ModelProvider(name="local-judge", endpoint="http://127.0.0.1:1/v1", api_key="unused")]
        ),
        client=RecordedClient(),
    )
    text = lambda message: "".join(block["text"] for block in message["content"])  # noqa: E731
    value, _ = model.generate(
        prompt=text(messages[1]),
        system_prompt=text(messages[0]),
        parser=main_toolchain()[1].parse,
        max_correction_steps=2,
        max_conversation_restarts=0,
    )
    return value.model_dump()


def execute_case(root: Path, row: dict, endpoint: str, frozen: dict, render_coverage: Callable) -> dict:
    key, payload = row["canonical_pair_id"], row["payload"]
    result = {
        "canonical_pair_id": key,
        "review_id": row["review_id"],
        "version": VERSION,
        "status": "VALID",
        "components": {},
        "stages": [],
        "main_attempts": [],
    }

    def call(stage: str, request: dict) -> dict:
        require(all(request[k] == v for k, v in GENERATION.items()), "EXP1_GENERATION", "frozen generation only")
        receipt = collect(root, key + "-" + stage, request, endpoint)
        result["stages"].append(
            {"stage": stage, "response_sha256": sha256_file(root / "responses" / (key + "-" + stage + ".json"))}
        )
        require(receipt["status"] == "RECEIVED", receipt.get("error_code", "EXP1_RESPONSE"), "fresh response required")
        return receipt["raw_response"]

    def review(stage: str, messages: list[dict], schema: dict) -> dict:
        response = call(stage, body(messages, schema))
        choice = response["choices"][0]
        require(
            choice["finish_reason"] == "stop", "EXP1_STAGE_FINISH", "historical critic requires a complete response"
        )
        return common.strict_json(choice["message"]["content"])

    try:
        require(frozen["request_sha256"] == sha256_json(frozen["body"]), "EXP1_MAIN_BINDING", "frozen initial request")
        feedback = None
        for outer in range(1, 4):
            inner = 0
            messages = frozen["body"]["messages"] if outer == 1 else main_messages(payload, feedback)

            def main_call(request: dict, *, attempt: int = outer) -> dict:
                nonlocal inner
                inner += 1
                if attempt == inner == 1:
                    require(request == frozen["body"], "EXP1_MAIN_INITIAL", "exact frozen initial body")
                return call(f"main-{attempt:02d}-{inner:02d}", request)

            try:
                raw = generate_main(messages, main_call)
                public = common.critic.main_decision(raw, payload)
                result["raw_main"] = raw
                result["main_attempts"].append({"outer": outer, "requests": inner, "status": "VALID"})
                break
            except Exception as exc:
                feedback = local_ndd._safe_retry_feedback(exc)
                result["main_attempts"].append(
                    {"outer": outer, "requests": inner, "status": "FAILURE", "feedback": feedback}
                )
                if outer == 3:
                    raise
        result["components"]["main"] = deepcopy(public)
        if common.critic.route(public, payload) == "REVIEW_POSITIVE_DIRECTIONS_ONLY":
            raw = review("coverage", render_coverage(payload), coverage.response_schema())
            public, rule = coverage.apply_review(public, payload, raw)
            result["coverage_review"], result["coverage_rule"] = raw, rule
        result["components"]["coverage"] = deepcopy(public)
        if scope.route(public, payload) == "REVIEW_BILATERAL_SUBJECTS":
            raw = review(
                "subject",
                subject.messages(payload, (RESOURCES / "v06212_subject_binding_v1_system.txt").read_text()),
                subject.response_schema(payload),
            )
            result["subject_proposal"] = raw
            bound = {**payload, "subject_proposal": raw}
            if verifier.route(public, bound) == "VERIFY_FIXED_SUBJECT_VETO":
                verification = review(
                    "verifier",
                    verifier.messages(bound, (RESOURCES / "v06212_subject_proof_verifier_v4_system.txt").read_text()),
                    verifier.response_schema(),
                )
                public, rule = verifier.apply_review(public, bound, verification)
                result["verifier_review"], result["verifier_rule"] = verification, rule
        validate_judge_output_v3(public)
        validate_evidence_offsets(public, payload)
        result["public"] = public
    except Exception as exc:  # noqa: BLE001 - failures remain in the full evaluation denominator
        result.update(
            status="ENGINEERING_FAILURE",
            error_code=exc.issue.code if isinstance(exc, DedupEvaluationError) else type(exc).__name__,
            public=unresolved_judge_output_v3(),
        )
    write_json_atomic(root / "results" / (key + ".json"), result)
    return result
