# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit experiment registry and blind rendering; production defaults are untouched."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from eval.dedup.judging import retention_v4 as adapter
from eval.dedup.judging.payload import assert_blind_payload
from eval.dedup.judging.schema_v4 import JUDGE_SCHEMA_V4
from eval.dedup.validation import require, sha256_file, sha256_json

VERSION = "v0.6.2.33-exp2"
CONFIG = Path(__file__).resolve().parents[1] / "resources/local_ndd/hs_v06233_exp2.yaml"
EXPERIMENT_VERSIONS = {VERSION: CONFIG}


def specification(version: str = VERSION) -> dict:
    require(version in EXPERIMENT_VERSIONS, "RETENTION_V4_VERSION", "unknown isolated experiment version")
    path = EXPERIMENT_VERSIONS[version]
    config = yaml.safe_load(path.read_text())
    require(
        config["version"] == version
        and config["status"] == "EXPERIMENT_NOT_RELEASE"
        and config["output_schema"] == JUDGE_SCHEMA_V4
        and config["model_response_contract"] == adapter.CONTRACT,
        "RETENTION_V4_REGISTRY",
        "config, adapter and output schema must agree",
    )
    require(
        set(config["rubric"]) == set(adapter.response_schema()["required"]),
        "RETENTION_V4_RUBRIC",
        "pair rubric must describe every model response field",
    )
    paths = {
        str(path),
        str(Path(__file__).resolve()),
        str(Path(adapter.__file__).resolve()),
        str(Path(__file__).with_name("schema_v4.py")),
    }
    paths.update(str(path.parent / v) for k, v in config.items() if k.endswith(("_path", "_config")))
    sources = {p: sha256_file(p) for p in sorted(paths)}
    return {
        "version": version,
        "config": config,
        "sources": sources,
        "contract_digest": sha256_json({"config": config, "sources": sources}),
        "release_eligible": False,
    }


def messages(payload: dict, *, stage: str = "main", version: str = VERSION) -> list[dict]:
    require(stage in {"main", "critic"}, "RETENTION_V4_STAGE", "explicit main or critic stage required")
    require(
        set(payload)
        <= {"document_a", "document_b", "long_document_evidence", "semantic_diff_evidence", "payload_schema_version"},
        "RETENTION_V4_BLIND_PAYLOAD",
        "only frozen evidence fields may enter the model renderer",
    )
    assert_blind_payload(payload)
    spec = specification(version)
    config = spec["config"]
    # Plain-text prompts must preserve quotation characters and Unicode unchanged.
    env = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701
    values = {
        "payload": payload,
        "rubric": json.dumps(config["rubric"], ensure_ascii=False),
        "retention_policy": (CONFIG.parent / config["policy_path"]).read_text(),
        "response_schema": json.dumps(adapter.response_schema(), ensure_ascii=False),
    }
    return [
        {"role": role, "content": env.from_string((CONFIG.parent / config[key]).read_text()).render(**values)}
        for role, key in (("system", f"{stage}_system_path"), ("user", "pair_path"))
    ]
