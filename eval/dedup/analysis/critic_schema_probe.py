# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small actual-endpoint test for schema parameters, not a Judge benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

from eval.dedup.analysis import critic_retention_experiment as base
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

MODES = ("json_object", "guided_json", "json_schema_standard", "json_schema_flat")
GENERATION = {**base.previous.GENERATION, "max_tokens": 128}


def request_body(mode: str, sentinel: str) -> dict:
    require(mode in MODES, "SCHEMA_PROBE_MODE", "unknown protocol encoding")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["probe"],
        "properties": {"probe": {"type": "string", "enum": [sentinel]}},
    }
    body = {
        "model": "Qwen/Qwen3.8-27B-FP8",
        **GENERATION,
        "messages": [
            {
                "role": "user",
                "content": 'Return exactly this JSON instance, with no extra fields: {"probe":"PROMPT_VALUE"}',
            }
        ],
    }
    if mode == "guided_json":
        body["guided_json"] = schema
    elif mode == "json_schema_standard":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "schema_capability_probe", "strict": True, "schema": schema},
        }
    elif mode == "json_schema_flat":
        body["response_format"] = {"type": "json_schema", "schema": json.dumps(schema)}
    else:
        body["response_format"] = {"type": "json_object"}
    return body


def classify(mode: str, sentinel: str, raw: dict) -> str:
    try:
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            return "INCOMPLETE_OUTPUT"
        value = base.previous.strict_json(choice["message"]["content"])
    except Exception:  # noqa: BLE001 - probe reports parse failures, never repairs them
        return "INVALID_OUTPUT"
    if mode == "json_object":
        return "BASELINE_PROMPT_FOLLOWED" if value == {"probe": "PROMPT_VALUE"} else "BASELINE_UNEXPECTED"
    return "SCHEMA_CONSTRAINT_OBSERVED" if value == {"probe": sentinel} else "CONSTRAINT_NOT_OBSERVED"


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    root = root.resolve()
    require(not root.exists(), "SCHEMA_PROBE_ROOT", "new capability probe root required")
    require(time.time() < base.DEADLINE - base.STOP_ADMISSION_MARGIN, "SCHEMA_PROBE_DEADLINE", "no late requests")
    predecessor = root.parent / "retention-v3-pilot"
    sources = base.previous.reference.verify_freeze(predecessor / "manifest.json")
    review = json.loads((predecessor / "review_complete.json").read_text())
    require(
        review["assessment_sha256"] == sha256_file(predecessor / "assessment.json")
        and review["all_observed_disagreements_reviewed"] is True,
        "SCHEMA_PROBE_REVIEW",
        "reviewed current failure required",
    )
    for path in (
        Path(__file__).resolve(),
        Path(__file__).with_name("critic_schema_probe_protocol.md").resolve(),
        predecessor / "review_complete.json",
        predecessor / "assessment.json",
    ):
        sources[str(path)] = sha256_file(path)
    frozen = [
        {
            "mode": mode,
            "repeat": repeat,
            "sentinel": f"SCHEMA_VALUE_{repeat}",
            "body": request_body(mode, f"SCHEMA_VALUE_{repeat}"),
        }
        for repeat in (1, 2)
        for mode in MODES
    ]
    for request in frozen:
        request["request_sha256"] = sha256_json(request["body"])
    write_json_atomic(root / "requests_frozen.json", frozen)
    manifest = {
        "purpose": "ACTUAL_ENDPOINT_CAPABILITY_NOT_JUDGE_PERFORMANCE",
        "logical_requests": 8,
        "deadline_utc_epoch": base.DEADLINE,
        "sources": sources,
        "artifacts": {str(root / "requests_frozen.json"): sha256_file(root / "requests_frozen.json")},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "SCHEMA_PROBE_CREDENTIAL", "existing credential unavailable")
    receipts = []
    profile = base.previous.TransportProfile(
        min_interval_seconds=2,
        max_in_flight=1,
        max_attempts=1,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=120,
        max_external_attempts=8,
    )
    with base.previous.PacedRelay(
        profile=profile,
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url="https://inference-api.nvidia.com/v1",
        upstream_model="nvidia/qwen/qwen3.8-27b",
        upstream_api_key=credential,
        timeout_seconds=100,
        expected_generation_parameters=GENERATION,
    ) as relay:
        relay.set_context(base.previous.RelayContext("schema-capability-v1", 0, root / "transport_events.jsonl"))
        for request in frozen:
            record = {key: request[key] for key in ("mode", "repeat", "sentinel", "request_sha256")}
            if time.time() >= base.DEADLINE - base.STOP_ADMISSION_MARGIN:
                record["status"] = "DEADLINE_NOT_SUBMITTED"
            else:
                try:
                    response = requests.post(relay.endpoint + "/chat/completions", json=request["body"], timeout=130)
                    record["http_status"] = response.status_code
                    if response.status_code == 200:
                        record["raw_response"] = response.json()
                        write_json_atomic(
                            root / "raw_received" / f"{request['repeat']}-{request['mode']}.json", record
                        )
                        record["status"] = classify(request["mode"], request["sentinel"], record["raw_response"])
                    else:
                        record["status"] = "HTTP_REJECTED"
                except Exception as exc:  # noqa: BLE001 - do not log credential-bearing error messages
                    record.update(status="TRANSPORT_OR_FORMAT_FAILURE", error_type=type(exc).__name__)
            write_json_atomic(root / "responses" / f"{request['repeat']}-{request['mode']}.json", record)
            receipts.append(record)
            print(json.dumps({k: record.get(k) for k in ("mode", "repeat", "http_status", "status")}), flush=True)
    result = {
        "judge_pairs_scored": 0,
        "reference_changed": False,
        "logical_requests": 8,
        "results": [{k: r.get(k) for k in ("mode", "repeat", "http_status", "status")} for r in receipts],
        "observed_schema_modes": [
            mode
            for mode in MODES[1:]
            if all(r["status"] == "SCHEMA_CONSTRAINT_OBSERVED" for r in receipts if r["mode"] == mode)
        ],
        "limitation": "Two finite synthetic probes per mode are not a guarantee for the full critic schema or for semantic correctness.",
        "response_artifacts": {str(p): sha256_file(p) for p in (root / "responses").glob("*.json")},
    }
    base.previous.reference.verify_freeze(root / "manifest.json")
    write_json_atomic(root / "assessment.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.env_file)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
